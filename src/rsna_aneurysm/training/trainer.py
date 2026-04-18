"""Training loop, validation, and competition metric on held-out fold."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rsna_aneurysm.data.dataset import (
    AneurysmVolumeDataset,
    SyntheticVolumeDataset,
    filter_dataframe_with_existing_series,
)
from rsna_aneurysm.data.dicom import DICOMVolumeProcessor
from rsna_aneurysm.device import pick_device, should_pin_memory
from rsna_aneurysm.labels import ID_COL, LABEL_COLUMNS, NUM_LABELS, PRESENCE_COL
from rsna_aneurysm.metrics.competition import weighted_multilabel_auc
from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet
from rsna_aneurysm.training.losses import MultiLabelFocalLoss

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Any,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    processor = DICOMVolumeProcessor(target_size=cfg.target_size)

    if cfg.synthetic:
        train_ds = SyntheticVolumeDataset(
            cfg.synthetic_samples // 2,
            cfg.target_size,
            NUM_LABELS,
            seed=cfg.seed,
        )
        val_ds = SyntheticVolumeDataset(
            max(8, cfg.synthetic_samples // 4),
            cfg.target_size,
            NUM_LABELS,
            seed=cfg.seed + 1,
        )
        pin = should_pin_memory(device)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )
        return train_loader, val_loader

    sdir = str(cfg.series_dir)
    if cfg.strict_paths:
        train_df = filter_dataframe_with_existing_series(train_df, sdir, label="train")
        val_df = filter_dataframe_with_existing_series(val_df, sdir, label="val")
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(
                "strict_paths: no rows left after requiring existing series folders under "
                f"{sdir}. Run scripts/seed_minimal_series.py or add DICOM data."
            )

    train_ds = AneurysmVolumeDataset(
        train_df,
        sdir,
        processor,
        cfg.target_size,
        mode="train",
    )
    val_ds = AneurysmVolumeDataset(
        val_df,
        sdir,
        processor,
        cfg.target_size,
        mode="val",
    )

    targets = train_df[PRESENCE_COL].astype(float).tolist()

    class_counts = Counter(targets)
    if len(class_counts) < 2:
        weights = [1.0] * len(targets)
    else:
        weights = [len(targets) / (len(class_counts) * class_counts[t]) for t in targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    pin = should_pin_memory(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )
    return train_loader, val_loader


def _pos_weight_tensor(train_df: pd.DataFrame) -> torch.Tensor:
    """Per-class positive weights for BCE (inverse frequency)."""
    weights = []
    for col in LABEL_COLUMNS:
        pos = float(train_df[col].sum())
        neg = len(train_df) - pos
        w = neg / max(pos, 1.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    running = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="train", leave=False):
        vol = batch["label"].shape[0]
        if vol == 0:
            continue
        x = batch["volume"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        running += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    if not all_probs:
        return {"loss": 0.0, "auc_macro": 0.0}

    y_score = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    try:
        aucs = roc_auc_score(y_true, y_score, average=None)
        auc_macro = float(np.nanmean(aucs))
    except ValueError:
        auc_macro = 0.0

    return {"loss": running / len(loader), "auc_macro": auc_macro}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["volume"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        running += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())

    y_score = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    try:
        aucs = roc_auc_score(y_true, y_score, average=None)
        auc_macro = float(np.nanmean(aucs))
        comp_auc = weighted_multilabel_auc(y_true, y_score, class_weights=None)
    except ValueError:
        auc_macro = 0.0
        comp_auc = 0.0

    preds = (y_score >= 0.5).astype(int)
    acc = accuracy_score(y_true.flatten(), preds.flatten())
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    return {
        "loss": running / max(len(loader), 1),
        "auc_macro": auc_macro,
        "competition_auc": comp_auc,
        "acc": acc,
        "f1_macro": f1,
    }


def run_training(cfg: Any) -> list[dict[str, Any]]:
    cfg.resolve_paths()
    set_seed(cfg.seed)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)

    if cfg.localizers_csv and Path(cfg.localizers_csv).is_file():
        loc = pd.read_csv(cfg.localizers_csv)
        logger.info("Optional localizers CSV loaded: %s rows", len(loc))

    device = pick_device()
    logger.info("Device: %s", device)

    if cfg.synthetic:
        n = max(cfg.synthetic_samples, 16)
        rng = np.random.default_rng(cfg.seed)
        mat = rng.integers(0, 2, size=(n, len(LABEL_COLUMNS)))
        train_df = pd.DataFrame(mat, columns=list(LABEL_COLUMNS))
        train_df[ID_COL] = [f"syn_{i}" for i in range(len(train_df))]
        # Ensure stratified K-fold can split on presence
        half = n // 2
        train_df.loc[train_df.index[:half], PRESENCE_COL] = 0
        train_df.loc[train_df.index[half:], PRESENCE_COL] = 1
    else:
        train_df = pd.read_csv(cfg.train_csv)
        train_df = train_df.dropna(subset=list(LABEL_COLUMNS) + [ID_COL])

    if cfg.debug_samples:
        pos = train_df[train_df[PRESENCE_COL] == 1]
        neg = train_df[train_df[PRESENCE_COL] == 0]
        n = min(cfg.debug_samples // 2, len(pos), len(neg))
        train_df = pd.concat(
            [pos.sample(n=n, random_state=cfg.seed), neg.sample(n=n, random_state=cfg.seed)]
        ).reset_index(drop=True)

    y_strat = train_df[PRESENCE_COL].values
    min_class = int(train_df[PRESENCE_COL].value_counts().min())
    if min_class < 2:
        raise ValueError(
            "Stratified K-fold needs at least 2 samples per class; "
            "reduce --debug-samples or use more data."
        )
    n_splits_eff = min(cfg.n_folds, min_class)
    if n_splits_eff < cfg.n_folds:
        logger.info(
            "Using n_splits=%s (min class count=%s, requested %s)",
            n_splits_eff,
            min_class,
            cfg.n_folds,
        )
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=cfg.seed)
    fold_results: list[dict[str, Any]] = []

    writer: SummaryWriter | None = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_strat)):
        if cfg.current_fold >= 0 and fold != cfg.current_fold:
            continue

        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)

        train_loader, val_loader = _build_loaders(tr, va, cfg, device)

        model = EfficientAneurysmNet(num_classes=NUM_LABELS).to(device)
        pos_w = _pos_weight_tensor(tr).to(device)
        criterion = MultiLabelFocalLoss(alpha=1.0, gamma=2.0, pos_weight=pos_w)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=cfg.scheduler_patience,
            factor=0.5,
        )

        run_name = f"fold{fold}_{cfg.seed}"
        writer = SummaryWriter(log_dir=str(cfg.runs_dir / run_name))

        best_auc = -1.0
        patience_c = 0
        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_competition_auc": [],
        }

        for epoch in range(cfg.epochs):
            tr_m = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                gradient_clip=cfg.gradient_clip,
            )
            va_m = validate(model, val_loader, criterion, device)
            scheduler.step(va_m["loss"])

            history["train_loss"].append(tr_m["loss"])
            history["val_loss"].append(va_m["loss"])
            history["val_competition_auc"].append(va_m["competition_auc"])

            writer.add_scalar("loss/train", tr_m["loss"], epoch)
            writer.add_scalar("loss/val", va_m["loss"], epoch)
            writer.add_scalar("auc/competition_val", va_m["competition_auc"], epoch)
            writer.add_scalar("auc/macro_val", va_m["auc_macro"], epoch)

            logger.info(
                "Epoch %s | train_loss=%.4f val_loss=%.4f comp_auc=%.4f",
                epoch + 1,
                tr_m["loss"],
                va_m["loss"],
                va_m["competition_auc"],
            )

            if va_m["competition_auc"] > best_auc:
                best_auc = va_m["competition_auc"]
                patience_c = 0
                ckpt = cfg.checkpoint_dir / f"fold_{fold}_best.pt"
                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_competition_auc": best_auc,
                        "config": {
                            "target_size": cfg.target_size,
                            "num_labels": NUM_LABELS,
                        },
                    },
                    ckpt,
                )
            else:
                patience_c += 1
                if patience_c >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %s", epoch + 1)
                    break

        fold_results.append({"fold": fold, "best_competition_auc": best_auc, "history": history})
        if writer:
            writer.close()

        if cfg.current_fold >= 0:
            break

    out = cfg.checkpoint_dir / "training_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            [
                {k: v for k, v in r.items() if k != "history"}
                | {"history": {kk: vv for kk, vv in r.get("history", {}).items()}}
                for r in fold_results
            ],
            f,
            indent=2,
        )
    return fold_results


@torch.no_grad()
def predict_dataframe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return submission-like frame with probabilities per label column."""
    model.eval()
    rows: list[dict[str, Any]] = []
    all_logits: list[np.ndarray] = []

    for batch in tqdm(loader, desc="predict", leave=False):
        x = batch["volume"].to(device, non_blocking=True)
        ids = batch["series_id"]
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_logits.append(probs)
        for i, sid in enumerate(ids):
            row = {ID_COL: sid}
            for j, col in enumerate(LABEL_COLUMNS):
                row[col] = float(probs[i, j])
            rows.append(row)

    return pd.DataFrame(rows), np.vstack(all_logits)
