"""Command-line interface for training, evaluation, and submission export."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
import typer
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from rsna_aneurysm.config import TrainConfig
from rsna_aneurysm.data.dataset import AneurysmVolumeDataset
from rsna_aneurysm.data.dicom import DICOMVolumeProcessor
from rsna_aneurysm.device import pick_device
from rsna_aneurysm.labels import ID_COL, LABEL_COLUMNS, NUM_LABELS, PRESENCE_COL
from rsna_aneurysm.metrics.competition import score as competition_score
from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet
from rsna_aneurysm.training.trainer import (
    _build_loaders,  # noqa: SLF001
    predict_dataframe,
    run_training,
    validate,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load_checkpoint(path: Path, device: torch.device):
    """Load full checkpoint dict (PyTorch 2.6+ defaults weights_only=True)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def train(
    train_csv: Optional[Path] = typer.Option(None, help="Path to train.csv"),
    series_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing one subfolder per SeriesInstanceUID",
    ),
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(4),
    fold: int = typer.Option(0, help="Which fold to run (0..n_folds-1); set -1 for all folds"),
    synthetic: bool = typer.Option(False, help="Smoke train on synthetic tensors (no DICOM)"),
    strict_paths: bool = typer.Option(
        False, help="Drop rows whose series folder is missing (recommended for real training)",
    ),
    debug_samples: Optional[int] = typer.Option(None, help="Limit to balanced subset size"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints")),
    runs_dir: Path = typer.Option(Path("runs")),
    seed: int = typer.Option(42),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train multilabel model with stratified K-fold; logs TensorBoard under runs/."""
    _setup_logging(verbose)
    cfg = TrainConfig(
        train_csv=train_csv or TrainConfig().train_csv,
        series_dir=series_dir or TrainConfig().series_dir,
        epochs=epochs,
        batch_size=batch_size,
        current_fold=fold,
        synthetic=synthetic,
        strict_paths=strict_paths,
        debug_samples=debug_samples,
        checkpoint_dir=checkpoint_dir,
        runs_dir=runs_dir,
        seed=seed,
    )
    if fold < 0:
        cfg.current_fold = -1
    run_training(cfg)


@app.command("eval")
def eval_model(
    checkpoint: Path = typer.Argument(..., exists=True),
    train_csv: Optional[Path] = typer.Option(None),
    series_dir: Optional[Path] = typer.Option(None),
    fold: int = typer.Option(0),
    batch_size: int = typer.Option(4),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run validation metrics on one fold using a saved checkpoint."""
    _setup_logging(verbose)
    cfg = TrainConfig(
        train_csv=train_csv or TrainConfig().train_csv,
        series_dir=series_dir or TrainConfig().series_dir,
        batch_size=batch_size,
        current_fold=fold,
        synthetic=False,
    )
    cfg.resolve_paths()
    train_df = pd.read_csv(cfg.train_csv).dropna(subset=list(LABEL_COLUMNS) + [ID_COL])
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    y = train_df[PRESENCE_COL].values
    splits = list(skf.split(train_df, y))
    tr_idx, va_idx = splits[fold]
    tr = train_df.iloc[tr_idx].reset_index(drop=True)
    va = train_df.iloc[va_idx].reset_index(drop=True)

    device = pick_device()
    train_loader, val_loader = _build_loaders(tr, va, cfg, device)

    ckpt = _load_checkpoint(checkpoint, device)
    model = EfficientAneurysmNet(num_classes=NUM_LABELS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    from rsna_aneurysm.training.losses import MultiLabelFocalLoss

    pos_w = torch.ones(NUM_LABELS, device=device)
    criterion = MultiLabelFocalLoss(pos_weight=pos_w)

    m = validate(model, val_loader, criterion, device)
    typer.echo(f"Validation: {m}")


@app.command()
def predict(
    checkpoint: Path = typer.Argument(..., exists=True),
    test_csv: Path = typer.Option(Path("Data/test.csv"), exists=True),
    series_dir: Optional[Path] = typer.Option(None),
    out: Path = typer.Option(Path("submission.csv")),
    batch_size: int = typer.Option(4),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Write submission CSV with per-label probabilities for test SeriesInstanceUIDs."""
    _setup_logging(verbose)
    cfg = TrainConfig(series_dir=series_dir or TrainConfig().series_dir, batch_size=batch_size)
    cfg.resolve_paths()
    test_df = pd.read_csv(test_csv)
    for c in LABEL_COLUMNS:
        test_df[c] = 0
    device = pick_device()
    processor = DICOMVolumeProcessor(target_size=cfg.target_size)
    ds = AneurysmVolumeDataset(
        test_df,
        str(cfg.series_dir),
        processor,
        cfg.target_size,
        mode="val",
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    ckpt = _load_checkpoint(checkpoint, device)
    model = EfficientAneurysmNet(num_classes=NUM_LABELS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    df_out, _ = predict_dataframe(model, loader, device)
    df_out.to_csv(out, index=False)
    typer.echo(f"Wrote {out}")


@app.command("metric-self-check")
def metric_self_check(verbose: bool = typer.Option(False, "-v")) -> None:
    """Run built-in competition metric examples (from notebook docstrings)."""
    _setup_logging(verbose)
    solution = pd.DataFrame(
        {"id": [1, 2, 3], "cat": [1, 0, 1], "dog": [0, 1, 1]}
    )
    submission = pd.DataFrame(
        {"id": [1, 2, 3], "cat": [0.9, 0.2, 0.8], "dog": [0.1, 0.7, 0.6]}
    )
    s = competition_score(solution.copy(), submission.copy(), "id")
    typer.echo(f"Perfect-example score: {s:.6f} (expect 1.0)")


if __name__ == "__main__":
    app()
