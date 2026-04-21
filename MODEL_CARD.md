# Model card: RSNA intracranial aneurysm (multilabel)

## Model details

- **Architecture:** `EfficientAneurysmNet` — lightweight 3D CNN with global average pooling and a linear head.
- **Outputs:** 14 independent sigmoid logits corresponding to the 13 vessel-site columns and `Aneurysm Present` in `Data/train.csv`.
- **Training objective:** Multilabel focal loss on logits with optional per-class `pos_weight` (inverse frequency over the training fold).
- **Primary offline metric:** Weighted columnwise macro-averaged ROC AUC (`rsna_aneurysm.metrics.weighted_multilabel_auc` / `score`), aligned with [Notebooks/Mean_Weighted_Columnwise_AUCROC.ipynb](Notebooks/Mean_Weighted_Columnwise_AUCROC.ipynb).

## Intended use

Research and benchmarking on the RSNA-style tabular labels and volumetric imaging pipeline. **Not for clinical use.**

## Limitations

- **Localization:** The default model performs **series-level multilabel classification** only. It does not output voxel-level segmentations or slice coordinates; optional `train_localizers.csv` is not consumed by the loss (only logged if `RSNA_LOCALIZERS_CSV` is set).
- **DICOM handling:** Missing or invalid series fall back to a synthetic volume in non-strict mode (`strict_paths=False`), which can add label noise. Prefer `strict_paths=True` for real training when series folders are complete.
- **Class imbalance:** Sampler balances by `Aneurysm Present`; rare sites may still be underrepresented.
- **Generalization:** Performance depends on scanner, protocol, and institution mix; external validation is required before any non-research use.

## Metrics and evaluation

- Validation logs **competition-style AUC** (uniform class weights by default) and TensorBoard scalars under `runs/`.
- For held-out evaluation with the official `score()` API, build `solution` and `submission` DataFrames with the same label columns (excluding the row id column from the metric input as implemented in `metrics/competition.py`).

## Ethical and data notes

- Use imaging data only under the license or competition rules you accepted.
- Avoid deploying outputs as diagnostic advice; this is experimental software.

## Security notes (research tooling)

- Treat **checkpoints** and **local DICOM folders** as trusted inputs from your environment. The CLI loads checkpoints with `weights_only=True` when available and validates the expected dict shape, but you should still only use artifacts you produced or fully trust.
- Treat **CSVs** as trusted metadata; the pipeline validates required columns but is not a general-purpose spreadsheet sanitizer.
- `SeriesInstanceUID` values are constrained to safe subpaths under your configured series root (see `resolve_series_path`).

## How to reproduce training

```bash
pip install -e ".[dev]"
rsna-aneurysm train --train-csv Data/train.csv --series-dir Data/series --fold 0 --strict-paths
```

Smoke test (no imaging data):

```bash
pytest tests/ -v
```
