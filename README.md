# RSNA-Intracranial-Aneurysm-Detection

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/0ded079f-a8b8-4dad-bcb0-ba29c105d277" />
<img src="https://github.com/Bempong-Sylvester-Obese/RSNA-Intracranial-Aneurysm-Detection/blob/main/aneurysm_trace-ezgif.com-optimize.gif?raw=true" alt="Aneurysm Detection Demo" width="1000" height="500" />

## Overview

Accurate and timely diagnosis is critical in healthcare, especially for potentially life-threatening medical conditions like brain aneurysms. Brain aneurysms (or intracranial aneurysms) are focal dilations in the arteries of the brain that may not show symptoms initially but can be deadly if not diagnosed accurately and treated appropriately.

Experienced radiologists can detect aneurysms on images of the brain, but they can be easily overlooked, particularly when imaging studies are performed for other purposes. Rapid and accurate automated detection of aneurysms on routine brain imaging studies could help prevent devastating outcomes for patients.

## Description

Intracranial aneurysms affect around 3% of the global population. Unfortunately, up to 50% of these aneurysms are only diagnosed after they rupture, which can result in severe illness or death. Worldwide, intracranial aneurysms cause approximately 500,000 deaths each year, and roughly half of the victims are younger than 50.

This project aims to detect and precisely locate intracranial aneurysms across various types of medical images, including CTA, MRA, and T1 post-contrast and T2-weighted MRI. The challenge includes real clinical variation, with data from different institutions, scanners and imaging protocols that will test a model’s ability to generalize.

## Clinical and research disclaimer

**This software is for research and education only.** It is not a medical device and must not be used for clinical diagnosis, treatment, or any decision affecting patient care. Always follow institutional policies and applicable regulations when handling medical imaging data.

## Data layout

| Path | Purpose |
|------|---------|
| `Data/train.csv` | Per-series labels: demographics, modality, 13 vessel-site columns, and `Aneurysm Present` |
| `Data/test.csv` | `SeriesInstanceUID` only (for inference / submission) |
| `Data/train_localizers.csv` | Optional slice-level localizer metadata (coordinates); not required by the default training CLI |
| `Data/series/` | **You create this locally:** one subdirectory per `SeriesInstanceUID` containing DICOM files (`.dcm`). Not committed to git. |

Obtain competition or institutional data according to the RSNA / Kaggle terms you agree to. Do not redistribute restricted imaging without permission.

## Model evaluation (competition metric)

The weighted columnwise macro-averaged ROC AUC is implemented in Python as `rsna_aneurysm.metrics.score` (see also [Notebooks/Mean_Weighted_Columnwise_AUCROC.ipynb](Notebooks/Mean_Weighted_Columnwise_AUCROC.ipynb)).

<img width="450" height="76" alt="Evaluation calculation" src="https://github.com/user-attachments/assets/784a3a20-7205-4c02-9dd5-8000bf58529b" />

## Labels (tabular)

1. ACA - Anterior Communicating Artery  
2. BT - Basilar Tip  
3. LACA - Left Anterior Cerebral Artery  
4. LIICA - Left Infraclinoid Internal Carotid Artery  
5. LMCA - Left Middle Cerebral Artery  
6. LPCA - Left Posterior Communicating Artery  
7. LSICA - Left Supraclinoid Internal Carotid Artery  
8. OPC - Other Posterior Circulation  
9. RACA - Right Anterior Cerebral Artery  
10. RIICA - Right Infraclinoid Internal Carotid Artery  
11. RMCA - Right Middle Cerebral Artery  
12. RPCA - Right Posterior Communicating Artery  
13. RSICA - Right Supraclinoid Internal Carotid Artery  

CSV columns use full vessel names; the training package predicts **13 sites plus `Aneurysm Present`** (14 sigmoid outputs), matching `Data/train.csv`.

## Install and run

Requires **Python 3.10+** (3.11–3.13 recommended).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Environment variables (optional)

| Variable | Meaning |
|----------|---------|
| `RSNA_TRAIN_CSV` | Path to `train.csv` |
| `RSNA_SERIES_DIR` | Root folder of DICOM series |
| `RSNA_LOCALIZERS_CSV` | Optional path to `train_localizers.csv` (logged at train start if present) |

### CLI

```bash
rsna-aneurysm train --train-csv Data/train.csv --series-dir Data/series --epochs 20 --fold 0
```

- **`--synthetic`**: CPU smoke run with random tensors (no DICOM), for CI and debugging.
- **`--strict-paths`**: Drop rows whose `Data/series/<SeriesInstanceUID>` folder is missing.

```bash
rsna-aneurysm eval checkpoints/fold_0_best.pt --train-csv Data/train.csv --series-dir Data/series --fold 0
rsna-aneurysm predict checkpoints/fold_0_best.pt --test-csv Data/test.csv --series-dir Data/series --out submission.csv
rsna-aneurysm metric-self-check
```

Training writes **checkpoints** under `checkpoints/` and **TensorBoard** logs under `runs/` (`tensorboard --logdir runs`).

### Project layout

- `src/rsna_aneurysm/` — library: data loading, 3D CNN, multilabel loss, competition metric, training loop  
- `tests/` — metric unit tests and synthetic training smoke test  
- `Notebooks/` — exploratory EDA and metric reference notebook  
- `AneurysmNet.ipynb` — original Kaggle-oriented notebook (superseded by the package for reproducible runs)

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, limitations, and evaluation notes.

## License

See [LICENSE](LICENSE) (MIT).
