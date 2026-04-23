# RSNA Aneurysm Clinician Webapp (MVP)

This web application is **for research and education only**. It is **not** a medical device and must **not** be used for clinical diagnosis, treatment, or any decision affecting patient care. See the root [README.md](../README.md) for the full project disclaimer.

## What it does

- Upload a DICOM series: a single `.zip`, a **folder** of `.dcm` files (Chromium / Safari “Choose DICOM folder”), a multi-file selection, or drag-and-drop.
- Browse slices in a Cornerstone-based stack viewer (PNG tiles from the API).
- Run the bundled `EfficientAneurysmNet` checkpoint (optional) to view per-vessel probabilities and toggle Grad-CAM overlays by vessel.

## Prerequisites

- Python 3.10+ with the package installed in editable mode including webapp extras.
- Node.js 20+ (for the Vite frontend).

## Setup

From the **repository root**:

```bash
python -m pip install -e ".[dev,webapp]"
cd webapp/frontend && npm install && cd ../..
```

### Model checkpoint (optional)

Point `RSNA_CHECKPOINT` at a `.pt` file that contains `model_state_dict` (same format as the training CLI). If unset or missing, the API loads an **untrained** model so you can still exercise the UI.

```bash
export RSNA_CHECKPOINT=/path/to/checkpoint.pt
```

Other useful variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RSNA_STORAGE_DIR` | `webapp/backend/storage` | Uploaded series extraction root |
| `RSNA_MAX_UPLOAD_MB` | `512` | Total upload budget per request |
| `RSNA_CORS_ORIGINS` | `http://localhost:5173,...` | Allowed browser origins |

## Run (development)

**Terminal 1 — API**

```bash
cd /path/to/RSNA-Intracranial-Aneurysm-Detection
uvicorn webapp.backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend**

```bash
cd webapp/frontend
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to `http://localhost:8000` (override with `VITE_BACKEND_URL` if needed).

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness + checkpoint / device info |
| `POST` | `/api/series` | Multipart: one `.zip` or many `.dcm` files |
| `GET` | `/api/series/{id}/metadata` | Slice counts, shape, modality |
| `GET` | `/api/series/{id}/slice/{i}.png` | Greyscale tile; `?overlay=gradcam&vessel=<key>` for heatmap |
| `POST` | `/api/series/{id}/predict` | JSON probabilities + vessel list |
| `DELETE` | `/api/series/{id}` | Remove cached series from storage |

Vessel keys are derived from label names (lowercase, non-alphanumeric → `_`). The presence logit uses the key for `Aneurysm Present` (`aneurysm_present`).

## Production notes

- Serve the FastAPI app behind HTTPS and add authentication, audit logging, and data retention policies before any real PHI handling.
- This MVP stores uploads only on local disk under `RSNA_STORAGE_DIR`; add TTL cleanup or external object storage for deployments.
