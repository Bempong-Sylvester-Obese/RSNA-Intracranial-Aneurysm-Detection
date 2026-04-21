"""Series upload / inspection / prediction endpoints."""

from __future__ import annotations

import io
import logging
import re
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from PIL import Image

from rsna_aneurysm.inference.gradcam import GradCAM3D, gradcam_to_rgba_slice
from rsna_aneurysm.labels import LABEL_COLUMNS, PRESENCE_COL
from webapp.backend.app.routes.schemas import (
    PredictionResponse,
    SeriesMetadata,
    UploadResponse,
    VesselScore,
)
from webapp.backend.app.services.state import AppState
from webapp.backend.app.services.storage import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/series", tags=["series"])


_VESSEL_KEY_RE = re.compile(r"[^a-z0-9]+")


def _vessel_key(label: str) -> str:
    """Stable URL-safe key for a vessel column."""
    key = _VESSEL_KEY_RE.sub("_", label.lower()).strip("_")
    return key or "vessel"


_KEY_TO_INDEX: dict[str, int] = {
    _vessel_key(col): idx for idx, col in enumerate(LABEL_COLUMNS)
}
_INDEX_TO_KEY: dict[int, str] = {idx: key for key, idx in _KEY_TO_INDEX.items()}
_PRESENCE_INDEX: int = LABEL_COLUMNS.index(PRESENCE_COL)


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a DICOM series",
    description=(
        "Accepts either a single `.zip` archive or multiple `.dcm` files. "
        "Returns a generated `series_id` used for subsequent viewer/predict calls."
    ),
)
async def upload_series(
    files: List[UploadFile] = File(
        ..., description="One .zip archive or several .dcm files."
    ),
) -> UploadResponse:
    state = AppState.get()

    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files uploaded.")

    # Single zip archive path.
    if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
        upload = files[0]
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
            total = 0
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > state.settings.max_upload_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        "Upload exceeds total size limit.",
                    )
                tmp.write(chunk)
            tmp.flush()
            try:
                stored = state.storage.ingest_zip(Path(tmp.name))
            except StorageError as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return UploadResponse(series_id=stored.series_id, num_files=stored.num_files)

    # Multi-file DICOM upload path.
    payload: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        data = await upload.read()
        total += len(data)
        if total > state.settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "Upload exceeds total size limit.",
            )
        payload.append((upload.filename or "unnamed.dcm", data))

    try:
        stored = state.storage.ingest_files(payload)
    except StorageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return UploadResponse(series_id=stored.series_id, num_files=stored.num_files)


def _read_modality(series_path: Path) -> Optional[str]:
    """Best-effort modality read from the first DICOM we can parse."""
    for entry in sorted(series_path.iterdir()):
        if not entry.is_file() or not entry.name.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(entry, force=False, stop_before_pixels=True)
        except Exception:  # noqa: BLE001
            continue
        modality = getattr(ds, "Modality", None)
        if modality:
            return str(modality)
    return None


@router.get("/{series_id}/metadata", response_model=SeriesMetadata)
async def series_metadata(series_id: str) -> SeriesMetadata:
    state = AppState.get()
    try:
        path = state.storage.resolve(series_id)
    except StorageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    entry = state.get_or_load_volume(series_id)
    volume = entry.volume
    d, h, w = volume.shape
    num_files = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")

    return SeriesMetadata(
        series_id=series_id,
        num_files=num_files,
        num_slices=int(d),
        height=int(h),
        width=int(w),
        modality=_read_modality(path),
    )


def _slice_to_png(slice_2d: np.ndarray) -> bytes:
    """Encode a 2D ``[0, 1]`` float slice as a greyscale PNG."""
    arr = np.clip(slice_2d, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _overlay_png(slice_2d: np.ndarray, cam_slice: np.ndarray) -> bytes:
    """Alpha-blend a Grad-CAM heatmap onto a greyscale slice."""
    base = np.clip(slice_2d, 0.0, 1.0)
    base_u8 = (base * 255.0).astype(np.uint8)
    base_rgba = np.stack([base_u8, base_u8, base_u8, np.full_like(base_u8, 255)], axis=-1)

    heat_rgba = gradcam_to_rgba_slice(cam_slice)

    base_img = Image.fromarray(base_rgba, mode="RGBA")
    heat_img = Image.fromarray(heat_rgba, mode="RGBA")
    composed = Image.alpha_composite(base_img, heat_img)

    buf = io.BytesIO()
    composed.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _resolve_class_index(vessel: Optional[str]) -> int:
    if vessel is None:
        return _PRESENCE_INDEX
    idx = _KEY_TO_INDEX.get(vessel)
    if idx is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown vessel key '{vessel}'. Expected one of: {sorted(_KEY_TO_INDEX)}",
        )
    return idx


def _get_cam(state: AppState, series_id: str, class_index: int) -> np.ndarray:
    entry = state.get_or_load_volume(series_id)
    cached = entry.cams.get(class_index)
    if cached is not None:
        return cached

    predictor = state.predictor
    with GradCAM3D(predictor.model) as cam:
        cam_vol = cam.compute(entry.volume, class_index=class_index, device=predictor.device)
    state.set_cam(series_id, class_index, cam_vol)
    return cam_vol


@router.get(
    "/{series_id}/slice/{index}.png",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def series_slice(
    series_id: str,
    index: int,
    overlay: Optional[str] = Query(None, description="Set to 'gradcam' to overlay CAM."),
    vessel: Optional[str] = Query(
        None,
        description="Vessel key for Grad-CAM; defaults to 'Aneurysm Present'.",
    ),
) -> Response:
    state = AppState.get()
    try:
        state.storage.resolve(series_id)
    except StorageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    entry = state.get_or_load_volume(series_id)
    volume = entry.volume
    if not 0 <= index < volume.shape[0]:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Slice index {index} out of range [0, {volume.shape[0]}).",
        )

    slice_2d = volume[index]

    if overlay == "gradcam":
        class_index = _resolve_class_index(vessel)
        cam_vol = _get_cam(state, series_id, class_index)
        png = _overlay_png(slice_2d, cam_vol[index])
    elif overlay in (None, "", "none"):
        png = _slice_to_png(slice_2d)
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown overlay '{overlay}'. Expected 'gradcam' or unset.",
        )

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.post("/{series_id}/predict", response_model=PredictionResponse)
async def predict_series(series_id: str) -> PredictionResponse:
    state = AppState.get()
    try:
        state.storage.resolve(series_id)
    except StorageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    entry = state.get_or_load_volume(series_id)
    if entry.result is None:
        result = state.predictor.predict_volume(entry.volume)
        state.set_result(series_id, result)
    else:
        result = entry.result

    vessels: list[VesselScore] = []
    for idx, col in enumerate(LABEL_COLUMNS):
        if idx == _PRESENCE_INDEX:
            continue
        vessels.append(
            VesselScore(
                key=_INDEX_TO_KEY[idx],
                label=col,
                probability=float(result.labels[col]),
                class_index=idx,
            )
        )

    return PredictionResponse(
        presence=float(result.labels[PRESENCE_COL]),
        presence_index=_PRESENCE_INDEX,
        vessels=vessels,
        gradcam_ready=True,
    )


@router.delete("/{series_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_series(series_id: str) -> Response:
    state = AppState.get()
    try:
        state.storage.delete(series_id)
    except StorageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileNotFoundError:
        pass
    state.drop(series_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
