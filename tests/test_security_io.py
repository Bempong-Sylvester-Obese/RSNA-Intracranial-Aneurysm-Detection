"""Security-focused tests for local file and artifact handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from rsna_aneurysm.cli import _load_checkpoint
from rsna_aneurysm.data.dataset import filter_dataframe_with_existing_series, resolve_series_path
from rsna_aneurysm.data.dicom import DICOMVolumeProcessor
from rsna_aneurysm.models.aneurysm_net import EfficientAneurysmNet


def _write_dicom(path: Path, pixel_array: np.ndarray | None = None) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientID = "TEST"
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    if pixel_array is not None:
        array = pixel_array.astype(np.uint16)
        ds.Rows, ds.Columns = array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = array.tobytes()

    ds.save_as(path, write_like_original=False)


def test_load_checkpoint_round_trip(tmp_path: Path) -> None:
    model = EfficientAneurysmNet()
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    checkpoint = _load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert "model_state_dict" in checkpoint
    assert set(checkpoint["model_state_dict"]) == set(model.state_dict())


def test_load_checkpoint_invalid_file_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.pt"
    bad_path.write_text("not a checkpoint", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
        _load_checkpoint(bad_path, torch.device("cpu"))


def test_load_checkpoint_requires_model_state_dict(tmp_path: Path) -> None:
    bad_path = tmp_path / "missing_model_state.pt"
    torch.save({"epoch": 1}, bad_path)

    with pytest.raises(RuntimeError, match="model_state_dict"):
        _load_checkpoint(bad_path, torch.device("cpu"))


@pytest.mark.parametrize(
    "series_id",
    ["", ".", "..", "../escape", "/absolute", "nested/name", r"nested\\name"],
)
def test_resolve_series_path_rejects_unsafe_ids(tmp_path: Path, series_id: str) -> None:
    with pytest.raises(ValueError):
        resolve_series_path(tmp_path, series_id)


def test_filter_dataframe_with_existing_series_drops_missing_and_unsafe(tmp_path: Path) -> None:
    safe_uid = "1.2.3"
    (tmp_path / safe_uid).mkdir()
    df = pd.DataFrame(
        {
            "SeriesInstanceUID": [safe_uid, "../escape", "missing"],
            "label": [1, 0, 0],
        }
    )

    filtered = filter_dataframe_with_existing_series(df, str(tmp_path), label="test")

    assert filtered["SeriesInstanceUID"].tolist() == [safe_uid]


def test_corrupt_dicom_returns_dummy_volume(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    (series_dir / "broken.dcm").write_text("not a dicom", encoding="utf-8")
    processor = DICOMVolumeProcessor(target_size=(4, 8, 8))

    volume = processor.load_dicom_series(str(series_dir))

    assert volume.shape == (4, 8, 8)
    assert processor.stats["dummy"] == 1


def test_missing_pixel_data_returns_dummy_volume(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    _write_dicom(series_dir / "missing_pixel_data.dcm", pixel_array=None)
    processor = DICOMVolumeProcessor(target_size=(4, 8, 8))

    volume = processor.load_dicom_series(str(series_dir))

    assert volume.shape == (4, 8, 8)
    assert processor.stats["dummy"] == 1


def test_candidate_dicom_files_skips_oversized_and_respects_max_slices(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    (series_dir / "0000_big.dcm").write_bytes(b"x" * 50_000)
    _write_dicom(series_dir / "0001_small.dcm", np.arange(64, dtype=np.uint16).reshape(8, 8))
    _write_dicom(series_dir / "0002_small.dcm", np.arange(64, dtype=np.uint16).reshape(8, 8))
    _write_dicom(series_dir / "0003_small.dcm", np.arange(64, dtype=np.uint16).reshape(8, 8))
    processor = DICOMVolumeProcessor(target_size=(2, 8, 8), max_file_size_bytes=10_000)

    files = processor._candidate_dicom_files(str(series_dir))

    assert len(files) == 2
    assert all("small" in Path(file).name for file in files)
