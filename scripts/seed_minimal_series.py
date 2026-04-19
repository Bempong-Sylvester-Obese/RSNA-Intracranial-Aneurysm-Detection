#!/usr/bin/env python3
"""Write minimal DICOM slices under Data/series/<SeriesInstanceUID>/ for local training smoke tests.
Does not replace real RSNA volumes; use after downloading competition data or for pipeline checks.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset


def _write_slice(
    out_path: Path,
    *,
    series_uid: str,
    rows: int,
    cols: int,
    instance_num: int,
) -> None:
    rng = np.random.default_rng(instance_num)
    arr = rng.integers(100, 2000, (rows, cols), dtype=np.uint16)

    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = "SEED"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.InstanceNumber = instance_num
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = arr.tobytes()
    ds.save_as(str(out_path), write_like_original=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("Data/train.csv"),
        help="Path to train.csv (used to pick balanced UIDs)",
    )
    p.add_argument(
        "--series-dir",
        type=Path,
        default=Path("Data/series"),
        help="Output root (one folder per SeriesInstanceUID)",
    )
    p.add_argument(
        "--num-slices",
        type=int,
        default=16,
        help="DICOM files per series (match model depth)",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=128,
        help="Slice height",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=128,
        help="Slice width",
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=4,
        help="How many series with Aneurysm Present=0 and =1 (ignored if --debug-samples set)",
    )
    p.add_argument(
        "--debug-samples",
        type=int,
        default=None,
        help="Match rsna-aneurysm train --debug-samples (same balanced subsample as trainer)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (must match training --seed when using --debug-samples)",
    )
    args = p.parse_args()

    if not args.train_csv.is_file():
        print(f"Missing {args.train_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.train_csv)
    if "Aneurysm Present" not in df.columns or "SeriesInstanceUID" not in df.columns:
        print("train.csv missing required columns", file=sys.stderr)
        return 1

    df = df.dropna(subset=["Aneurysm Present", "SeriesInstanceUID"])
    if args.debug_samples is not None:
        pos = df[df["Aneurysm Present"] == 1]
        neg = df[df["Aneurysm Present"] == 0]
        n = min(args.debug_samples // 2, len(pos), len(neg))
        if n < 1:
            print("Not enough pos/neg rows for debug-samples", file=sys.stderr)
            return 1
        picked = pd.concat(
            [
                pos.sample(n=n, random_state=args.seed),
                neg.sample(n=n, random_state=args.seed),
            ]
        )
        uids = picked["SeriesInstanceUID"].astype(str).tolist()
    else:
        neg = df[df["Aneurysm Present"] == 0]["SeriesInstanceUID"].astype(str).head(args.per_class)
        pos = df[df["Aneurysm Present"] == 1]["SeriesInstanceUID"].astype(str).head(args.per_class)
        uids = list(neg) + list(pos)
    if len(uids) < 2:
        print("Not enough rows for balanced seed", file=sys.stderr)
        return 1

    args.series_dir.mkdir(parents=True, exist_ok=True)
    for uid in uids:
        folder = args.series_dir / uid
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.num_slices + 1):
            out = folder / f"slice_{i:04d}.dcm"
            _write_slice(out, series_uid=uid, rows=args.rows, cols=args.cols, instance_num=i)

    print(f"Wrote {args.num_slices} slices each for {len(uids)} series under {args.series_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
