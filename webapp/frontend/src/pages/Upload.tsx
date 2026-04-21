import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { uploadSeries } from "@/lib/api";

export default function Upload() {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  const onFiles = useCallback(
    async (list: FileList | File[]) => {
      const files = Array.from(list);
      if (files.length === 0) {
        return;
      }
      setBusy(true);
      setError(null);
      try {
        const res = await uploadSeries(files);
        navigate(`/viewer/${res.series_id}`);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setBusy(false);
      }
    },
    [navigate],
  );

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-10">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50">
          RSNA intracranial aneurysm — review workspace
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Upload a CTA, MRA, or MRI DICOM series as a <code className="text-slate-300">.zip</code>{" "}
          archive or as individual <code className="text-slate-300">.dcm</code> files. Slices are
          shown in a Cornerstone stack viewer with optional Grad-CAM overlays for each label.
        </p>
      </header>

      <div
        className={clsx(
          "panel cursor-pointer border-2 border-dashed px-6 py-14 text-center transition-colors",
          drag ? "border-brand-500 bg-slate-900" : "border-slate-700 hover:border-slate-600",
          busy && "pointer-events-none opacity-60",
        )}
        onDragEnter={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          void onFiles(e.dataTransfer.files);
        }}
        onClick={() => document.getElementById("file-input")?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            document.getElementById("file-input")?.click();
          }
        }}
      >
        <p className="text-slate-200">
          {busy ? "Uploading…" : "Drop files here or click to browse"}
        </p>
        <p className="mt-2 text-xs text-slate-500">
          One .zip of DICOMs, or multiple .dcm files (max size set on the server).
        </p>
        <input
          id="file-input"
          type="file"
          className="hidden"
          multiple
          accept=".zip,.dcm,application/zip"
          onChange={(e) => {
            const f = e.target.files;
            if (f) {
              void onFiles(f);
            }
            e.target.value = "";
          }}
        />
      </div>

      {error && (
        <p className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-200">
          {error}
        </p>
      )}
    </main>
  );
}
