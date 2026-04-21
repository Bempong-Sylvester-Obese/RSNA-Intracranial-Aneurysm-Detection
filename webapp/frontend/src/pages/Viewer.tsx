import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CornerstoneStackViewport } from "@/components/CornerstoneStackViewport";
import { ProbabilitiesPanel } from "@/components/ProbabilitiesPanel";
import {
  PRESENCE_VESSEL_KEY,
  fetchMetadata,
  runPredict,
  type PredictionResponse,
  type SeriesMetadata,
} from "@/lib/api";

export default function Viewer() {
  const { seriesId = "" } = useParams<{ seriesId: string }>();
  const [meta, setMeta] = useState<SeriesMetadata | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);
  const [gradcamEnabled, setGradcamEnabled] = useState(false);
  const [selectedVesselKey, setSelectedVesselKey] = useState<string | null>(PRESENCE_VESSEL_KEY);

  useEffect(() => {
    if (!seriesId) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const m = await fetchMetadata(seriesId);
        if (!cancelled) {
          setMeta(m);
          setMetaError(null);
          setSliceIndex(0);
        }
      } catch (e) {
        if (!cancelled) {
          setMetaError(e instanceof Error ? e.message : "Failed to load metadata.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [seriesId]);

  useEffect(() => {
    if (!seriesId) {
      return;
    }
    let cancelled = false;
    setPredLoading(true);
    setPredError(null);
    void (async () => {
      try {
        const p = await runPredict(seriesId);
        if (!cancelled) {
          setPrediction(p);
        }
      } catch (e) {
        if (!cancelled) {
          setPredError(e instanceof Error ? e.message : "Prediction failed.");
        }
      } finally {
        if (!cancelled) {
          setPredLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [seriesId]);

  const overlay = useMemo(() => {
    if (!gradcamEnabled) {
      return {};
    }
    if (!selectedVesselKey) {
      return { overlay: "gradcam" as const };
    }
    return { overlay: "gradcam" as const, vessel: selectedVesselKey };
  }, [gradcamEnabled, selectedVesselKey]);

  if (!seriesId) {
    return null;
  }

  return (
    <div className="flex h-[calc(100vh-52px)] min-h-[560px] flex-col gap-3 px-3 py-3 lg:flex-row">
      <div className="flex min-h-0 flex-1 flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-400">
          <div className="flex flex-wrap items-center gap-3">
            <Link to="/" className="text-brand-500 hover:text-brand-400">
              ← Upload
            </Link>
            <span className="font-mono text-xs text-slate-500">{seriesId}</span>
          </div>
          {meta && (
            <span>
              Modality {meta.modality ?? "—"} · {meta.num_slices} slices processed ·{" "}
              {meta.num_files} DICOM files
            </span>
          )}
        </div>

        {metaError && (
          <p className="rounded border border-red-900/50 bg-red-950/30 px-3 py-2 text-sm text-red-200">
            {metaError}
          </p>
        )}

        {meta && meta.num_slices > 0 && (
          <>
            <div className="panel min-h-0 flex-1 overflow-hidden p-2">
              <CornerstoneStackViewport
                seriesId={seriesId}
                numSlices={meta.num_slices}
                sliceIndex={sliceIndex}
                onSliceChange={setSliceIndex}
                overlay={overlay}
              />
            </div>
            <div className="flex items-center gap-3 px-1 text-xs text-slate-500">
              <label className="flex flex-1 items-center gap-2">
                <span className="shrink-0 text-slate-400">Slice {sliceIndex + 1}</span>
                <input
                  type="range"
                  min={0}
                  max={meta.num_slices - 1}
                  value={sliceIndex}
                  onChange={(e) => setSliceIndex(Number(e.target.value))}
                  className="w-full accent-brand-500"
                />
                <span className="shrink-0 tabular-nums text-slate-400">{meta.num_slices}</span>
              </label>
            </div>
          </>
        )}
      </div>

      <aside className="flex w-full shrink-0 flex-col lg:w-[380px]">
        <ProbabilitiesPanel
          prediction={prediction}
          loading={predLoading}
          error={predError}
          selectedVesselKey={selectedVesselKey}
          onSelectVessel={(key) => setSelectedVesselKey(key)}
          gradcamEnabled={gradcamEnabled}
          onToggleGradcam={setGradcamEnabled}
        />
      </aside>
    </div>
  );
}
