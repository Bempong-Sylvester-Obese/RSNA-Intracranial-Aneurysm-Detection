import clsx from "clsx";
import type { PredictionResponse, VesselScore } from "@/lib/api";
import { PRESENCE_VESSEL_KEY } from "@/lib/api";

type Props = {
  prediction: PredictionResponse | null;
  loading: boolean;
  error: string | null;
  selectedVesselKey: string | null;
  onSelectVessel: (key: string) => void;
  gradcamEnabled: boolean;
  onToggleGradcam: (on: boolean) => void;
};

function Bar({ value }: { value: number }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 1000) / 10;
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
      <div
        className="h-full rounded-full bg-brand-500 transition-[width]"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function VesselRow({
  row,
  active,
  onClick,
}: {
  row: VesselScore;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "w-full rounded-md border px-3 py-2 text-left text-sm transition-colors",
        active
          ? "border-brand-500 bg-slate-800"
          : "border-transparent hover:border-slate-700 hover:bg-slate-800/60",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="line-clamp-2 text-slate-200">{row.label}</span>
        <span className="shrink-0 tabular-nums text-slate-400">
          {(row.probability * 100).toFixed(1)}%
        </span>
      </div>
      <div className="mt-2">
        <Bar value={row.probability} />
      </div>
    </button>
  );
}

export function ProbabilitiesPanel({
  prediction,
  loading,
  error,
  selectedVesselKey,
  onSelectVessel,
  gradcamEnabled,
  onToggleGradcam,
}: Props) {
  return (
    <div className="panel flex h-full min-h-0 flex-col p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-100">Model output</h2>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-900"
            checked={gradcamEnabled}
            onChange={(e) => onToggleGradcam(e.target.checked)}
            disabled={!prediction?.gradcam_ready}
          />
          Grad-CAM overlay
        </label>
      </div>

      {loading && <p className="text-sm text-slate-400">Running inference…</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      {prediction && !loading && (
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
          <button
            type="button"
            onClick={() => onSelectVessel(PRESENCE_VESSEL_KEY)}
            className={clsx(
              "w-full rounded-md border px-3 py-2 text-left transition-colors",
              selectedVesselKey === PRESENCE_VESSEL_KEY
                ? "border-brand-500 bg-slate-800"
                : "border-transparent hover:border-slate-700 hover:bg-slate-800/60",
            )}
          >
            <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
              <span>Aneurysm present (Grad-CAM target)</span>
              <span className="tabular-nums">{(prediction.presence * 100).toFixed(1)}%</span>
            </div>
            <Bar value={prediction.presence} />
          </button>

          <div className="space-y-2">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Per-vessel (select for Grad-CAM)
            </p>
            <div className="space-y-2">
              {prediction.vessels.map((v) => (
                <VesselRow
                  key={v.key}
                  row={v}
                  active={selectedVesselKey === v.key}
                  onClick={() => onSelectVessel(v.key)}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {!prediction && !loading && !error && (
        <p className="text-sm text-slate-500">Upload a series and open the viewer to run the model.</p>
      )}
    </div>
  );
}
