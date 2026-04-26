/** Thin API client; paths are relative so Vite proxies `/api` in dev. */

/** Must match backend `_vessel_key("Aneurysm Present")`. */
export const PRESENCE_VESSEL_KEY = "aneurysm_present";

export type SeriesMetadata = {
  series_id: string;
  num_files: number;
  num_slices: number;
  height: number;
  width: number;
  modality: string | null;
};

export type VesselScore = {
  key: string;
  label: string;
  probability: number;
  class_index: number;
};

export type PredictionResponse = {
  presence: number;
  presence_index: number;
  vessels: VesselScore[];
  gradcam_ready: boolean;
};

export type UploadResponse = {
  series_id: string;
  num_files: number;
};

export async function uploadSeries(files: File[]): Promise<UploadResponse> {
  const body = new FormData();
  for (const f of files) {
    body.append("files", f);
  }
  const res = await fetch("/api/series", { method: "POST", body });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed (${res.status})`);
  }
  return res.json() as Promise<UploadResponse>;
}

export async function fetchMetadata(seriesId: string): Promise<SeriesMetadata> {
  const res = await fetch(`/api/series/${seriesId}/metadata`);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json() as Promise<SeriesMetadata>;
}

export async function runPredict(seriesId: string): Promise<PredictionResponse> {
  const res = await fetch(`/api/series/${seriesId}/predict`, { method: "POST" });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json() as Promise<PredictionResponse>;
}

export function sliceUrl(
  seriesId: string,
  index: number,
  opts?: { overlay?: "gradcam"; vessel?: string | null },
): string {
  const u = new URL(`/api/series/${seriesId}/slice/${index}.png`, window.location.origin);
  if (opts?.overlay) {
    u.searchParams.set("overlay", opts.overlay);
  }
  if (opts?.vessel) {
    u.searchParams.set("vessel", opts.vessel);
  }
  return u.pathname + u.search;
}
