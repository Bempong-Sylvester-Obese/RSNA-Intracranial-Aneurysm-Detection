/**
 * Registers a Cornerstone image loader that fetches pre-windowed PNG tiles from our API.
 */
import { Enums, imageLoader, type Types } from "@cornerstonejs/core";

const SCHEME = "rsna-png";

function parseImageId(imageId: string): { seriesId: string; index: number; qs: string } {
  const without = imageId.slice(SCHEME.length + "://".length);
  const [pathPart, query = ""] = without.split("?");
  const lastSlash = pathPart.lastIndexOf("/");
  const seriesId = pathPart.slice(0, lastSlash);
  const index = Number(pathPart.slice(lastSlash + 1));
  if (!seriesId || !Number.isFinite(index)) {
    throw new Error(`Bad rsna-png imageId: ${imageId}`);
  }
  return { seriesId, index, qs: query };
}

function imageIdToUrl(imageId: string): string {
  const { seriesId, index, qs } = parseImageId(imageId);
  const path = `/api/series/${seriesId}/slice/${index}.png`;
  return qs ? `${path}?${qs}` : path;
}

async function loadImage(imageId: string): Promise<Types.IImage> {
  const url = imageIdToUrl(imageId);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${url}: ${res.status}`);
  }
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);

  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2D canvas unsupported");
  }
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const float32 = new Float32Array(bitmap.width * bitmap.height);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
    float32[j] = data[i]! / 255;
  }

  const image: Types.IImage = {
    imageId,
    color: false,
    rgba: false,
    columnPixelSpacing: 1,
    rowPixelSpacing: 1,
    columns: bitmap.width,
    rows: bitmap.height,
    height: bitmap.height,
    width: bitmap.width,
    intercept: 0,
    slope: 1,
    windowCenter: 0.5,
    windowWidth: 1,
    minPixelValue: 0,
    maxPixelValue: 1,
    sizeInBytes: float32.byteLength,
    getPixelData: () => float32,
    getCanvas: () => canvas,
    voiLUTFunction: Enums.VOILUTFunctionType.LINEAR,
    invert: false,
    photometricInterpretation: "MONOCHROME2",
    dataType: "Float32Array",
    numberOfComponents: 1,
  };

  bitmap.close();
  return image;
}

export function registerRsnaPngLoader(): void {
  imageLoader.registerImageLoader(SCHEME, (imageId) => ({
    promise: loadImage(imageId).then((img) => img as unknown as Record<string, unknown>),
  }));
}

export function buildRsnaImageId(
  seriesId: string,
  index: number,
  opts?: { overlay?: "gradcam"; vessel?: string | null },
): string {
  const qs = new URLSearchParams();
  if (opts?.overlay) {
    qs.set("overlay", opts.overlay);
  }
  if (opts?.vessel) {
    qs.set("vessel", opts.vessel);
  }
  const q = qs.toString();
  return q
    ? `${SCHEME}://${seriesId}/${index}?${q}`
    : `${SCHEME}://${seriesId}/${index}`;
}
