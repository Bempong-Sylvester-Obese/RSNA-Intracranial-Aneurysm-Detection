import { useEffect, useRef } from "react";
import { Enums, RenderingEngine, getRenderingEngine, init, type Types } from "@cornerstonejs/core";
import { buildRsnaImageId, registerRsnaPngLoader } from "@/cornerstone/rsnaPngLoader";

let cornerstoneReady = false;

function ensureCoreInit(): void {
  if (!cornerstoneReady) {
    init();
    registerRsnaPngLoader();
    cornerstoneReady = true;
  }
}

type OverlayOpts = {
  overlay?: "gradcam";
  vessel?: string | null;
};

type Props = {
  seriesId: string;
  numSlices: number;
  sliceIndex: number;
  onSliceChange: (index: number) => void;
  overlay: OverlayOpts;
};

export function CornerstoneStackViewport({
  seriesId,
  numSlices,
  sliceIndex,
  onSliceChange,
  overlay,
}: Props) {
  const elRef = useRef<HTMLDivElement | null>(null);
  const renderingEngineId = `re-${seriesId}`;
  const viewportId = `stack-${seriesId}`;
  const onSliceChangeRef = useRef(onSliceChange);
  onSliceChangeRef.current = onSliceChange;
  const removeWheelRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const el = elRef.current;
    if (!el || numSlices <= 0) {
      return undefined;
    }

    let cancelled = false;
    removeWheelRef.current?.();
    removeWheelRef.current = null;

    void (async () => {
      ensureCoreInit();
      if (cancelled || !elRef.current) {
        return;
      }

      const renderingEngine = new RenderingEngine(renderingEngineId);
      renderingEngine.enableElement({
        viewportId,
        type: Enums.ViewportType.STACK,
        element: el,
        defaultOptions: {
          background: [0.05, 0.05, 0.07] as Types.Point3,
        },
      });

      const viewport = renderingEngine.getViewport(viewportId) as Types.IStackViewport;
      const imageIds = Array.from({ length: numSlices }, (_, i) =>
        buildRsnaImageId(seriesId, i, overlay),
      );
      const start = Math.min(Math.max(0, sliceIndex), numSlices - 1);
      await viewport.setStack(imageIds, start);
      viewport.setProperties({
        voiRange: { lower: 0, upper: 1 } as Types.VOIRange,
      });
      viewport.render();

      const onWheel = (ev: WheelEvent) => {
        ev.preventDefault();
        const vp = getRenderingEngine(renderingEngineId)?.getViewport(viewportId) as
          | Types.IStackViewport
          | undefined;
        if (!vp) {
          return;
        }
        const cur = vp.getCurrentImageIdIndex();
        const delta = ev.deltaY > 0 ? 1 : -1;
        const next = Math.min(numSlices - 1, Math.max(0, cur + delta));
        void (async () => {
          await vp.setImageIdIndex(next);
          vp.render();
          onSliceChangeRef.current(next);
        })();
      };

      el.addEventListener("wheel", onWheel, { passive: false });
      removeWheelRef.current = () => el.removeEventListener("wheel", onWheel);
    })();

    return () => {
      cancelled = true;
      removeWheelRef.current?.();
      removeWheelRef.current = null;
      try {
        getRenderingEngine(renderingEngineId)?.destroy();
      } catch {
        /* ignore */
      }
    };
  }, [seriesId, numSlices, overlay.overlay, overlay.vessel ?? "", renderingEngineId, viewportId]);

  useEffect(() => {
    const re = getRenderingEngine(renderingEngineId);
    if (!re) {
      return;
    }
    const viewport = re.getViewport(viewportId) as Types.IStackViewport | undefined;
    if (!viewport) {
      return;
    }
    const idx = Math.min(Math.max(0, sliceIndex), numSlices - 1);
    void (async () => {
      if (viewport.getCurrentImageIdIndex() !== idx) {
        await viewport.setImageIdIndex(idx);
        viewport.render();
      }
    })();
  }, [sliceIndex, numSlices, renderingEngineId, viewportId]);

  return (
    <div
      ref={elRef}
      className="h-full min-h-[420px] w-full outline-none ring-1 ring-slate-800"
      tabIndex={0}
    />
  );
}
