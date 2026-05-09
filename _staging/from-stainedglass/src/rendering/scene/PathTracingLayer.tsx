import { useCallback, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { Pathtracer } from '@react-three/gpu-pathtracer';
import type { ReactNode } from 'react';
import type { AppDispatch } from '@/store';
import { setPathTracerEnabled, setPtDeviceLost } from '@/store/viewportSlice';
import {
  PT_RESOLUTION_FACTOR,
  PT_LOW_RES_SCALE,
} from './pathtracerConstants';
import { usePTSampleTarget } from './lighting/usePTSampleTarget';
import { usePTPipelineConfig } from './lighting/usePTPipelineConfig';
import { PathtracerSceneSync } from './PathtracerSceneSync';
import { PathtracerDebugBridge } from './PathtracerDebugBridge';
import { PTDeviceLostBoundary } from './PTDeviceLostBoundary';
import { clearSkyEquirectCache } from './ptIblBaker';

interface PathTracingLayerProps {
  enabled: boolean;
  children: ReactNode;
}

/**
 * Wrap the path-traceable subtree of the 3D scene in a `<Pathtracer>`.
 * The pmndrs wrapper instantiates `WebGLPathTracer` from
 * three-gpu-pathtracer and re-renders the entire `useThree().scene`
 * each frame using accumulated samples.
 *
 * Render-mode invariant (S4-T20): raster is the editor surface — every
 * interactive edit (chip, drag, color picker) targets the raster path
 * (`pathTracerEnabled === false`). PT is opt-in via `pathTracerEnabled`.
 * Each PT re-render is 60–120s; users typically toggle PT off while
 * editing. The PT toggle + render-progress + final-render checkbox live
 * in a small dedicated control at the top of the viewport — never in
 * the chip or side panel.
 *
 * Mounting invariant: `WebGLPathTracer.setScene(...)` walks every
 * material in the scene — any mesh whose material lacks a `.color`
 * field crashes the canvas. Sky shaders and GizmoHelper internals
 * are PT-incompatible offenders; they live in `<RasterStage>` only
 * (JSX-gated, never mounted in the PT subtree).
 *
 * Performance settings come from `pathtracerConstants.ts`; the 60s
 * honeycomb timing gate imports the same constants so bumping
 * convergence updates the gate too.
 *
 * The dev-only `__PT__` debug bridge lives in `PathtracerDebugBridge`
 * (sibling file) so this component stays focused on the wrapper.
 */
export function PathTracingLayer({ enabled, children }: PathTracingLayerProps) {
  // Drop the sky-equirect DataTexture cache on PT unmount. Each cached
  // entry holds a Uint16Array (HalfFloat) buffer + the DataTexture wrapping
  // it; without explicit cleanup, toggling PT off/on across a day's worth
  // of timeOfDay scrubs accumulates ~32 ~256KB buffers per cycle.
  useEffect(() => {
    if (!enabled) return;
    return () => clearSkyEquirectCache();
  }, [enabled]);

  if (!enabled) return <>{children}</>;
  return <PathTracingInner>{children}</PathTracingInner>;
}

function PathTracingInner({ children }: { children: ReactNode }) {
  // Pipeline config — preview vs final-render mode (S7-T7). Final mode
  // bumps samples + bounces + drops glossy factor; exact values live
  // in PT_FINAL within pathtracerConstants.ts (kept there so test
  // fixtures and timing budgets read the same numbers).
  const pipeline = usePTPipelineConfig();
  // Per-scene sample target — fixture-shaded scenes bump preview
  // (S6-T10). Final mode's sample count dwarfs that bump, so max()
  // picks final whenever the user toggles final-render-mode on.
  const fixtureTarget = usePTSampleTarget();
  const targetSamples = Math.max(pipeline.samples, fixtureTarget);

  const dispatch = useDispatch<AppDispatch>();
  const handleDeviceLost = useCallback(
    (_error: Error) => {
      dispatch(setPathTracerEnabled(false));
      dispatch(setPtDeviceLost(true));
    },
    [dispatch],
  );

  return (
    <PTDeviceLostBoundary onError={handleDeviceLost}>
      <Pathtracer
        enabled
        bounces={pipeline.bounces}
        samples={targetSamples}
        minSamples={1}
        filteredGlossyFactor={pipeline.filteredGlossyFactor}
        resolutionFactor={PT_RESOLUTION_FACTOR}
        tiles={[1, 1]}
        dynamicLowRes
        lowResScale={PT_LOW_RES_SCALE}
        renderDelay={0}
      >
        <PathtracerDebugBridge targetSamples={targetSamples} />
        <PathtracerSceneSync />
        {children}
      </Pathtracer>
    </PTDeviceLostBoundary>
  );
}
