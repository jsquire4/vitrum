import { useEffect } from 'react';

interface HybridLayeredStageProps {
  readonly backdropMode: unknown;
  readonly skyParams: unknown;
  readonly nightSkyParams: unknown;
  readonly frameLayout: { cx: number; cy: number; w: number; h: number };
  readonly orbitTarget: [number, number, number];
}

/**
 * Staging-only compatibility shell.
 * The canonical implementation lives in @vitrum/walkaround-hybrid.
 */
export function HybridLayeredStage(props: HybridLayeredStageProps) {
  useEffect(() => {
    if (typeof console !== 'undefined') {
      console.warn('[staging] HybridLayeredStage is deprecated. Use @vitrum/walkaround-hybrid.', props);
    }
  }, [props]);
  return null;
}
