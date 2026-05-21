import { useEffect } from 'react';

interface RestirStageProps {
  readonly backdropMode: unknown;
  readonly skyParams: unknown;
  readonly nightSkyParams: unknown;
  readonly frameLayout: { cx: number; cy: number; w: number; h: number };
  readonly orbitTarget: [number, number, number];
}

/**
 * Staging-only compatibility shell.
 * Runtime ReSTIR stage behavior is implemented in @vitrum/walkaround-hybrid.
 */
export function RestirStage(props: RestirStageProps) {
  useEffect(() => {
    if (typeof console !== 'undefined') {
      console.warn('[staging] RestirStage is deprecated. Use @vitrum/walkaround-hybrid.', props);
    }
  }, [props]);
  return null;
}
