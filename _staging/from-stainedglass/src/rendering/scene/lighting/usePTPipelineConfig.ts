// S7-T7: PT pipeline-config hook. Returns PT_PREVIEW or PT_FINAL based on
// the session-y `ui.finalRenderMode` flag. Consumers (PathTracingLayer)
// thread the values into the <Pathtracer> props; mid-run prop changes
// trigger a re-render at the new config.
//
// **Verification:** mid-run prop changes to <Pathtracer> (`bounces`,
// `samples`, `filteredGlossyFactor`) appear to take effect on next
// frame in @react-three/gpu-pathtracer 0.3.2 — visually verified by
// toggling final-render mid-render. If a future version regresses,
// add `key={cfgKey}` to <Pathtracer> to force remount.

import { useSelector } from 'react-redux';
import { selectFinalRenderMode } from '@/store/selectors';
import {
  PT_PREVIEW,
  PT_FINAL,
  type PTPipelineConfig,
} from '../pathtracerConstants';

export function usePTPipelineConfig(): PTPipelineConfig {
  const finalMode = useSelector(selectFinalRenderMode);
  return finalMode ? PT_FINAL : PT_PREVIEW;
}
