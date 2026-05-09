// Photorealism Phase 1.2 — outdoor HDRI presets.
//
// Each preset maps to a Polyhaven CC0 HDRI that gets loaded as
// scene.background + scene.environment for the world OUTSIDE the
// panel-wall in PT room mode. The colour cast of the loaded HDRI
// modulates the light entering each cell — different HDRIs produce
// different inter-cell hue boundaries on the floor caustic.
//
// Asset deployment: HDRs are committed via Git LFS into public/hdri/.
// Use the 4k variant for hero renders; 2k is enough for editor preview
// (PT_PREVIEW config). Filenames here MUST match the LFS-committed
// paths exactly.
//
// 'auto' falls back to the legacy time-of-day-driven dispatch in
// outdoorHdri.ts (the bucket-based selection). The four explicit
// presets pin the HDRI regardless of timeOfDay so the user can A/B
// scenes without touching the time-of-day slider.

import type { OutdoorScenePreset } from '@/store/viewportSlice';
import { outdoorHdriForTimeOfDay } from './outdoorHdri';

/**
 * Per-preset HDRI URLs. 4k variants (hero-quality). The paths are the
 * download URLs Polyhaven exposes; for production the user mirrors
 * them under public/hdri/ via Git LFS so the app loads them locally.
 *
 * Sources (CC0):
 *  - sunnyDay:    Kiara 5 Noon          (clear-sky, slightly warm noon sun)
 *  - overcast:    Cloudy Field          (uniform dome, no visible sun-disc)
 *  - goldenHour:  Belfast Sunset Puresky (warm sun low above horizon)
 *  - forest:      Forest Slope           (dappled green light through canopy)
 */
export const OUTDOOR_SCENE_PRESET_URL: Record<Exclude<OutdoorScenePreset, 'auto'>, string> = {
  sunnyDay:   '/hdri/kiara_5_noon_4k.hdr',
  overcast:   '/hdri/cloudy_field_4k.hdr',
  goldenHour: '/hdri/belfast_sunset_puresky_4k.hdr',
  forest:     '/hdri/forest_slope_4k.hdr',
};

/** Human-friendly label for each preset (for UI). */
export const OUTDOOR_SCENE_PRESET_LABEL: Record<OutdoorScenePreset, string> = {
  auto:       'Auto (time-of-day)',
  sunnyDay:   'Sunny mid-day',
  overcast:   'Overcast soft',
  goldenHour: 'Golden hour',
  forest:     'Forest / vegetation',
};

/**
 * Resolve the HDRI URL for a preset given current timeOfDay. 'auto'
 * delegates to the bucket-based dispatch.
 */
export function outdoorHdriForPreset(
  preset: OutdoorScenePreset,
  timeOfDay: number,
): string | undefined {
  if (preset === 'auto') return outdoorHdriForTimeOfDay(timeOfDay);
  return OUTDOOR_SCENE_PRESET_URL[preset];
}

/** Iteration order for UI dropdowns. */
export const OUTDOOR_SCENE_PRESET_ORDER: readonly OutdoorScenePreset[] = [
  'sunnyDay',
  'overcast',
  'goldenHour',
  'forest',
  'auto',
] as const;
