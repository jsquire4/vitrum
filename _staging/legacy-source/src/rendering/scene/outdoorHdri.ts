// Outdoor HDRI selection by timeOfDay. Each bucket maps to a CC0 Poly Haven
// HDR. v1 uses a discrete bucket map (no cross-fade); cross-fade lands in
// S7 polish.
//
// HDRI URLs are placeholders for the autonomous run — replace with the
// actual `public/hdri/<name>.hdr` paths once the assets are committed via
// LFS (S5-T0 + the contributor's `git lfs install` step).

export type TimeOfDayBucket = 'dawn' | 'noon' | 'sunset' | 'dusk' | 'night';

/** Map a normalized timeOfDay [0,1] to a discrete bucket. Aligns with
 *  the SUN_INTENSITY buckets in lightingIntensityTable.ts so HDRI changes
 *  match sun-intensity steps. */
export function bucketForTimeOfDay(t: number): TimeOfDayBucket {
  const tt = Math.max(0, Math.min(1, t));
  if (tt < 0.05 || tt > 0.95) return 'night';
  if (tt < 0.10 || tt > 0.90) return 'dusk';
  if (tt < 0.20 || tt > 0.80) return 'sunset';
  if (tt < 0.40 || tt > 0.60) return 'dawn';
  return 'noon';
}

/** Public/-relative HDR path for a bucket. CC0 assets from Poly Haven
 *  (https://polyhaven.com), committed via Git LFS. */
export const OUTDOOR_HDRI_BY_TIME_OF_DAY: Record<TimeOfDayBucket, string | undefined> = {
  dawn:   '/hdri/kloofendal_43d_clear_2k.hdr',
  noon:   '/hdri/kiara_5_noon_2k.hdr',
  sunset: '/hdri/belfast_sunset_puresky_2k.hdr',
  dusk:   '/hdri/qwantani_dusk_2_puresky_2k.hdr',
  night:  '/hdri/dikhololo_night_2k.hdr',
};

export function outdoorHdriForTimeOfDay(t: number): string | undefined {
  return OUTDOOR_HDRI_BY_TIME_OF_DAY[bucketForTimeOfDay(t)];
}
