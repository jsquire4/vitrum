// tonemap.ts — output tonemap operators (P4). Pure-TS reference implementations
// + the mode-index enum shared with the WGSL twin (`wgsl/tonemap.wgsl.ts`,
// kept in lockstep by `__tests__/tonemap.test.ts`).
//
// These are the operators `FrameQualitySettings.tonemap` selects. 'aces' is the
// default (matching the historical hardcoded composite curve). Backends apply
// exposure first, then the operator.

export type TonemapMode = 'aces' | 'agx' | 'reinhard' | 'linear' | 'none';

/** Numeric mode index shared with the WGSL `vitrumTonemap` selector. */
export const TONEMAP_MODE_INDEX: Readonly<Record<TonemapMode, number>> = {
  aces: 0,
  agx: 1,
  reinhard: 2,
  linear: 3,
  none: 4,
};

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/** Narkowicz 2015 ACES filmic approximation, per channel. */
export function acesFilmic(x: number): number {
  const a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp01((x * (a * x + b)) / (x * (c * x + d) + e));
}

/** Reinhard x/(1+x), per channel. */
export function reinhard(x: number): number {
  const v = Math.max(x, 0);
  return v / (1 + v);
}

/**
 * Minimal AgX (Wrensch) — the per-channel log2 sigmoid without the input/output
 * "look" rotation matrices. A punchy filmic curve; bounded [0,1].
 */
export function agx(x: number): number {
  const v = Math.max(x, 1e-6);
  const lx = clamp01((Math.log2(v) + 12.47393) / (12.47393 + 4.026069));
  const x2 = lx * lx, x4 = x2 * x2;
  return clamp01(
    15.5 * x4 * x2 - 40.14 * x4 * lx + 31.96 * x4 - 6.868 * x2 * lx + 0.4298 * x2 + 0.1191 * lx - 0.00232,
  );
}

const OPS: Readonly<Record<TonemapMode, (c: number) => number>> = {
  aces: acesFilmic,
  agx,
  reinhard,
  linear: clamp01,
  none: (c) => c,
};

/** Apply exposure then the selected operator to a linear RGB triple. */
export function applyTonemap(
  rgb: readonly [number, number, number],
  mode: TonemapMode,
  exposure = 1,
): [number, number, number] {
  const op = OPS[mode];
  return [op(rgb[0] * exposure), op(rgb[1] * exposure), op(rgb[2] * exposure)];
}

/** Linear → sRGB OETF (IEC 61966-2-1), per channel. Used when
 *  `FrameQualitySettings.outputColorSpace === 'srgb'` (the default); 'linear'
 *  output skips this. */
export function linearToSrgb(c: number): number {
  const v = Math.max(0, c);
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}

/** sRGB → linear EOTF, per channel (inverse of {@link linearToSrgb}). */
export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
