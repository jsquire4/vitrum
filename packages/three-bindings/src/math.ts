/**
 * Shared math helpers for the three-bindings layer.
 *
 * Centralized so the Rec. 709 luminance constants don't drift across
 * call sites (mesh-emitter detection, light packing for cellPower, etc.).
 */

/** Rec. 709 luminance of an RGB triple, optionally scaled by an
 *  intensity multiplier. Used to gate mesh-emitter detection and to
 *  pack `userData.cellPower` for emissive area lights. */
export function luminance(r: number, g: number, b: number, intensity = 1): number {
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) * intensity;
}
