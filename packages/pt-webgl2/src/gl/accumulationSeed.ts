/**
 * Derive the signed 32-bit GLSL `seed` for one canonical accumulation sample.
 *
 * Hosts are allowed to repeat `FrameInput.frameSeed`; accumulation still has to
 * advance its geometry/RNG sample. For fixed host/frame inputs the mapping from
 * `accumulatedSample` to the returned bit pattern is bijective over uint32, so
 * consecutive samples cannot collapse onto the same seed. Returning the signed
 * representation preserves those exact bits through WebGL's `uniform1i`.
 */
export function accumulationSeed(
  frameSeed: number,
  frameIndex: number,
  accumulatedSample: number,
): number {
  const hostWord = Math.trunc(frameSeed) >>> 0;
  const frameWord = mixUint32((Math.trunc(frameIndex) >>> 0) ^ 0x9e37_79b9);
  const sampleWord = mixUint32(
    (Math.trunc(accumulatedSample) >>> 0) ^ 0x243f_6a88,
  );
  const combined =
    hostWord ^
    rotateLeftUint32(frameWord, 11) ^
    sampleWord;
  return mixUint32(combined) | 0;
}

function rotateLeftUint32(value: number, shift: number): number {
  return ((value << shift) | (value >>> (32 - shift))) >>> 0;
}

/** Reversible uint32 finalizer (odd multipliers + reversible xor shifts). */
function mixUint32(value: number): number {
  let mixed = value >>> 0;
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x7feb_352d) >>> 0;
  mixed = Math.imul(mixed ^ (mixed >>> 15), 0x846c_a68b) >>> 0;
  return (mixed ^ (mixed >>> 16)) >>> 0;
}
