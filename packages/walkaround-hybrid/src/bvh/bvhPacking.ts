/**
 * Generic BVH vertex-stream packing shared across the walkaround subsystems.
 *
 * These helpers carry no ReSTIR/RC/DDGI-specific semantics — they are pure
 * typed-array bit-packing over a vec4f-strided stream — so they live in a
 * neutral `bvh/` module rather than in any one subsystem's `restir/`/`rc/`/
 * `ddgi/` folder (I3-1: `restir/` was acting as a de-facto shared-foundation
 * sink). Consumers: `restir/bvhCore.ts` (position.w + normal.w), `rc/bvhCore.ts`
 * (normal.w). `restir/packingHelpers.ts` re-exports these for back-compat with
 * existing test imports.
 */

export interface BufferAttributeLike {
  readonly array: ArrayLike<number>;
}

/**
 * Pack UV (two f16 values) into the .w slot of every vec4f position.
 * See the ReSTIR BVH builders for the rationale (single storage buffer per stage).
 */
export function packUVIntoPositionW(
  positions: Float32Array,
  uvAttr: BufferAttributeLike | undefined,
  vertCount: number,
): Float32Array<ArrayBuffer> {
  return packUVIntoVec4W(positions, uvAttr, vertCount);
}

/**
 * Pack UV (two f16 values) into the .w slot of a vec4f-strided stream.
 * The xyz lanes are preserved verbatim. Used for position.w (uv0, consumed by
 * traversal) and normal.w (uv1, consumed by material texture sampling).
 */
export function packUVIntoVec4W(
  values: Float32Array,
  uvAttr: BufferAttributeLike | undefined,
  vertCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(values.length);
  out.set(values);
  const u32View = new Uint32Array(out.buffer);
  const sourceUvs = uvAttr?.array;

  for (let i = 0; i < vertCount; i++) {
    const u16 = floatToHalfBits(sourceUvs?.[i * 2 + 0] ?? 0);
    const v16 = floatToHalfBits(sourceUvs?.[i * 2 + 1] ?? 0);
    u32View[i * 4 + 3] = (v16 << 16) | u16;
  }
  return out;
}

function floatToHalfBits(value: number): number {
  const input = Number.isFinite(value) ? Math.fround(value) : 0;
  const sign = input < 0 || Object.is(input, -0) ? 0x8000 : 0;
  const abs = Math.abs(input);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 2 ** -24) return sign;
  if (abs < 2 ** -14) {
    return sign | Math.min(0x03ff, Math.round(abs / (2 ** -24)));
  }

  let exp = Math.floor(Math.log2(abs));
  let mant = Math.round((abs / (2 ** exp) - 1) * 1024);
  if (mant === 1024) {
    mant = 0;
    exp += 1;
  }
  const halfExp = exp + 15;
  if (halfExp >= 31) return sign | 0x7bff;
  return sign | ((halfExp & 0x1f) << 10) | (mant & 0x03ff);
}
