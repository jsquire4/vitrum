/**
 * textureDecode.ts — shared CPU-readable texture decode helpers (D12-1/2/10).
 *
 * These fold the byte-identical decode preambles that were previously copied
 * across `emitterClassify.ts` (3× decode preamble in `averageReadableTextureRgb`,
 * `readTextureRgbAtTexel`, `sampleReadableTextureRgbAtUv`) and the parallel
 * `decoderFor` in `vertexDisplacement.ts`.
 *
 *  - {@link halfToFloat} — IEEE-754 half → float32 (bit-identical across callers).
 *  - {@link resolveReadableTexture} — the `handle.data ?? handle.image?.data` +
 *    width/height/pixelCount + hint + stride + validity resolution used by the
 *    three `emitterClassify` texel readers (D12-1).
 *  - {@link makeChannelDecoder} — the `useHalf/useUint16/useFloat/bpe/intMax`
 *    decode-fn factory (D12-10 folds the `BYTES_PER_ELEMENT`-heuristic vs
 *    `__vitrum_hint__.dataType` dual-branching into one place).
 */

/** IEEE-754 half-precision (float16) → float32. */
export function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/** Normalized decode-hint shape (the resolved `__vitrum_hint__`). */
export interface TextureDecodeHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32';
  readonly colorSpace?: 'srgb' | 'linear';
}

/**
 * Build the per-value decode fn for a CPU-readable texture buffer, folding the
 * `__vitrum_hint__.dataType` path and the `BYTES_PER_ELEMENT` heuristic into one
 * place (D12-10). Semantics preserved bit-for-bit from the former inline
 * preambles in `emitterClassify.ts`:
 *   - `float16`/`half-float`     → {@link halfToFloat}
 *   - `float32` (or Float32Array) → identity
 *   - `uint16` (or Uint16Array)  → clamp(v / 65535)
 *   - otherwise                  → `intMax = 2^(8·BYTES_PER_ELEMENT) − 1`,
 *                                   `intMax > 0 ? v / intMax : v` (default bpe=1).
 */
export function makeChannelDecoder(
  src: ArrayLike<number>,
  dataType: TextureDecodeHint['dataType'] | undefined,
): (v: number) => number {
  const isFloat = src instanceof Float32Array;
  const useHalf = dataType != null
    ? dataType === 'float16' || dataType === 'half-float'
    : false;
  const useUint16 = dataType != null ? dataType === 'uint16' : src instanceof Uint16Array;
  const useFloat = dataType != null ? dataType === 'float32' : isFloat;
  const bpe = (src as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useUint16 || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  return (v: number): number => (
    useHalf ? halfToFloat(v) :
      useFloat ? v :
      useUint16 ? Math.min(1, Math.max(0, v / 65535)) :
      intMax > 0 ? v / intMax : v
  );
}

/** sRGB → linear (KHR / core color-space convention). */
export function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** Minimal shape a CPU-readable texture handle exposes to the decoders. */
export interface DecodableTextureHandle {
  readonly width?: number;
  readonly height?: number;
  readonly data?: ArrayLike<number>;
  readonly image?: { readonly width?: number; readonly height?: number; readonly data?: ArrayLike<number> };
  /**
   * Optional immutable CPU snapshot for an otherwise opaque/GPU-resident
   * texture. Backends use this structural lane when they must build an
   * emissive-light distribution without reading a host-owned GPU resource.
   */
  readonly cpuMirror?: {
    readonly width?: number;
    readonly height?: number;
    readonly data?: ArrayLike<number>;
  };
}

/** Resolved decode state shared by the emitter-map texel readers. */
export interface ResolvedReadableTexture {
  readonly src: ArrayLike<number>;
  readonly width: number;
  readonly height: number;
  readonly pixelCount: number;
  readonly stride: number;
  readonly decode: (v: number) => number;
  readonly needsSrgbDecode: boolean;
}

/**
 * Resolve the `data`/dims/stride/decode-fn/srgb-flag for a CPU-readable texture
 * handle (D12-1). Returns `null` when the handle is not readable, dims are
 * non-positive, the stride is out of range, or the buffer is too short — the
 * exact reject conditions the three `emitterClassify` texel readers used inline.
 *
 * `width`/`height` are floored; `pixelCount = floor(w)·floor(h)` — identical to
 * the former per-reader arithmetic.
 */
export function resolveReadableTexture(
  handle: DecodableTextureHandle | null | undefined,
  fieldColorSpace: 'srgb' | 'linear',
  hintChannels: TextureDecodeHint['channels'] | undefined,
  hintDataType: TextureDecodeHint['dataType'] | undefined,
  hintColorSpace: TextureDecodeHint['colorSpace'] | undefined,
): ResolvedReadableTexture | null {
  if (handle == null) return null;
  const mirror = handle.cpuMirror;
  const src = mirror?.data ?? handle.data ?? handle.image?.data;
  const width = Math.floor(Number(mirror?.width ?? handle.width ?? handle.image?.width ?? 0));
  const height = Math.floor(Number(mirror?.height ?? handle.height ?? handle.image?.height ?? 0));
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) {
    return null;
  }
  const pixelCount = width * height;
  if (pixelCount <= 0) return null;
  const heuristicStride = Math.max(1, Math.round(src.length / pixelCount));
  const stride = hintChannels ?? heuristicStride;
  if (stride < 1 || stride > 4 || src.length < pixelCount * stride) {
    return null;
  }
  const decode = makeChannelDecoder(src, hintDataType);
  // Raw floating-point texture payloads are the library's implicit linear-HDR
  // lane. Both path-tracer atlas uploaders preserve unhinted Float32 RGB above
  // one instead of treating it as normalized sRGB, so CPU emitter
  // classification must make the same choice. An explicit color-space hint
  // always wins; this inference exists only for the otherwise-ambiguous
  // unhinted Float32 case (including immutable cpuMirror proxies whose
  // dataType, rather than instanceof, carries the representation).
  const unhintedFloat32IsLinear =
    hintColorSpace == null &&
    (
      hintDataType === 'float32' ||
      (hintDataType == null && src instanceof Float32Array)
    );
  const needsSrgbDecode =
    fieldColorSpace === 'srgb' &&
    hintColorSpace !== 'linear' &&
    !unhintedFloat32IsLinear;
  return { src, width, height, pixelCount, stride, decode, needsSrgbDecode };
}
