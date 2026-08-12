/**
 * B12 — Lite-tier light and environment texture packing.
 *
 * On capped adapters (≤8 storage buffers/stage) the lite tier cannot bind the
 * full-tier group-1 storage buffers (point/spot/rect lights, env texels+CDF).
 * Instead, we pack all analytic light data and the HDRI env (radiance + CDF) into
 * sampled textures (`texture_2d<f32>`).  Sampled textures use a SEPARATE per-stage
 * limit (`maxSampledTexturesPerShaderStage` ≥ 16 in the WebGPU baseline) that does
 * NOT count against `maxStorageBuffersPerShaderStage`.
 *
 * Three new group-0 texture bindings are added to the lite layout:
 *   @group(0) @binding(12)  liteEnvTex      — RGBA32F  envWidth × envHeight
 *                                              (.rgb = HDR radiance, .a = pdf/sr)
 *   @group(0) @binding(13)  liteEnvCdfTex   — R32F-packed-into-RGBA envWidth × envHeight
 *                                              Each texel .r = normalised CDF value at
 *                                              pixel index (y * W + x).  CDF[0] = 0 is
 *                                              implicit; only CDF[1..W*H] are stored.
 *   @group(0) @binding(14)  liteLightTex    — RGBA32F  liteLightTexWidth × 1
 *                                              Contiguous directional / point /
 *                                              spot / rect-area light records
 *                                              packed in vec4 rows.
 *
 * Layout of liteLightTex (row 0, consecutive texels):
 *   [0, directionalLightCount*2):              directional records  (2 vec4 each)
 *   [dirOff, dirOff + pointLightCount*3):       point light records  (3 vec4 each)
 *   [pointOff, pointOff + spotLightCount*4):   spot light records   (4 vec4 each)
 *   [spotOff,  spotOff  + rectLightCount*4):   rect-area records    (4 vec4 each)
 *
 * This mirrors the exact float layout the full-tier storage buffers use (see
 * emitterPacking.ts) so the same GPU-side constants/strides apply in WGSL.
 *
 * Placeholder textures (1×1, black) are used when the scene has no lights / no
 * env map, so the bind-group layout is always satisfied.
 */

/**
 * Minimum texture width for the light data texture.  A 1×1 placeholder is used
 * when the scene has no point/spot/rect-area lights so the binding is always valid.
 */
const LITE_LIGHT_TEX_MIN_WIDTH = 1;

/** Minimum env-texture size (1×1) when no HDRI / procedural-sky is present. */
const LITE_ENV_TEX_MIN_WIDTH  = 1;
const LITE_ENV_TEX_MIN_HEIGHT = 1;

/**
 * Packed light data texture descriptor.
 *
 * `data`  — flat RGBA32F pixel data (4 floats/pixel), row-major, 1 row high.
 * `width` — pixel width (≥ 1).
 */
export interface LiteLightTexData {
  readonly data:  Float32Array;
  readonly width: number;
}

/**
 * Packed env-radiance texture descriptor (radiance map + per-pixel PDF).
 *
 * `texels` — flat RGBA32F pixel data (.rgb = HDR radiance, .a = pdf/sr).
 * `width`  — texture width (pixels).
 * `height` — texture height (pixels).
 */
export interface LiteEnvTexData {
  readonly texels: Float32Array;
  readonly width:  number;
  readonly height: number;
}

/**
 * Packed env-CDF texture descriptor.
 *
 * `data`   — flat RGBA32F pixel data (.r = normalised CDF entry per pixel).
 *            Each texel stores one CDF value: texel at (x, y) = cdf[(y*W + x) + 1]
 *            (cdf[0]=0 is implicit; the binary search starts at lo=0).
 * `width`  — must equal the env map width.
 * `height` — must equal the env map height.
 */
export interface LiteEnvCdfData {
  readonly data:   Float32Array;
  readonly width:  number;
  readonly height: number;
}

// ---------------------------------------------------------------------------
// Light texture packer
// ---------------------------------------------------------------------------

/**
 * Pack the directional / point / spot / rect-area light arrays (already in full-tier float
 * layout from `emitterPacking.ts`) into a single RGBA32F data texture.  The
 * data is written contiguously: directional records first, then point, then spot,
 * then rect-area.
 * Counts are read from the UBO at runtime (`params.pointLightCount` etc.) so no
 * header texel is needed.
 *
 * Returns a 1-row RGBA32F texture of width = total vec4 texels required (≥ 1).
 */
export function packLiteLightTexture(
  directionalLightsData: Float32Array,
  pointLightsData: Float32Array,
  spotLightsData:  Float32Array,
  rectAreaLightsData: Float32Array,
): LiteLightTexData {
  // Each array is already packed as float32 in vec4 rows.
  // Total vec4 rows = directionalCount*2 + pointCount*3 + spotCount*4 + rectCount*4.
  // array.length / 4 gives the number of vec4 rows.
  const directionalVec4s = (directionalLightsData.length / 4) | 0;
  const pointVec4s = (pointLightsData.length / 4) | 0;
  const spotVec4s  = (spotLightsData.length  / 4) | 0;
  const rectVec4s  = (rectAreaLightsData.length / 4) | 0;
  const totalVec4s = directionalVec4s + pointVec4s + spotVec4s + rectVec4s;

  const width = Math.max(LITE_LIGHT_TEX_MIN_WIDTH, totalVec4s);
  const data  = new Float32Array(width * 4);

  let off = 0;
  // Directional lights (stride 2 × vec4f = 8 floats per light)
  for (let i = 0; i < directionalLightsData.length; i++) {
    data[off++] = directionalLightsData[i]!;
  }
  // Point lights (stride 3 × vec4f = 12 floats per light)
  for (let i = 0; i < pointLightsData.length; i++) {
    data[off++] = pointLightsData[i]!;
  }
  // Spot lights (stride 4 × vec4f = 16 floats per light)
  for (let i = 0; i < spotLightsData.length; i++) {
    data[off++] = spotLightsData[i]!;
  }
  // Rect-area lights (stride 4 × vec4f = 16 floats per light)
  for (let i = 0; i < rectAreaLightsData.length; i++) {
    data[off++] = rectAreaLightsData[i]!;
  }

  return { data, width };
}

// ---------------------------------------------------------------------------
// Env texture packer (pass-through: data is already built by environmentPacking)
// ---------------------------------------------------------------------------

/**
 * Wrap the CPU-packed HDRI texel array (from `environmentPacking.ts`) in the
 * `LiteEnvTexData` descriptor.  No copy — the caller's `Float32Array` is
 * referenced directly.  Returns a 1×1 black placeholder when the scene has no
 * valid HDRI.
 */
export function packLiteEnvTexture(
  hdriTexels: Float32Array,
  hdriWidth:  number,
  hdriHeight: number,
  hasHdri: boolean,
): LiteEnvTexData {
  if (!hasHdri || hdriWidth <= 0 || hdriHeight <= 0 || hdriTexels.length < hdriWidth * hdriHeight * 4) {
    return {
      texels: new Float32Array(4), // 1×1 black
      width:  LITE_ENV_TEX_MIN_WIDTH,
      height: LITE_ENV_TEX_MIN_HEIGHT,
    };
  }
  return { texels: hdriTexels, width: hdriWidth, height: hdriHeight };
}

/**
 * Build the env-CDF texture from the flat CDF array (from `environmentPacking.ts`).
 * The CDF array has `W*H + 1` entries; `cdf[0]` = 0 is implicit.  We store
 * `cdf[1..W*H]` in a W×H RGBA texture with CDF values in the `.r` channel.
 *
 * Returns a W×H texture.  A 1×1 zero placeholder is returned when no valid HDRI.
 */
export function packLiteEnvCdfTexture(
  hdriCdf:    Float32Array,
  hdriWidth:  number,
  hdriHeight: number,
  hasHdri: boolean,
): LiteEnvCdfData {
  const count = hdriWidth * hdriHeight;
  if (!hasHdri || hdriWidth <= 0 || hdriHeight <= 0 || hdriCdf.length < count + 1) {
    return {
      data:   new Float32Array(4), // 1×1 zero
      width:  LITE_ENV_TEX_MIN_WIDTH,
      height: LITE_ENV_TEX_MIN_HEIGHT,
    };
  }
  // Pack cdf[1..count] into RGBA texels (.r channel; .gba = 0).
  // Texel at (x, y) → cdf[(y * hdriWidth + x) + 1].
  const data = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    data[i * 4] = hdriCdf[i + 1]!; // .r = cdf value
    // .g, .b, .a = 0 (default)
  }
  return { data, width: hdriWidth, height: hdriHeight };
}

/**
 * Packed material-texture descriptor atlas for the lite tier.
 *
 * Full-tier descriptors live in a storage buffer. Lite spends its last storage
 * slot on `meshUvs`, so the same vec4 records are uploaded as an unfilterable
 * RGBA32F texture: width = MATERIAL_TEX_VEC4_STRIDE, height = materialCount,
 * texel (x, y) = descriptor vec4 `y * width + x`.
 */
export interface LiteDescriptorTexData {
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
}

export function packLiteMaterialDescriptorTexture(
  descriptors: Float32Array,
  vec4Stride: number,
): LiteDescriptorTexData {
  const width = Math.max(1, vec4Stride);
  const floatStride = width * 4;
  const materialCount = Math.max(1, Math.floor(descriptors.length / floatStride));
  const height = materialCount;
  const data = new Float32Array(width * height * 4);
  const copy = Math.min(descriptors.length, data.length);
  if (copy > 0) data.set(descriptors.subarray(0, copy));
  return { data, width, height };
}
