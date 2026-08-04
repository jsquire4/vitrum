/**
 * B3 (road-to-100) — directional IBL GPU resources for the scene bind group
 * (bindings 15-19): the equirect radiance map + the PBRT 2D-distribution
 * forward CDFs as textures (so the scene group stays at three versioned
 * storage arenas), plus the small EnvParams uniform.
 *
 * A 1×1 placeholder set + envParams.hasEnv=0 backs non-HDRI scenes so the bind
 * group is always valid; the WGSL `environmentSample` helpers fall back to the
 * scalar sky (skyTint·skyIrradiance) when hasEnv==0 → no-HDRI byte-identity.
 */

import type { DirectionalEnvData } from '../environment/equirectDirectional.js';
import {
  assertWalkaroundEnvironmentMapScaleEnvelopeF32,
  packWalkaroundEnvironmentRotationF32,
} from '../environment/environmentRadianceScale.js';

/** GPUTextureUsage.TEXTURE_BINDING | COPY_DST (literals — Node vitest has no globals). */
const TEX_BINDING = 0x04;
const COPY_DST = 0x02;
/** GPUBufferUsage.UNIFORM | COPY_DST. */
const UNIFORM = 0x40;
const BUF_COPY_DST = 0x08;

/** EnvParams uniform: { hasEnv:u32, width:u32, height:u32, rotationY:f32,
 *  intensity:f32 } padded to 16-byte alignment (32 bytes). */
const ENV_PARAMS_BYTES = 32;

export interface EnvironmentTextures {
  map: GPUTexture;
  pdf: GPUTexture;
  marginal: GPUTexture;
  conditional: GPUTexture;
  sampler: GPUSampler;
  paramsBuffer: GPUBuffer;
  /** World-to-unrotated-map Y rotation carried by EnvParams. */
  rotationY: number;
  /** Radiance multiplier carried by EnvParams; map texels stay unit-intensity. */
  intensity: number;
  /** True only when `map` contains a live directional environment payload.
   *  The no-HDRI placeholder is still bindable, but must not suppress the
   *  scalar-sky fallback in RC and the main/ReSTIR shading path. */
  hasDirectionalEnvironment: boolean;
}

function createTex(
  device: GPUDevice,
  label: string,
  width: number,
  height: number,
  format: GPUTextureFormat,
): GPUTexture {
  return device.createTexture({
    label,
    size: { width: Math.max(1, width), height: Math.max(1, height), depthOrArrayLayers: 1 },
    format,
    usage: TEX_BINDING | COPY_DST,
  });
}

function reportedTextureDimensionLimit(device: GPUDevice): number | undefined {
  const raw = (
    device.limits as unknown as Record<string, unknown> | undefined
  )?.maxTextureDimension2D;
  return typeof raw === 'number' && Number.isFinite(raw) && raw > 0
    ? raw
    : undefined;
}

function assertFinitePayload(
  values: Float32Array,
  label: string,
  nonNegative: boolean,
): void {
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (!Number.isFinite(value) || (nonNegative && value < 0)) {
      throw new RangeError(
        `environment ${label}[${index}] must be finite` +
        `${nonNegative ? ' and non-negative' : ''}; got ${String(value)}`,
      );
    }
  }
}

/**
 * WebGPU uploads must preserve a typed-array view's byte offset and length.
 * Deno's WebGPU declarations reject SharedArrayBuffer-backed views, so copy
 * only that uncommon case into an ArrayBuffer-backed view.
 */
function gpuUploadFloat32View(
  values: Float32Array,
): Float32Array<ArrayBuffer> {
  return values.buffer instanceof ArrayBuffer
    ? values as Float32Array<ArrayBuffer>
    : new Float32Array(values);
}

/** Write the EnvParams uniform. */
function writeParams(
  device: GPUDevice,
  buf: GPUBuffer,
  hasEnv: boolean,
  width: number,
  height: number,
  rotationY: number,
  intensity: number,
): void {
  const ab = new ArrayBuffer(ENV_PARAMS_BYTES);
  const u = new Uint32Array(ab);
  const f = new Float32Array(ab);
  u[0] = hasEnv ? 1 : 0;
  u[1] = width >>> 0;
  u[2] = height >>> 0;
  f[3] = rotationY;
  f[4] = intensity;
  device.queue.writeBuffer(buf, 0, ab);
}

/** Create the placeholder (no-HDRI) env resource set. */
export function createPlaceholderEnvironment(device: GPUDevice): EnvironmentTextures {
  const textures: GPUTexture[] = [];
  let paramsBuffer: GPUBuffer | null = null;
  try {
    const map = createTex(device, 'vitrum.env.map.placeholder', 1, 1, 'rgba32float');
    textures.push(map);
    const pdf = createTex(device, 'vitrum.env.pdf.placeholder', 1, 1, 'r32float');
    textures.push(pdf);
    const marginal = createTex(device, 'vitrum.env.marginal.placeholder', 1, 1, 'r32float');
    textures.push(marginal);
    const conditional = createTex(device, 'vitrum.env.conditional.placeholder', 1, 1, 'r32float');
    textures.push(conditional);
    const sampler = device.createSampler({ label: 'vitrum.env.sampler' });
    paramsBuffer = device.createBuffer({
      label: 'vitrum.env.params',
      size: ENV_PARAMS_BYTES,
      usage: UNIFORM | BUF_COPY_DST,
    });
    writeParams(device, paramsBuffer, false, 0, 0, 0, 0);
    return {
      map,
      pdf,
      marginal,
      conditional,
      sampler,
      paramsBuffer,
      rotationY: 0,
      intensity: 0,
      hasDirectionalEnvironment: false,
    };
  } catch (error) {
    for (const texture of textures) {
      try {
        texture.destroy();
      } catch {
        // Preserve the allocation/write failure that triggered cleanup.
      }
    }
    try {
      paramsBuffer?.destroy();
    } catch {
      // Preserve the allocation/write failure that triggered cleanup.
    }
    throw error;
  }
}

type EnvironmentTextureSet = Pick<EnvironmentTextures, 'map' | 'pdf' | 'marginal' | 'conditional'>;

function destroyTextureSet(set: EnvironmentTextureSet): void {
  for (const texture of [
    set.map,
    set.pdf,
    set.marginal,
    set.conditional,
  ]) {
    try {
      texture.destroy();
    } catch {
      // Resource retirement is best-effort and must not mask an upload error.
    }
  }
}

function createEnvironmentTextureSet(
  device: GPUDevice,
  width: number,
  height: number,
  placeholder: boolean,
): EnvironmentTextureSet {
  const suffix = placeholder ? '.placeholder' : '';
  const textures: GPUTexture[] = [];
  try {
    const map = createTex(device, `vitrum.env.map${suffix}`, width, height, 'rgba32float');
    textures.push(map);
    const pdf = createTex(device, `vitrum.env.pdf${suffix}`, width, height, 'r32float');
    textures.push(pdf);
    const marginal = createTex(device, `vitrum.env.marginal${suffix}`, height, 1, 'r32float');
    textures.push(marginal);
    const conditional = createTex(
      device,
      `vitrum.env.conditional${suffix}`,
      width,
      height,
      'r32float',
    );
    textures.push(conditional);
    return { map, pdf, marginal, conditional };
  } catch (error) {
    for (const texture of textures) {
      try {
        texture.destroy();
      } catch {
        // Preserve the allocation failure that triggered cleanup.
      }
    }
    throw error;
  }
}

/** Transactionally replace directional environment texture content.
 *
 * All four candidate textures are allocated and every queue write is validated
 * before the live texture set is retired. This intentionally replaces even a
 * same-sized environment: preserving an old texture identity is not worth
 * exposing a partially updated map/CDF set when a later upload fails.
 */
export function uploadEnvironment(
  device: GPUDevice,
  prev: EnvironmentTextures,
  data: DirectionalEnvData,
  rotationY: number,
  intensity: number,
): EnvironmentTextures {
  const { width, height } = data;
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError(
      `environment dimensions must be positive safe integers; got ${width}x${height}`,
    );
  }
  const packedRotationY = packWalkaroundEnvironmentRotationF32(
    rotationY,
    'environment rotation',
  );
  const packedIntensity =
    assertWalkaroundEnvironmentMapScaleEnvelopeF32(data.map, intensity);
  const pixels = width * height;
  if (
    data.map.length !== pixels * 4 ||
    data.pdf.length !== pixels ||
    data.marginal.length !== height * 4 ||
    data.conditional.length !== pixels * 4
  ) {
    throw new RangeError('environment map/CDF payload lengths do not match its dimensions');
  }
  const maxDimension = reportedTextureDimensionLimit(device);
  if (
    maxDimension !== undefined
    && (width > maxDimension || height > maxDimension)
  ) {
    throw new RangeError(
      `environment ${width}x${height} exceeds device ` +
      `maxTextureDimension2D=${maxDimension}; no textures were allocated`,
    );
  }
  assertFinitePayload(data.map, 'radiance', true);
  assertFinitePayload(data.pdf, 'pdf', true);
  assertFinitePayload(data.marginal, 'marginal CDF', false);
  assertFinitePayload(data.conditional, 'conditional CDF', false);
  const marginalR = extractR(data.marginal, height);
  const conditionalR = extractR(data.conditional, pixels);
  const mapUpload = gpuUploadFloat32View(data.map);
  const pdfUpload = gpuUploadFloat32View(data.pdf);
  const next = createEnvironmentTextureSet(device, width, height, false);

  try {
    // Keep radiance and its importance density in independent full-range
    // textures. A half-float combined payload silently turns bright HDR texels
    // and narrow-lobe PDFs into infinity, invalidating both MIS and radiometry.
    device.queue.writeTexture(
      { texture: next.map },
      mapUpload,
      { bytesPerRow: width * 4 * 4, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.writeTexture(
      { texture: next.pdf },
      pdfUpload,
      { bytesPerRow: width * 4, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );

    // env_marginal: r32float, H×1 (width=height, height=1).
    // The WGSL consumer reads: textureLoad(env_marginal, vec2i(row, 0), 0)
    // where row ∈ [0, H), confirming the marginal is H wide and 1 tall.
    // The marginal Float32Array is height×4 (RGBA); extract .r into a height-long row.
    device.queue.writeTexture(
      { texture: next.marginal },
      marginalR.buffer,
      { bytesPerRow: height * 4, rowsPerImage: 1 },
      { width: height, height: 1, depthOrArrayLayers: 1 },
    );

    // env_conditional: r32float, W×H. Extract .r from the RGBA Float32Array.
    device.queue.writeTexture(
      { texture: next.conditional },
      conditionalR.buffer,
      { bytesPerRow: width * 4, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );

    writeParams(
      device,
      prev.paramsBuffer,
      true,
      width,
      height,
      packedRotationY,
      packedIntensity,
    );
  } catch (error) {
    destroyTextureSet(next);
    throw error;
  }
  destroyTextureSet(prev);
  return {
    ...next,
    sampler: prev.sampler,
    paramsBuffer: prev.paramsBuffer,
    rotationY: packedRotationY,
    intensity: packedIntensity,
    hasDirectionalEnvironment: true,
  };
}

/**
 * Build a complete environment generation without touching any live
 * generation. The returned set owns its texture set, sampler, and params
 * buffer; callers may therefore stage texture views and dependent DDGI state
 * before publishing it with a pointer swap.
 */
export function createEnvironmentCandidate(
  device: GPUDevice,
  data: DirectionalEnvData | null,
  rotationY: number,
  intensity: number,
): EnvironmentTextures {
  const placeholder = createPlaceholderEnvironment(device);
  if (data == null) return placeholder;
  try {
    return uploadEnvironment(
      device,
      placeholder,
      data,
      rotationY,
      intensity,
    );
  } catch (error) {
    disposeEnvironment(placeholder);
    throw error;
  }
}

/** Reset a resource set back to the no-HDRI placeholder (hasEnv=0). */
export function clearEnvironment(
  device: GPUDevice,
  prev: EnvironmentTextures,
): EnvironmentTextures {
  const next = createEnvironmentTextureSet(device, 1, 1, true);
  try {
    writeParams(device, prev.paramsBuffer, false, 0, 0, 0, 0);
  } catch (error) {
    destroyTextureSet(next);
    throw error;
  }
  destroyTextureSet(prev);
  return {
    ...next,
    sampler: prev.sampler,
    paramsBuffer: prev.paramsBuffer,
    rotationY: 0,
    intensity: 0,
    hasDirectionalEnvironment: false,
  };
}

export function disposeEnvironment(env: EnvironmentTextures): void {
  // Retirement is deliberately non-throwing. A hostile/mock destroy must not
  // mask the mutation's primary failure or prevent the remaining generation
  // resources from being released.
  const resources: readonly { destroy(): void }[] = [
    env.map,
    env.pdf,
    env.marginal,
    env.conditional,
    env.paramsBuffer,
  ];
  for (const resource of resources) {
    try {
      resource.destroy();
    } catch {
      // Continue best-effort retirement.
    }
  }
}

/** Extract the .r channel from an interleaved RGBA Float32Array into a packed
 *  Float32Array of `count` elements. */
function extractR(rgba: Float32Array, count: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = rgba[i * 4] ?? 0;
  }
  return out;
}

/** Convert a Float32Array to IEEE-754 half-precision (Uint16Array). */
export function float32ArrayToFloat16(src: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i += 1) {
    out[i] = floatToHalf(src[i] ?? 0);
  }
  return out;
}

/** Single-value Float32 → Float16 (round-to-nearest-even, with inf/denormal
 *  handling). Standard bit-twiddling conversion. */
const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);
function floatToHalf(value: number): number {
  f32View[0] = value;
  const x = u32View[0]!;
  const sign = (x >>> 16) & 0x8000;
  const exp32 = (x >>> 23) & 0xff;
  const mant32 = x & 0x007fffff;
  if (exp32 === 0xff) {
    return sign | (mant32 === 0 ? 0x7c00 : 0x7e00);
  }

  let exp16 = exp32 - 127 + 15;
  if (exp16 >= 0x1f) return sign | 0x7c00;
  if (exp16 <= 0) {
    if (exp16 < -10) return sign;
    const mantissa = mant32 | 0x00800000;
    const shift = 14 - exp16;
    let rounded = mantissa >>> shift;
    const remainder = mantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (rounded & 1) !== 0)) {
      rounded++;
    }
    return sign | rounded;
  }

  let mant16 = mant32 >>> 13;
  const remainder = mant32 & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (mant16 & 1) !== 0)) {
    mant16++;
    if (mant16 === 0x400) {
      mant16 = 0;
      exp16++;
      if (exp16 >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (exp16 << 10) | mant16;
}
