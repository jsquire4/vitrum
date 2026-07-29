/**
 * B3 (road-to-100) — directional IBL GPU resources for the scene bind group
 * (bindings 15-19): the equirect radiance map + the PBRT 2D-distribution
 * importance CDFs as textures (so the scene group stays at three versioned
 * storage arenas), plus the small EnvParams uniform.
 *
 * A 1×1 placeholder set + envParams.hasEnv=0 backs non-HDRI scenes so the bind
 * group is always valid; the WGSL `environmentSample` helpers fall back to the
 * scalar sky (skyTint·skyIrradiance) when hasEnv==0 → no-HDRI byte-identity.
 */

import type { DirectionalEnvData } from '../environment/equirectDirectional.js';

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
    const map = createTex(device, 'vitrum.env.map.placeholder', 1, 1, 'rgba16float');
    textures.push(map);
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
      marginal,
      conditional,
      sampler,
      paramsBuffer,
      rotationY: 0,
      intensity: 0,
      hasDirectionalEnvironment: false,
    };
  } catch (error) {
    for (const texture of textures) texture.destroy();
    paramsBuffer?.destroy();
    throw error;
  }
}

type EnvironmentTextureSet = Pick<EnvironmentTextures, 'map' | 'marginal' | 'conditional'>;

function destroyTextureSet(set: EnvironmentTextureSet): void {
  set.map.destroy();
  set.marginal.destroy();
  set.conditional.destroy();
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
    const map = createTex(device, `vitrum.env.map${suffix}`, width, height, 'rgba16float');
    textures.push(map);
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
    return { map, marginal, conditional };
  } catch (error) {
    for (const texture of textures) texture.destroy();
    throw error;
  }
}

/** Transactionally replace directional environment texture content.
 *
 * All three candidate textures are allocated and every queue write is validated
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
  if (!Number.isFinite(rotationY) || !Number.isFinite(intensity) || intensity < 0) {
    throw new RangeError('environment rotation must be finite and intensity finite/non-negative');
  }
  const pixels = width * height;
  if (
    data.map.length !== pixels * 4 ||
    data.marginal.length !== height * 4 ||
    data.conditional.length !== pixels * 4
  ) {
    throw new RangeError('environment map/CDF payload lengths do not match its dimensions');
  }
  const next = createEnvironmentTextureSet(device, width, height, false);
  const mapHalf = float32ArrayToFloat16(data.map);
  const marginalR = extractR(data.marginal, height);
  const conditionalR = extractR(data.conditional, pixels);

  try {
    // env_map: rgba16float ← Float32 RGBA. WebGPU writeTexture accepts a Float32
    // source for rgba16float? No — the bytes must already be half-float. We convert
    // the radiance + pdf to Float16 on the CPU.
    device.queue.writeTexture(
      { texture: next.map },
      mapHalf.buffer,
      { bytesPerRow: width * 4 * 2, rowsPerImage: height },
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

    writeParams(device, prev.paramsBuffer, true, width, height, rotationY, intensity);
  } catch (error) {
    destroyTextureSet(next);
    throw error;
  }
  destroyTextureSet(prev);
  return {
    ...next,
    sampler: prev.sampler,
    paramsBuffer: prev.paramsBuffer,
    rotationY,
    intensity,
    hasDirectionalEnvironment: true,
  };
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
  env.map.destroy();
  env.marginal.destroy();
  env.conditional.destroy();
  env.paramsBuffer.destroy();
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
