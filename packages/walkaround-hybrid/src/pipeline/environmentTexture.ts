/**
 * B3 (road-to-100) — directional IBL GPU resources for the scene bind group
 * (bindings 15-19): the equirect radiance map + the PBRT 2D-distribution
 * importance CDFs as TEXTURES (so the scene group stays at the 16-storage
 * shade-pass floor), plus the small EnvParams uniform.
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
  const map = createTex(device, 'vitrum.env.map.placeholder', 1, 1, 'rgba16float');
  const marginal = createTex(device, 'vitrum.env.marginal.placeholder', 1, 1, 'r32float');
  const conditional = createTex(device, 'vitrum.env.conditional.placeholder', 1, 1, 'r32float');
  const sampler = device.createSampler({ label: 'vitrum.env.sampler' });
  const paramsBuffer = device.createBuffer({
    label: 'vitrum.env.params',
    size: ENV_PARAMS_BYTES,
    usage: UNIFORM | BUF_COPY_DST,
  });
  writeParams(device, paramsBuffer, false, 0, 0, 0, 0);
  return { map, marginal, conditional, sampler, paramsBuffer };
}

/** Write directional env data into an existing resource set.
 *
 *  When the new dimensions match the existing textures (a rotation / intensity
 *  update, or a repeated upload of the same-resolution env), the three GPU
 *  textures are reused in-place via `writeTexture` — no destroy + recreate.
 *  Reuse preserves the GPUTexture object identities, which means the scene
 *  bind-group cache key (which includes the env texture views) is unchanged
 *  and the bind group is reused without invalidation.
 *
 *  When dimensions change (first upload from a 1×1 placeholder, or a
 *  different-resolution env), the old textures are destroyed and new ones
 *  allocated at the correct size. */
export function uploadEnvironment(
  device: GPUDevice,
  prev: EnvironmentTextures,
  data: DirectionalEnvData,
  rotationY: number,
  intensity: number,
): EnvironmentTextures {
  const { width, height } = data;

  // Probe the existing texture dimensions. GPUTexture exposes .width / .height
  // as read-only properties (WebGPU spec §GPUTexture).  We compare against the
  // required map dimensions; if they match we write in-place.
  const prevW = (prev.map as unknown as { width: number }).width ?? 0;
  const prevH = (prev.map as unknown as { height: number }).height ?? 0;
  const dimsMatch = prevW === width && prevH === height;

  let map        = prev.map;
  let marginal   = prev.marginal;
  let conditional = prev.conditional;

  if (!dimsMatch) {
    // Dimensions changed — destroy the old textures and allocate new ones at
    // the required size.  Sampler + params buffer are always reused.
    prev.map.destroy();
    prev.marginal.destroy();
    prev.conditional.destroy();
    map        = createTex(device, 'vitrum.env.map',         width,  height, 'rgba16float');
    marginal   = createTex(device, 'vitrum.env.marginal',    height, 1,      'r32float');
    conditional = createTex(device, 'vitrum.env.conditional', width,  height, 'r32float');
  }

  // env_map: rgba16float ← Float32 RGBA. WebGPU writeTexture accepts a Float32
  // source for rgba16float? No — the bytes must already be half-float. We convert
  // the radiance + pdf to Float16 on the CPU.
  const mapHalf = float32ArrayToFloat16(data.map);
  device.queue.writeTexture(
    { texture: map },
    mapHalf.buffer,
    { bytesPerRow: width * 4 * 2, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );

  // env_marginal: r32float, H×1 (width=height, height=1).
  // The WGSL consumer reads: textureLoad(env_marginal, vec2i(row, 0), 0)
  // where row ∈ [0, H), confirming the marginal is H wide and 1 tall.
  // The marginal Float32Array is height×4 (RGBA); extract .r into a height-long row.
  const marginalR = extractR(data.marginal, height);
  device.queue.writeTexture(
    { texture: marginal },
    marginalR.buffer,
    { bytesPerRow: height * 4, rowsPerImage: 1 },
    { width: height, height: 1, depthOrArrayLayers: 1 },
  );

  // env_conditional: r32float, W×H. Extract .r from the RGBA Float32Array.
  const conditionalR = extractR(data.conditional, width * height);
  device.queue.writeTexture(
    { texture: conditional },
    conditionalR.buffer,
    { bytesPerRow: width * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );

  writeParams(device, prev.paramsBuffer, true, width, height, rotationY, intensity);
  return { map, marginal, conditional, sampler: prev.sampler, paramsBuffer: prev.paramsBuffer };
}

/** Reset a resource set back to the no-HDRI placeholder (hasEnv=0). */
export function clearEnvironment(
  device: GPUDevice,
  prev: EnvironmentTextures,
): EnvironmentTextures {
  prev.map.destroy();
  prev.marginal.destroy();
  prev.conditional.destroy();
  const map = createTex(device, 'vitrum.env.map.placeholder', 1, 1, 'rgba16float');
  const marginal = createTex(device, 'vitrum.env.marginal.placeholder', 1, 1, 'r32float');
  const conditional = createTex(device, 'vitrum.env.conditional.placeholder', 1, 1, 'r32float');
  writeParams(device, prev.paramsBuffer, false, 0, 0, 0, 0);
  return { map, marginal, conditional, sampler: prev.sampler, paramsBuffer: prev.paramsBuffer };
}

export function disposeEnvironment(env: EnvironmentTextures): void {
  env.map.destroy();
  env.marginal.destroy();
  env.conditional.destroy();
  env.paramsBuffer.destroy();
}

/** Extract the .r channel from an interleaved RGBA Float32Array into a packed
 *  Float32Array of `count` elements. */
function extractR(rgba: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = rgba[i * 4] ?? 0;
  }
  return out;
}

/** Convert a Float32Array to IEEE-754 half-precision (Uint16Array). */
function float32ArrayToFloat16(src: Float32Array): Uint16Array {
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
  let mantissa = x & 0x007fffff;
  let exp = (x >>> 23) & 0xff;
  if (exp === 0xff) {
    // Inf / NaN.
    return sign | 0x7c00 | (mantissa ? 0x0200 : 0);
  }
  exp = exp - 127 + 15;
  if (exp >= 0x1f) {
    return sign | 0x7c00; // overflow → inf
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign; // underflow → 0
    }
    mantissa = (mantissa | 0x00800000) >>> (1 - exp);
    return sign | (mantissa >>> 13);
  }
  return sign | (exp << 10) | (mantissa >>> 13);
}
