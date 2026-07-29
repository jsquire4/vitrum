// Equirectangular environment importance-sampling builder — THREE-free port of
// the fork's `EquirectHdrInfoUniform.js` (three-gpu-pathtracer) onto the
// @vitrum/core `SceneEnvironment` contract. Produces the CPU float grids the
// WebGL2 path tracer uploads (the GL format conversion to RedFormat-HalfFloat is
// the integrator's concern — this module only produces the float data + dims).
//
// Provenance: gkjohnson/three-gpu-pathtracer EquirectHdrInfoUniform.js (MIT);
// see CREDITS.md. Algorithm: PBRT InfiniteAreaLight 2D distribution (marginal
// row CDF + per-row conditional column CDF). Layout spec:
// plan/three-removal/03-scene-bvh-packers.md §6.
//
// Outputs (mapping a uniform random number → a centred equirect texel):
//   marginal     1×height : random → row-centre v  (which row to sample)
//   conditional  width×height : random → column-centre u (which column, per row)
//   map          width×height : the equirect radiance (RGB, .a unused here)
//   totalSum     unnormalized luminance integral over all pixels
//
// The per-pixel importance weight includes the equirectangular texel
// solid-angle term sin(theta). The GLSL decoder's
// returned PDF cancels that same factor back to a per-steradian density so the
// sample CDF and MIS PDF stay measure-consistent.

import type { EngineWarning, SceneEnvironment } from '@vitrum/core';
import {
  bakePreethamSkyEquirect,
  readEnvironmentMapPixels,
  type EnvironmentMapHandleHint,
  type EnvironmentMapPixels,
} from '@vitrum/shared-samplers';
import type { EnvTextureData } from './sceneTextures.js';

interface EquirectSource {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
}

export interface EquirectInfoBuildOptions {
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: string;
  readonly warningMethod?: string;
}

function hdriPayloadError(
  handle: unknown,
  options: EquirectInfoBuildOptions | undefined,
  reason: string,
): Error {
  const operation = options?.warningMethod != null
    ? ` during ${options.warningMethod}`
    : '';
  return new Error(
    `[pt-webgl2] authored HDRI${operation} is not CPU-readable: ${reason}. ` +
    'Provide one coherent raw { width, height, data } or DataTexture-shaped image ' +
    'with exact RGB/RGBA typed-array pixels. ' +
    `Handle type: ${handle == null ? 'null' : Object.prototype.toString.call(handle)}`,
  );
}

function describePayloadValue(value: unknown): string {
  return value !== null && typeof value === 'object'
    ? Object.prototype.toString.call(value)
    : String(value);
}

type HdriPayload = {
  readonly width?: unknown;
  readonly height?: unknown;
  readonly data?: ArrayLike<number>;
};

type HdriHandle = HdriPayload & {
  readonly image?: HdriPayload;
  readonly __vitrum_hint__?: EnvironmentMapHandleHint;
  readonly channels?: unknown;
  readonly dataType?: unknown;
  readonly colorSpace?: unknown;
};

const HDRI_DATA_TYPES = new Set([
  'uint8',
  'uint16',
  'float16',
  'half-float',
  'float32',
]);
const HDRI_CPU_FLOAT_BUDGET = (512 * 1024 * 1024) / 4;

function strictHdriPixels(
  handle: unknown,
  options: EquirectInfoBuildOptions | undefined,
): EnvironmentMapPixels {
  if (handle == null || typeof handle !== 'object') {
    throw hdriPayloadError(handle, options, 'the handle is null or opaque');
  }
  const h = handle as HdriHandle;
  const payload = h.data != null ? h : h.image;
  const src = payload?.data;
  const width = payload?.width;
  const height = payload?.height;
  if (src == null || typeof src.length !== 'number') {
    throw hdriPayloadError(handle, options, 'no raw or DataTexture-shaped pixel data was supplied');
  }
  if (
    typeof width !== 'number' ||
    typeof height !== 'number' ||
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width <= 0 ||
    height <= 0
  ) {
    throw hdriPayloadError(
      handle,
      options,
      `width and height must be positive safe integers (received ${String(width)}×${String(height)})`,
    );
  }
  if (!Number.isSafeInteger(src.length) || src.length < 0) {
    throw hdriPayloadError(
      handle,
      options,
      `data.length must be a finite non-negative safe integer (received ${String(src.length)})`,
    );
  }
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount)) {
    throw hdriPayloadError(handle, options, 'width×height exceeds the safe integer range');
  }
  // Peak construction retains decoded RGBA plus map/CDF outputs: 13 floats per
  // pixel and five per row. Guard that complete allocation before decoding.
  const constructionFloats = BigInt(pixelCount) * 13n + BigInt(height) * 5n;
  if (constructionFloats > BigInt(HDRI_CPU_FLOAT_BUDGET)) {
    throw hdriPayloadError(
      handle,
      options,
      `the ${width}×${height} decode/CDF payload exceeds the 512 MiB CPU staging budget`,
    );
  }

  const directHint = h.__vitrum_hint__;
  if (directHint != null && (typeof directHint !== 'object' || Array.isArray(directHint))) {
    throw hdriPayloadError(handle, options, '__vitrum_hint__ must be an object');
  }
  const hintedChannels = directHint?.channels ?? h.channels;
  const hintedDataType = directHint?.dataType ?? h.dataType;
  const hintedColorSpace = directHint?.colorSpace ?? h.colorSpace;
  if (hintedChannels != null && hintedChannels !== 3 && hintedChannels !== 4) {
    throw hdriPayloadError(
      handle,
      options,
        `channels must be 3 (RGB) or 4 (RGBA), received ${describePayloadValue(hintedChannels)}`,
      );
    }
    if (
      hintedDataType != null &&
      (typeof hintedDataType !== 'string' || !HDRI_DATA_TYPES.has(hintedDataType))
    ) {
      throw hdriPayloadError(
        handle,
        options,
        `dataType "${describePayloadValue(hintedDataType)}" is unsupported`,
      );
  }
  if (
    hintedColorSpace != null &&
    hintedColorSpace !== 'linear' &&
    hintedColorSpace !== 'srgb'
  ) {
      throw hdriPayloadError(
        handle,
        options,
        `colorSpace "${describePayloadValue(hintedColorSpace)}" is unsupported`,
      );
  }

  let channels: 3 | 4;
  if (hintedChannels === 3 || hintedChannels === 4) {
    channels = hintedChannels;
  } else if (src.length === pixelCount * 3) {
    channels = 3;
  } else if (src.length === pixelCount * 4) {
    channels = 4;
  } else {
    throw hdriPayloadError(
      handle,
      options,
      `data length ${src.length} must exactly equal ${pixelCount * 3} (RGB) or ${pixelCount * 4} (RGBA)`,
    );
  }
  const expectedLength = pixelCount * channels;
  if (src.length !== expectedLength) {
    throw hdriPayloadError(
      handle,
      options,
      `data length ${src.length} does not equal width×height×channels (${expectedLength})`,
    );
  }

  const backingType = Object.prototype.toString.call(src);
  let dataType = hintedDataType;
  if (dataType == null) {
    if (backingType === '[object Uint8Array]' || backingType === '[object Uint8ClampedArray]') {
      dataType = 'uint8';
    } else if (backingType === '[object Uint16Array]') {
      dataType = 'uint16';
    } else if (backingType === '[object Float32Array]') {
      dataType = 'float32';
    } else {
      throw hdriPayloadError(
        handle,
        options,
        `pixel backing ${backingType} cannot be inferred; use Uint8Array, Uint16Array, or Float32Array`,
      );
    }
  }
  const compatibleBacking =
    (dataType === 'uint8' && (
      backingType === '[object Uint8Array]' || backingType === '[object Uint8ClampedArray]'
    )) ||
    ((dataType === 'uint16' || dataType === 'float16' || dataType === 'half-float') &&
      backingType === '[object Uint16Array]') ||
    (dataType === 'float32' && backingType === '[object Float32Array]');
  if (!compatibleBacking) {
    const expected = dataType === 'uint8'
      ? 'Uint8Array or Uint8ClampedArray'
      : dataType === 'float32'
        ? 'Float32Array'
        : 'Uint16Array';
    throw hdriPayloadError(
      handle,
      options,
      `dataType "${String(dataType)}" requires ${expected}, received ${backingType}`,
    );
  }

  const decoded = readEnvironmentMapPixels(handle);
  if (
    decoded == null ||
    decoded.width !== width ||
    decoded.height !== height ||
    decoded.sourceChannels !== channels
  ) {
    throw hdriPayloadError(
      handle,
      options,
      'every RGB radiance value and optional alpha value must decode to a finite float',
    );
  }
  for (let p = 0; p < pixelCount; p += 1) {
    const base = p * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      const radiance = decoded.data[base + channel]!;
      if (!Number.isFinite(radiance) || radiance < 0) {
        throw hdriPayloadError(
          handle,
          options,
          `radiance must be finite and nonnegative (pixel ${p}, channel ${channel}, value ${String(radiance)})`,
        );
      }
    }
  }
  return decoded;
}

function colorToLuminance(r: number, g: number, b: number): number {
  // https://en.wikipedia.org/wiki/Relative_luminance
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function equirectTexelSolidAngleWeight(y: number, height: number): number {
  const theta = ((y + 0.5) / height) * Math.PI;
  return Math.max(0, Math.sin(theta));
}

/**
 * Find the smallest index in `array[offset .. offset+count)` whose value is
 * `>= targetValue`, returned relative to `offset`. The array slice must be
 * non-decreasing (a CDF). Verbatim port of the fork's
 * `binarySearchFindClosestIndexOf`.
 */
function binarySearchFindClosestIndexOf(
  array: Float32Array,
  targetValue: number,
  offset = 0,
  count = array.length,
): number {
  let lower = offset;
  let upper = offset + count - 1;
  while (lower < upper) {
    const mid = (lower + upper) >> 1;
    if (array[mid]! < targetValue) {
      lower = mid + 1;
    } else {
      upper = mid;
    }
  }
  return lower - offset;
}

const EMPTY_ENV: EnvTextureData = {
  map: null,
  marginal: null,
  conditional: null,
  totalSum: 0,
};

/**
 * Build the equirect radiance grid + marginal/conditional importance CDFs for
 * HDRI and procedural-sky environments. Procedural skies are baked through the
 * shared Preetham model and then use the same equirect sampling path as HDRI.
 */
export function buildEquirectInfo(
  env: SceneEnvironment,
  options?: EquirectInfoBuildOptions,
): EnvTextureData {
  if (env.kind === 'none') {
    return EMPTY_ENV;
  }

  let source: EquirectSource;
  if (env.kind === 'procedural-sky') {
    const baked = bakePreethamSkyEquirect({
      sunDirection: env.sunDirection,
      turbidity: env.turbidity,
      rayleigh: env.rayleigh,
      mieCoefficient: env.mieCoefficient,
      mieDirectionalG: env.mieDirectionalG,
      ...(env.intensity !== undefined ? { intensity: env.intensity } : {}),
    });
    source = { width: baked.width, height: baked.height, data: baked.texels };
  } else {
    const hdri = strictHdriPixels(env.hdri, options);
    source = { width: hdri.width, height: hdri.height, data: hdri.data };
  }

  const width = source.width;
  const height = source.height;
  const src = source.data;

  const pixelCount = width * height;

  // The equirect radiance map, RGBA32F (alpha unused by this CPU side; the
  // integrator's GL upload converts to the runtime format).
  const map = new Float32Array(pixelCount * 4);

  // PBRT 2D distribution scratch (mirrors the fork's pdf/cdf arrays).
  const cdfConditional = new Float32Array(pixelCount);
  const cdfMarginal = new Float32Array(height);

  let totalSumValue = 0.0;
  let cumulativeWeightMarginal = 0.0;

  for (let y = 0; y < height; y += 1) {
    let cumulativeRowWeight = 0.0;
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const base = i * 4;
      const r = Number(src[base] ?? 0);
      const g = Number(src[base + 1] ?? 0);
      const b = Number(src[base + 2] ?? 0);

      map[i * 4 + 0] = r;
      map[i * 4 + 1] = g;
      map[i * 4 + 2] = b;
      map[i * 4 + 3] = 0;

      const weight = colorToLuminance(r, g, b) * equirectTexelSolidAngleWeight(y, height);
      cumulativeRowWeight += weight;
      totalSumValue += weight;

      cdfConditional[i] = cumulativeRowWeight;
    }

    // Row-normalize the conditional CDF to [0, 1] (skip all-black rows).
    if (cumulativeRowWeight !== 0) {
      for (let i = y * width, lEnd = y * width + width; i < lEnd; i += 1) {
        cdfConditional[i] = cdfConditional[i]! / cumulativeRowWeight;
      }
    }

    cumulativeWeightMarginal += cumulativeRowWeight;
    cdfMarginal[y] = cumulativeWeightMarginal;
  }

  // Total-normalize the marginal CDF to [0, 1] (skip an all-black map).
  if (cumulativeWeightMarginal !== 0) {
    for (let i = 0, lEnd = cdfMarginal.length; i < lEnd; i += 1) {
      cdfMarginal[i] = cdfMarginal[i]! / cumulativeWeightMarginal;
    }
  }

  // Sampled inverse-CDF tables: random ∈ [0,1] → centred texel coordinate.
  // marginal: 1×height, RedFormat in GL but RGBA32F here (value in .r).
  const marginalData = new Float32Array(height * 4);
  for (let i = 0; i < height; i += 1) {
    const dist = (i + 1) / height;
    const row = binarySearchFindClosestIndexOf(cdfMarginal, dist);
    marginalData[i * 4 + 0] = (row + 0.5) / height; // half-texel recentre
  }

  // conditional: width×height, value in .r.
  const conditionalData = new Float32Array(pixelCount * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const dist = (x + 1) / width;
      const col = binarySearchFindClosestIndexOf(cdfConditional, dist, y * width, width);
      conditionalData[i * 4 + 0] = (col + 0.5) / width; // half-texel recentre
    }
  }

  return {
    map: { data: map, width, height },
    marginal: { data: marginalData, width: height, height: 1 },
    conditional: { data: conditionalData, width, height },
    totalSum: totalSumValue,
  };
}
