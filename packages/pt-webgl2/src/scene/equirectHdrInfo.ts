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
// NOTE: unlike the original fork TODO, the per-pixel importance weight includes
// the equirectangular texel solid-angle term sin(theta). The GLSL decoder's
// returned PDF cancels that same factor back to a per-steradian density so the
// sample CDF and MIS PDF stay measure-consistent.

import type { EngineWarning, SceneEnvironment } from '@vitrum/core';
import { bakePreethamSkyEquirect, readEnvironmentMapPixels } from '@vitrum/shared-samplers';
import type { EnvTextureData } from './sceneTextures.js';

interface EquirectSource {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number> | undefined;
}

export interface EquirectInfoBuildOptions {
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: string;
  readonly warningMethod?: string;
}

function emitEnvironmentWarning(
  options: EquirectInfoBuildOptions | undefined,
  warning: Omit<EngineWarning, 'backend' | 'phase' | 'method'>,
): void {
  const routed: EngineWarning = {
    ...warning,
    backend: 'pt-webgl2',
    phase: options?.warningPhase ?? 'scene-upload',
    ...(options?.warningMethod != null ? { method: options.warningMethod } : {}),
  };
  if (options?.onWarning != null) {
    options.onWarning(routed);
  } else {
    console.warn(routed.message);
  }
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
  let hdriHandleForDiagnostic: unknown = null;
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
    hdriHandleForDiagnostic = env.hdri;
    const raw = env.hdri as {
      readonly width?: unknown;
      readonly height?: unknown;
      readonly data?: ArrayLike<number>;
      readonly image?: {
        readonly width?: unknown;
        readonly height?: unknown;
        readonly data?: ArrayLike<number>;
      };
    } | null;
    const hdri = readEnvironmentMapPixels(env.hdri);
    source = hdri == null
      ? {
          width: Number(raw?.width ?? raw?.image?.width ?? 0),
          height: Number(raw?.height ?? raw?.image?.height ?? 0),
          data: raw?.data ?? raw?.image?.data,
        }
      : { width: hdri.width, height: hdri.height, data: hdri.data };
  }

  const width = source.width;
  const height = source.height;
  const src = source.data;

  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0 ||
    src == null ||
    typeof src.length !== 'number' ||
    src.length < width * height * 4
  ) {
    // H7 (2026-06-09): warn explicitly so the host knows the HDRI was dropped.
    // Without this, a missing/incorrectly-shaped HDRI payload results in a
    // flat-black environment with no error signal (a frequent source of
    // confusion during host integration).
    const message =
      '[pt-webgl2] HDRI environment is present (kind="hdri") but has no usable CPU pixel data. ' +
        'pt-webgl2 requires a raw {width, height, data} or DataTexture-shaped {image:{width,height,data}} RGB/RGBA payload (or use the ' +
        'sceneFromThreeJS on-ramp with texturePayload:"raw"). ' +
        'The environment will be ignored (EMPTY_ENV fallback). ' +
        // eslint-disable-next-line @typescript-eslint/no-base-to-string -- diagnostic warning; [object Object] output is acceptable here
        `Received hdri handle: ${String(hdriHandleForDiagnostic)}, width=${width}, height=${height}, ` +
        `src type=${src == null ? 'null' : Object.prototype.toString.call(src)}.`;
    emitEnvironmentWarning(options, {
      code: 'pt-webgl2.hdri-unreadable',
      message,
      details: {
        width,
        height,
        sourceType: src == null ? 'null' : Object.prototype.toString.call(src),
        handleType: hdriHandleForDiagnostic == null
          ? 'null'
          : Object.prototype.toString.call(hdriHandleForDiagnostic),
        handle: String(hdriHandleForDiagnostic),
      },
    });
    return EMPTY_ENV;
  }

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
