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
// NOTE: like the fork, the per-pixel weight is plain Rec.709 luminance WITHOUT a
// solid-angle (sin θ) term — the fork carries a TODO to add it but does not. We
// match the fork so the GLSL `equirect_sampling` decoder behaves identically.

import type { SceneEnvironment } from '@vitrum/core';
import type { EnvTextureData } from './sceneTextures.js';

/** The opaque @vitrum/core `EnvironmentMapRef` is, for the HDRI path, a raw
 *  equirect payload: row-major **RGB** float pixels (3 floats/pixel),
 *  `data.length >= width * height * 3`. Matches the pt-webgpu interpretation
 *  (`environmentPacking.ts:76`). */
interface HdriPayload {
  readonly width?: number;
  readonly height?: number;
  readonly data?: ArrayLike<number>;
}

function colorToLuminance(r: number, g: number, b: number): number {
  // https://en.wikipedia.org/wiki/Relative_luminance
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
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
 * Build the equirect radiance grid + marginal/conditional importance CDFs for an
 * HDRI environment. For non-`hdri` environment kinds (or an HDRI lacking CPU
 * pixel data), returns an all-null `EnvTextureData` — the integrator falls back
 * to its uniform/sky env path.
 */
export function buildEquirectInfo(env: SceneEnvironment): EnvTextureData {
  if (env.kind !== 'hdri') {
    return EMPTY_ENV;
  }

  const hdri = env.hdri as HdriPayload;
  const width = Number(hdri?.width ?? 0);
  const height = Number(hdri?.height ?? 0);
  const src = hdri?.data;

  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0 ||
    src == null ||
    typeof src.length !== 'number' ||
    src.length < width * height * 3
  ) {
    // H7 (2026-06-09): warn explicitly so the host knows the HDRI was silently
    // dropped. Without this, a missing/incorrectly-shaped HDRI payload results in
    // a flat-black environment with no error signal (a frequent source of confusion
    // during host integration). The EMPTY_ENV fallback is intentional — the
    // integrator's uniform/procedural-sky path takes over.
    console.warn(
      '[pt-webgl2] HDRI environment is present (kind="hdri") but has no usable CPU pixel data. ' +
        'pt-webgl2 requires a raw {width, height, data} RGB float payload (or use the ' +
        'sceneFromThreeJS on-ramp with texturePayload:"raw"). ' +
        'The environment will be ignored (EMPTY_ENV fallback). ' +
        // eslint-disable-next-line @typescript-eslint/no-base-to-string -- diagnostic warning; [object Object] output is acceptable here
        `Received hdri handle: ${String(hdri)}, width=${width}, height=${height}, ` +
        `src type=${src == null ? 'null' : Object.prototype.toString.call(src)}.`,
    );
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
      const r = Number(src[i * 3] ?? 0);
      const g = Number(src[i * 3 + 1] ?? 0);
      const b = Number(src[i * 3 + 2] ?? 0);

      map[i * 4 + 0] = r;
      map[i * 4 + 1] = g;
      map[i * 4 + 2] = b;
      map[i * 4 + 3] = 0;

      const weight = colorToLuminance(r, g, b);
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
      const col = binarySearchFindClosestIndexOf(
        cdfConditional,
        dist,
        y * width,
        width,
      );
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
