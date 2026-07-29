/**
 * B3 (road-to-100) — directional IBL: equirect radiance map + importance CDFs.
 *
 * Builds the GPU-uploadable directional environment payload from a raw equirect
 * HDRI so walkaround-hybrid can sample the ACTUAL map by direction (sky-miss
 * pixels in ris/risGi/shade, and GI rays that escape the scene), instead of
 * collapsing the dome to a single `skyTint · skyIrradiance` scalar.
 *
 * The scalar-tint path (resolveHybridEnvironment's `hdri-raw-average`) REMAINS
 * the fallback the host gets when there is no pixel-backed HDRI; this module is
 * purely ADDITIVE — it is built only when a raw {width,height,data} payload is
 * present and is consumed only by the new directional WGSL lookup.
 *
 * Algorithm — PBRT InfiniteAreaLight 2D distribution (marginal row CDF + per-row
 * conditional column CDF), with the solid-angle (sinθ) weighting that makes the
 * sampler distribute over the SPHERE (not the equirect texel grid). This mirrors
 * the pt-webgpu environmentPacking.ts per-texel pdf (also sinθ-weighted) so the
 * two backends importance-sample the SAME map identically — the V28 A/B target.
 *
 * Provenance: PBRT (Pharr/Jakob/Humphreys) §12.6 InfiniteAreaLight; the marginal/
 * conditional layout follows gkjohnson/three-gpu-pathtracer EquirectHdrInfoUniform
 * (MIT, see CREDITS.md) as ported THREE-free in pt-webgl2's equirectHdrInfo.ts.
 * This implementation includes the sinθ solid-angle term omitted by the
 * provenance fork, so the per-texel pdf is a true directional pdf matching
 * pt-webgpu and enabling correct env-importance MIS bookkeeping.
 */

export interface DirectionalEnvData {
  /** Equirect width in texels. */
  readonly width: number;
  /** Equirect height in texels. */
  readonly height: number;
  /**
   * Radiance + per-texel directional pdf, RGBA32F, row-major (width × height).
   * .rgb = unit-intensity radiance (host applies intensity at sample time);
   * .a   = solid-angle pdf p(ω) of selecting this texel (the importance pdf the
   *        WGSL importance sampler returns; 0 for all-black maps).
   */
  readonly map: Float32Array;
  /**
   * Forward marginal row CDF, 1 × height, value in .r of an RGBA32F texel.
   * The shader binary-searches the first entry strictly greater than ξ.
   */
  readonly marginal: Float32Array;
  /**
   * Forward conditional column CDF, width × height, value in .r. Each row ends
   * at one; zero-weight rows carry a uniform fallback CDF.
   */
  readonly conditional: Float32Array;
  /** Unnormalised sinθ-weighted luminance integral (0 ⇒ all-black, no IBL). */
  readonly totalWeight: number;
}

interface RawEquirectPayload {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly stride: 3 | 4;
}

function colorToLuminance(r: number, g: number, b: number): number {
  // Rec.709 relative luminance (matches resolveHybridEnvironment + pt-webgpu).
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Build the directional env payload from a raw equirect payload, or return null
 * for an all-black / degenerate map (the caller keeps the scalar-tint fallback).
 */
export function buildDirectionalEnv(payload: RawEquirectPayload): DirectionalEnvData | null {
  const { width, height, data, stride } = payload;
  const pixelCount = width * height;
  if (pixelCount <= 0) return null;

  const map = new Float32Array(pixelCount * 4);
  const cdfConditional = new Float32Array(pixelCount);
  const cdfMarginal = new Float32Array(height);

  // Per-row sinθ (the solid-angle weight is constant across a row). PBRT uses the
  // row-centre θ = (y+0.5)/height · π.
  let totalWeight = 0;
  let cumulativeMarginal = 0;
  for (let y = 0; y < height; y += 1) {
    const theta = ((y + 0.5) / height) * Math.PI;
    const sinTheta = Math.max(Math.sin(theta), 0);
    let cumulativeRow = 0;
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const off = i * stride;
      const r = Number(data[off] ?? 0);
      const g = Number(data[off + 1] ?? 0);
      const b = Number(data[off + 2] ?? 0);
      const rr = Number.isFinite(r) && r > 0 ? r : 0;
      const gg = Number.isFinite(g) && g > 0 ? g : 0;
      const bb = Number.isFinite(b) && b > 0 ? b : 0;
      map[i * 4] = rr;
      map[i * 4 + 1] = gg;
      map[i * 4 + 2] = bb;
      const weight = colorToLuminance(rr, gg, bb) * sinTheta;
      cumulativeRow += weight;
      totalWeight += weight;
      cdfConditional[i] = cumulativeRow;
    }
    if (cumulativeRow !== 0) {
      const base = y * width;
      for (let x = 0; x < width; x += 1) {
        cdfConditional[base + x] = cdfConditional[base + x]! / cumulativeRow;
      }
    } else {
      // A zero-weight row has zero marginal probability, but keeping its
      // conditional CDF valid makes the GPU lookup total even at exact
      // floating-point boundaries and under future marginal quantization.
      const base = y * width;
      for (let x = 0; x < width; x += 1) {
        cdfConditional[base + x] = (x + 1) / width;
      }
    }
    cumulativeMarginal += cumulativeRow;
    cdfMarginal[y] = cumulativeMarginal;
  }

  if (!(totalWeight > 0) || !Number.isFinite(totalWeight)) return null;

  for (let y = 0; y < height; y += 1) {
    cdfMarginal[y] = cdfMarginal[y]! / cumulativeMarginal;
  }

  // Per-texel solid-angle pdf p(ω) = pmf_texel / dω_texel, where
  //   pmf_texel = weight_texel / totalWeight
  //   dω_texel  = (2π/width)·(π/height)·sinθ   (equirect texel solid angle)
  // This is the directional pdf the WGSL importance sampler returns (.a lane),
  // matching pt-webgpu's environmentPacking per-texel pdf exactly.
  const dOmegaBase = ((2 * Math.PI) / width) * (Math.PI / height);
  for (let y = 0; y < height; y += 1) {
    const theta = ((y + 0.5) / height) * Math.PI;
    const sinTheta = Math.max(Math.sin(theta), 1e-5);
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const weight = colorToLuminance(map[i * 4]!, map[i * 4 + 1]!, map[i * 4 + 2]!) * sinTheta;
      const pmf = weight / totalWeight;
      map[i * 4 + 3] = pmf / (dOmegaBase * sinTheta);
    }
  }

  // Upload the exact forward CDFs. The shader binary-searches these arrays,
  // so each texel is selected over an interval equal to its true PMF rather
  // than over one uniformly quantized inverse-table bucket.
  const marginal = new Float32Array(height * 4);
  for (let i = 0; i < height; i += 1) {
    marginal[i * 4] = cdfMarginal[i]!;
  }

  const conditional = new Float32Array(pixelCount * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      conditional[i * 4] = cdfConditional[i]!;
    }
  }

  return { width, height, map, marginal, conditional, totalWeight };
}
