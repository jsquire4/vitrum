import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';

/**
 * seedBlit — progressive walkaround→PT handoff, increment 1 (P8).
 *
 * Injects an initial image (the "seed", typically the real-time engine's last
 * frame at handoff) into the path tracer's accumulation buffers as a DECAYING
 * PRIOR of virtual weight `W`, so a still camera shows a plausible image
 * immediately instead of a 1-sample blizzard — WITHOUT biasing the converged
 * result.
 *
 * ── Why the converged mean is unchanged (the load-bearing correctness math) ──
 * The path-trace kernel accumulates per pixel (kernel.wgsl.ts:accumulateFrame):
 *
 *     accumBuffer[i] += vec4f(sampleColor, 1.0);            // (Σcolor, count)
 *     display        =  accum.xyz / max(accum.w, 1.0);      // running mean
 *
 * and the variance moments (varianceMomentsBuffer[i]) accumulate
 *
 *     moments += vec3(lum, lum*lum, 1.0);                   // (Σl, Σl², count)
 *
 * This kernel seeds those buffers as if `W` virtual samples of value `seedRGB`
 * had already landed:
 *
 *     accumBuffer[i]          = vec4f(seedRGB * W, W);
 *     varianceMomentsBuffer[i]= vec3 (lum(seedRGB) * W, lum(seedRGB)² * W, W);
 *
 * After `M` REAL samples accumulate (true per-pixel mean μ), the display is
 *
 *     (seedRGB·W + Σ_real) / (W + M)
 *        →  (seedRGB·W + M·μ) / (W + M)
 *        =  μ + W/(W+M) · (seedRGB − μ).
 *
 * The seed's influence is the factor W/(W+M), which decays to 0 as M→∞. So the
 * seed only biases the EARLY (still-noisy) frames; the converged mean is μ,
 * EXACTLY the no-seed result, for ANY seed value (even a deliberately wrong one
 * — that is the negative-control the GPU A/B checks). `W` trades early-frame
 * stability (large W = trust the prior longer) against decay speed (small W =
 * forget it fast); it is NOT a real-sample count and must NOT bump the engine's
 * SPP counter, or convergence/telemetry would over-report.
 *
 * ── Resolution mismatch ──
 * The seed texture may differ in size from the accum buffers (e.g. the real-time
 * engine ran at a lower resolutionFactor). We sample it with a FILTERING sampler
 * at normalised UVs (pixel centres) via `textureSampleLevel(..., 0.0)`, so any
 * source size is bilinearly resampled to the accum grid. `params.seedDim` is the
 * accum (destination) grid; the source dims live in the bound texture.
 *
 * ── Colour space ──
 * The seed is treated as LINEAR HDR radiance, matching `accumBuffer`/the kernel's
 * `display` (pre-tonemap linear). The host MUST pass a linear-light seed (e.g. the
 * real-time engine's pre-tonemap output, or an `rgba16float` linear target). An
 * sRGB-encoded seed would be a mild early-frame tint only — it still DECAYS away —
 * but the host owns getting the seed into linear light. Flagged, not assumed.
 *
 * One @compute @workgroup_size(8,8,1) invocation per accum texel; runs ONCE at
 * handoff, AFTER the accum buffers exist and AFTER clearAccumBuffer (so the seed
 * is not subsequently zeroed). `lum()` is the canonical Rec.709 luminance.
 */
export const PT_WEBGPU_SEED_BLIT_WGSL = /* wgsl */ `
${LUMINANCE_WGSL}

struct SeedParams {
  // .xy = accum (destination) width/height in texels; .zw unused (padding to vec4).
  seedDim: vec4u,
  // .x = seed weight W (virtual-sample count); .yzw padding.
  seedWeight: vec4f,
};

@group(0) @binding(0) var<uniform> seedParams: SeedParams;
@group(0) @binding(1) var seedTexture: texture_2d<f32>;
@group(0) @binding(2) var seedSampler: sampler;
// vec4f(Rsum, Gsum, Bsum, count) per pixel — the kernel's accumBuffer.
@group(0) @binding(3) var<storage, read_write> accumBuffer: array<vec4f>;
// vec3 packed as vec4 (xyz used): (Σlum, Σlum², count) per pixel.
@group(0) @binding(4) var<storage, read_write> varianceMomentsBuffer: array<vec4f>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = seedParams.seedDim.x;
  let h = seedParams.seedDim.y;
  if (gid.x >= w || gid.y >= h) {
    return;
  }
  let W = max(seedParams.seedWeight.x, 0.0);

  // Sample the seed at the destination pixel CENTRE in normalised UVs, so a
  // differently-sized seed texture is bilinearly resampled onto the accum grid.
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5)) / vec2f(f32(w), f32(h));
  let seedRgb = max(textureSampleLevel(seedTexture, seedSampler, uv, 0.0).xyz, vec3f(0.0));

  let pixelIndex = gid.y * w + gid.x;
  // Seed the colour accumulator: W virtual samples of value seedRgb.
  accumBuffer[pixelIndex] = vec4f(seedRgb * W, W);

  // Seed the variance moments consistently (W virtual samples of luminance
  // lum(seedRgb)): Σl = lum·W, Σl² = lum²·W, count = W. (A zero-variance prior
  // — every virtual sample equals the seed — which is correct: the prior carries
  // no per-sample spread of its own; real samples reintroduce variance.)
  let lum = luminance(seedRgb);
  varianceMomentsBuffer[pixelIndex] = vec4f(lum * W, lum * lum * W, W, 0.0);
}
`;
