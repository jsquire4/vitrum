/**
 * spatialFilter.wgsl.ts — 37-tap hexagonal-kernel edge-stopping spatial filter.
 *
 * Exported as a WGSL string constant + a TypeScript descriptor of the bind
 * group layout.  Consumed by the host's PT preview post-process pipeline as
 * a compute pass inserted first in the EffectComposer chain (before Bloom).
 *
 * The filter auto-disables at the host level when `pathtracer.samples > 24`
 * (temporal accumulation has converged and the filter would over-blur).
 * The WGSL kernel itself is always applied when dispatched; the enable gate
 * lives in the host's orchestration code, not in the shader.
 *
 * Algorithm:
 *   Bilateral edge-stopping filter with a 37-tap hexagonal kernel.
 *   Hexagonal kernels cover more area per tap than square kernels of the same
 *   radius, giving better low-frequency noise reduction without the cross-axis
 *   bias artefacts of the 5×5 À-trous approximation.
 *
 *   Weight: w(x,y) = exp(-(|c_x - c_y|² / σ_c² + |n_x - n_y|² / σ_n² + (z_x - z_y)² / σ_z²))
 *
 *   where c = noisy color (HDR linear), n = world-space normal, z = linear depth.
 *
 * Kernel layout (hexagonal grid, screen-aligned):
 *   37 taps = 1 center + 6 inner ring (radius 1) + 12 middle ring (radius 2) +
 *             18 outer ring (radius 3) mapped onto a square pixel grid via
 *             flat-top hexagonal coordinates.
 *
 *   The hex-to-screen offsets below are computed for a flat-top hex grid with
 *   unit radius and quantized to integer pixel offsets.  Each ring samples
 *   one full revolution of the hexagon at that radius.
 *
 * References:
 *   Dammertz, Hanika, Keller "Edge-Avoiding Á-Trous Wavelet Transform for fast
 *   Global Illumination Filtering" HPG 2010.
 *
 *   McGuire et al. "A Fast and Stable Feature-Aware Motion Blur Filter" JCGT 2012
 *   (hexagonal kernel shape motivation).
 *
 *   Sprint 6 spec: plan/archive/phase-6-roadmap.md §Sprint 6.
 *
 * Bind group layout (group 0):
 *
 *   binding 0 — texture_2d<f32>              inputColor       (noisy RGBA16F color)
 *   binding 1 — texture_storage_2d<rgba16float, write>  outputColor   (filtered)
 *   binding 2 — texture_2d<f32>              gbufferNormal    (RGBA16F, .xyz = world normal)
 *   binding 3 — texture_2d<f32>              gbufferDepth     (RGBA16F, .w = linear depth)
 *   binding 4 — var<uniform> SpatialFilterUBO
 *
 * SpatialFilterUBO fields:
 *   sigmaColor: f32  — color/radiance edge-stop σ (HDR-aware; default ~0.1)
 *   sigmaNormal: f32 — normal edge-stop σ (default ~32.0, applied as exponent)
 *   sigmaDepth: f32  — depth edge-stop σ (world units; default ~0.01)
 *   _pad: f32        — alignment
 */

import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';

export const SPATIAL_FILTER_WGSL = /* wgsl */ `
${LUMINANCE_WGSL}

@group(0) @binding(0) var inputColor:    texture_2d<f32>;
@group(0) @binding(1) var outputColor:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var gbufferNormal: texture_2d<f32>;
@group(0) @binding(3) var gbufferDepth:  texture_2d<f32>;

struct SpatialFilterUBO {
  sigmaColor:  f32,
  sigmaNormal: f32,
  sigmaDepth:  f32,
  _pad:        f32,
};
@group(0) @binding(4) var<uniform> ubo: SpatialFilterUBO;

// ─── Hexagonal kernel offsets ────────────────────────────────────────────────
//
// 37 taps on a flat-top hexagonal grid, quantized to integer pixel coordinates.
//
// Ring 0: 1 tap  — center
// Ring 1: 6 taps — inner ring, radius ≈ 1 px
// Ring 2: 12 taps — middle ring, radius ≈ 2 px
// Ring 3: 18 taps — outer ring, radius ≈ 3 px
//
// Flat-top hex coordinates: for hex (q, r) the pixel offset is
//   x = size * (3/2 * q)
//   y = size * (sqrt(3)/2 * q + sqrt(3) * r)
// with size = 1 and quantized to nearest integer.
//
// The offsets below are listed as (dx, dy) pairs in row-major order.
// Ring membership is annotated for reference.

const NUM_TAPS: i32 = 37;

// Offsets stored as flat array: tap i → (OFFSETS[2*i], OFFSETS[2*i+1])
const OFFSETS: array<i32, 74> = array<i32, 74>(
  // Ring 0 — center (1 tap)
  0,  0,
  // Ring 1 — inner (6 taps, radius 1)
  1,  0,
 -1,  0,
  0,  1,
  0, -1,
  1, -1,
 -1,  1,
  // Ring 2 — middle (12 taps, radius ~2)
  2,  0,
 -2,  0,
  0,  2,
  0, -2,
  2, -1,
  2, -2,
  1,  1,
 -1, -1,
 -2,  1,
 -2,  2,
 -1,  2,
  1, -2,
  // Ring 3 — outer (18 taps, radius ~3)
  3,  0,
 -3,  0,
  0,  3,
  0, -3,
  3, -1,
  3, -2,
  3, -3,
  2,  1,
  1,  2,
 -1, -2,
 -2, -1,
 -3,  1,
 -3,  2,
 -3,  3,
 -2,  3,
 -1,  3,
  1, -3,
  2, -3,
);

// ─── Entry point ─────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16, 1)
fn spatialFilterMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(inputColor);
  if (any(gid.xy >= dims)) { return; }

  let cCenter = textureLoad(inputColor,    gid.xy, 0).rgb;
  // Normal is stored as (n*0.5+0.5) by the shade pass — decode to [-1,1].
  // Must match the encoding used by atrous.wgsl.ts.
  let nCenter = textureLoad(gbufferNormal, gid.xy, 0).xyz * 2.0 - 1.0;
  let zCenter = textureLoad(gbufferDepth,  gid.xy, 0).w;

  // Sky / miss pixels (depth == 0) pass through unfiltered.
  if (zCenter <= 0.0) {
    textureStore(outputColor, gid.xy, vec4f(cCenter, 1.0));
    return;
  }

  var sumColor  = vec3f(0.0);
  var sumWeight = 0.0;

  let sigC2 = ubo.sigmaColor  * ubo.sigmaColor + 1e-6;
  let sigZ2 = ubo.sigmaDepth  * ubo.sigmaDepth + 1e-6;
  // sigmaNormal used as an exponent on the clamped dot product (Dammertz 2010).
  let sigN  = max(1.0, ubo.sigmaNormal);

  for (var i: i32 = 0; i < NUM_TAPS; i++) {
    let dx = OFFSETS[2 * i];
    let dy = OFFSETS[2 * i + 1];
    let p  = vec2i(gid.xy) + vec2i(dx, dy);
    if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) { continue; }
    let pu = vec2u(p);

    let cP = textureLoad(inputColor,    pu, 0).rgb;
    let nP = textureLoad(gbufferNormal, pu, 0).xyz * 2.0 - 1.0;
    let zP = textureLoad(gbufferDepth,  pu, 0).w;

    // ── Edge-stopping weights ──────────────────────────────────────────────

    // Color: luminance-normalized chromaticity distance (matches à-trous
    // convention in atrous.wgsl.ts; avoids HDR magnitude bias).
    // luminance() is the canonical Rec.709 helper from LUMINANCE_WGSL
    // (prepended above at module build time).
    let lumP = max(1e-3, luminance(cP));
    let lumC = max(1e-3, luminance(cCenter));
    let dc   = length(cP / lumP - cCenter / lumC);
    let wc   = exp(-dc * dc / sigC2);

    // Normal: power of clamped dot product (Dammertz 2010 §3.2).
    let dn = max(0.0, dot(nCenter, nP));
    let wn = pow(dn, sigN);

    // Depth: Gaussian falloff on absolute depth difference.
    let wz = exp(-(zP - zCenter) * (zP - zCenter) / sigZ2);

    let w = wc * wn * wz;
    sumColor  += cP * w;
    sumWeight += w;
  }

  let result = select(cCenter, sumColor / sumWeight, sumWeight > 1e-6);
  textureStore(outputColor, gid.xy, vec4f(result, 1.0));
}
`;

// ────────────────────────────────────────────────────────────────────────────
// TypeScript descriptor (informational — no GPU objects created here)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Documents the bind group layout expected by `SPATIAL_FILTER_WGSL`.
 * Host code MUST bind these in the same order and with matching formats.
 *
 * File-local — this is a documentation-only descriptor with no compile-
 * time consumers (TypeScript or runtime). Kept here as inline schema for
 * humans reading the WGSL string above. 2026-05-18 dead-code sweep
 * verified zero non-self consumers.
 */
interface SpatialFilterBindGroupLayout {
  /** binding 0 — noisy input color.  Format: rgba16float (RGBA16F).
   *  Source: the PT accumulation buffer (FrameOutput.primaryRadiance). */
  inputColor: 'texture_2d<f32>';

  /** binding 1 — filtered output color.  Format: rgba16float (RGBA16F).
   *  Must be a storage texture (read_write or write). */
  outputColor: 'texture_storage_2d<rgba16float, write>';

  /** binding 2 — G-buffer world-space normal.  Format: rgba16float.
   *  Source: FrameOutput.normalDepth, channels .xyz.
   *  Encoding: (normal * 0.5 + 0.5), authored by the shade pass.
   *  Consumers decode with (xyz * 2.0 - 1.0) — see atrous.wgsl.ts and
   *  spatialFilter.wgsl.ts for the canonical decode. */
  gbufferNormal: 'texture_2d<f32>';

  /** binding 3 — G-buffer linear depth.  Format: rgba16float.
   *  Source: FrameOutput.normalDepth, channel .w.
   *  Linear camera-space depth, always positive for scene hits, 0 for sky. */
  gbufferDepth: 'texture_2d<f32>';

  /** binding 4 — SpatialFilterUBO.
   *  sigmaColor: f32  — default 0.1  (HDR chromaticity edge stop)
   *  sigmaNormal: f32 — default 32.0 (dot-product exponent for normal edge stop)
   *  sigmaDepth: f32  — default 0.01 (world-units depth edge stop)
   *  _pad: f32        — alignment padding */
  ubo: 'uniform SpatialFilterUBO';
}
