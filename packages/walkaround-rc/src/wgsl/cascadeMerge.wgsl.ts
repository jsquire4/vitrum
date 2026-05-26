/**
 * Cascade merge compute shader — assembled raw WGSL.
 *
 * Extracted from `_staging/legacy-source/src/rendering/shaders/walkaround/cascadeMerge.wgsl.ts`.
 * The original used Three.js TSL `wgslFn()` / `wgsl()` nodes.
 * This file assembles the equivalent single-module WGSL string.
 *
 * Composition order:
 *   1. MergeUniforms struct (20 floats = 80 bytes; matches buildMergeUniformData() layout)
 *   2. trilinearSampleUpper helper function
 *   3. Entry-point function with @compute @workgroup_size(64)
 *
 * The original TSL `instanceIndex` built-in becomes `@builtin(global_invocation_id)`.
 * Storage buffer pointer params become module-scope `var<storage, ...>` bindings.
 *
 * Merge pass bind group layout (one per merge step):
 *   binding 0: upperCascade  — read-only  — array<vec4f>
 *   binding 1: lowerCascade  — read_write — array<vec4f>
 *   binding 2: m_arr         — read-only  — array<MergeUniforms>
 *
 * See `src/rc/TSL_TO_RAW_MAPPING.md` for the full mapping rationale.
 */

import { OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

export const CASCADE_MERGE_WGSL = /* wgsl */`
${OCTAHEDRAL_CORE_WGSL}

// ─── octCellSolidAngle ───────────────────────────────────────────────────────
// Per-bin solid-angle estimate for a cell at grid position (cx, cy) in an
// N×N octahedral direction grid.
//
// The octahedral mapping is NOT solid-angle-uniform: cells near the fold
// edges subtend a smaller solid angle than central cells.  This helper
// computes the solid angle numerically by decoding the four corners of the
// cell into unit directions and summing the two-triangle spherical quad area.
//
// The merge kernel uses this to weight children by their actual solid-angle
// coverage instead of the Sannikov-paper ÷4 assumption, which is only valid
// when each parent covers exactly 4 children.  With the non-Sannikov probe
// scaling in CASCADE_DIMS, the ÷4 assumption is incorrect.
//
// Reference: Cigolle et al. 2014, "A Survey of Efficient Representations for
// Independent Unit Vectors", JCGT §A.2 — octahedral Jacobian / texel area.
// Reference: Sannikov 2023, §3 — cascade conservation law.

// Spherical quad area via two-triangle cross-product approximation.
fn sphericalQuadAreaForMerge(p00: vec3f, p10: vec3f, p01: vec3f, p11: vec3f) -> f32 {
  let d1 = cross(p10 - p00, p01 - p00);
  let d2 = cross(p10 - p11, p01 - p11);
  return (length(d1) + length(d2)) * 0.5;
}

// Solid angle of cell (cx, cy) in an N×N octahedral grid.
// cx, cy are 0-based column/row indices.  N = gridSize (e.g. 4, 8, 16, 32).
fn octCellSolidAngle(cx: u32, cy: u32, N: u32) -> f32 {
  let cellWidth = 2.0 / f32(N);
  let u0 = -1.0 + f32(cx) * cellWidth;
  let v0 = -1.0 + f32(cy) * cellWidth;
  let u1 = u0 + cellWidth;
  let v1 = v0 + cellWidth;
  let p00 = octDecode(vec2f(u0, v0));
  let p10 = octDecode(vec2f(u1, v0));
  let p01 = octDecode(vec2f(u0, v1));
  let p11 = octDecode(vec2f(u1, v1));
  return sphericalQuadAreaForMerge(p00, p10, p01, p11);
}

// ─── MergeUniforms struct ─────────────────────────────────────────────────────
// Must match buildMergeUniformData() layout in cascadeDispatch.ts
// (20 floats = 80 bytes).

struct MergeUniforms {
  lowerProbeCount  : vec3u,
  lowerRayCount    : u32,
  upperProbeCount  : vec3u,
  upperRayCount    : u32,
  lowerRayGridSize : u32,
  upperRayGridSize : u32,
  _pad0            : vec2u,
  probeOriginWorld : vec3f,
  _pad1            : f32,
  roomSize         : vec3f,
  _pad2            : f32,
};

// ─── Bind group declarations ──────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       rc_upperCascade: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> rc_lowerCascade: array<vec4f>;
@group(0) @binding(2) var<storage, read>       rc_m_arr:        array<MergeUniforms>;

// ─── trilinearSampleUpper ────────────────────────────────────────────────────
// Verbatim from trilinearSampleUpper wgslFn body in cascadeMerge.wgsl.ts.
// Module-scope upper cascade accessed directly instead of via ptr parameter.

fn trilinearSampleUpper(
  worldPos : vec3f,
  rayIdx   : u32,
  m        : MergeUniforms,
) -> vec3f {
  let probeUV = (worldPos - m.probeOriginWorld) / m.roomSize;
  let gridF   = probeUV * vec3f(m.upperProbeCount) - vec3f(0.5);
  let gridI   = vec3i(floor(gridF));
  let f       = gridF - floor(gridF);

  let maxCorner = vec3i(m.upperProbeCount) - vec3i(1);

  var sum = vec3f(0.0);
  for (var dz = 0; dz < 2; dz = dz + 1) {
    for (var dy = 0; dy < 2; dy = dy + 1) {
      for (var dx = 0; dx < 2; dx = dx + 1) {
        let corner = clamp(gridI + vec3i(dx, dy, dz), vec3i(0), maxCorner);
        let cornerProbeIdx = u32(corner.x)
                           + u32(corner.y) * m.upperProbeCount.x
                           + u32(corner.z) * m.upperProbeCount.x * m.upperProbeCount.y;
        let upperOutIdx = cornerProbeIdx * m.upperRayCount + rayIdx;

        if (upperOutIdx >= arrayLength(&rc_upperCascade)) {
          continue;
        }

        let rad = rc_upperCascade[upperOutIdx].rgb;
        let wx  = select(1.0 - f.x, f.x, dx != 0);
        let wy  = select(1.0 - f.y, f.y, dy != 0);
        let wz  = select(1.0 - f.z, f.z, dz != 0);
        sum = sum + rad * wx * wy * wz;
      }
    }
  }
  return sum;
}

// ─── Entry point ─────────────────────────────────────────────────────────────
// Verbatim from cascadeMergeKernel wgslFn body.
// TSL instanceIndex → @builtin(global_invocation_id) globalId, index = globalId.x.

@compute @workgroup_size(64)
fn cascadeMergeKernel(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let uMerge = rc_m_arr[0];
  let totalLower   = uMerge.lowerProbeCount.x * uMerge.lowerProbeCount.y * uMerge.lowerProbeCount.z;
  let totalThreads = totalLower * uMerge.lowerRayCount;
  if (index >= totalThreads) { return; }

  let lowerProbeIdx = index / uMerge.lowerRayCount;
  let lowerRayIdx   = index % uMerge.lowerRayCount;

  let lowerOutIdx = lowerProbeIdx * uMerge.lowerRayCount + lowerRayIdx;
  let local       = rc_lowerCascade[lowerOutIdx];

  if (local.a > 0.5) { return; }

  let pz = lowerProbeIdx / (uMerge.lowerProbeCount.x * uMerge.lowerProbeCount.y);
  let py = (lowerProbeIdx / uMerge.lowerProbeCount.x) % uMerge.lowerProbeCount.y;
  let px = lowerProbeIdx % uMerge.lowerProbeCount.x;
  let lowerUV  = (vec3f(f32(px), f32(py), f32(pz)) + 0.5) / vec3f(uMerge.lowerProbeCount);
  let probePos = uMerge.probeOriginWorld + lowerUV * uMerge.roomSize;

  let gx = f32(lowerRayIdx % uMerge.lowerRayGridSize);
  let gy = f32(lowerRayIdx / uMerge.lowerRayGridSize);

  // Solid-angle-weighted merge (Path A — non-Sannikov cascade dimensions).
  //
  // The standard Sannikov merge averages 4 children with equal weight (÷4),
  // which conserves energy only when each parent bin covers exactly 4×
  // the solid angle of each child.  The CASCADE_DIMS in cascadePyramid.ts
  // use non-paper scaling (~2.7–7.2× probe-count ratio, not the Sannikov /8),
  // so the ÷4 assumption is violated and energy leaks across cascade levels.
  //
  // Fix: weight each child by its octahedral solid angle and normalize by the
  // total child solid angle so the merge is a proper weighted average:
  //   merged = Σ child_i · Ω_i / Σ Ω_i
  //
  // For true Sannikov 2D/3D dimensions this simplifies to the original ÷4
  // since all four children are adjacent cells of equal solid angle.
  //
  // References:
  //   Sannikov 2023, §3 — cascade conservation law (violated by current dims).
  //   Cigolle et al. 2014, JCGT §A.2 — octahedral solid-angle per texel.
  var merged     = vec3f(0.0);
  var omegaTotal = 0.0;
  for (var ci = 0u; ci < 4u; ci = ci + 1u) {
    let dx = ci % 2u;
    let dy = ci / 2u;
    let childGx = u32(gx) * 2u + dx;
    let childGy = u32(gy) * 2u + dy;
    let childRayIdx = childGx + childGy * uMerge.upperRayGridSize;

    let childRad   = trilinearSampleUpper(probePos, childRayIdx, uMerge);
    let childOmega = octCellSolidAngle(childGx, childGy, uMerge.upperRayGridSize);

    merged     = merged + childRad * childOmega;
    omegaTotal = omegaTotal + childOmega;
  }
  // Normalize by total child solid angle; guard against degenerate zero.
  merged = merged / max(omegaTotal, 1e-6);

  rc_lowerCascade[lowerOutIdx] = vec4f(local.rgb + merged, 1.0);
}
`;
