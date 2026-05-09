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

export const CASCADE_MERGE_WGSL = /* wgsl */`

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
fn cascadeMergeKernel(@builtin(global_invocation_id) globalId: vec3u) -> void {
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

  var merged = vec3f(0.0);
  for (var ci = 0u; ci < 4u; ci = ci + 1u) {
    let dx = ci % 2u;
    let dy = ci / 2u;
    let childGx = u32(gx) * 2u + dx;
    let childGy = u32(gy) * 2u + dy;
    let childRayIdx = childGx + childGy * uMerge.upperRayGridSize;

    merged = merged + trilinearSampleUpper(probePos, childRayIdx, uMerge);
  }
  merged = merged * 0.25;

  rc_lowerCascade[lowerOutIdx] = vec4f(local.rgb + merged, 1.0);
}
`;
