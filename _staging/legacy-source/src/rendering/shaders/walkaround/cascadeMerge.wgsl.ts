/**
 * Cascade merge compute kernel (§7).
 *
 * Bottom-up merge: for each probe in the lower cascade, for each ray direction,
 * if the ray escaped (alpha == 0), sample the corresponding directions in the
 * upper cascade via trilinear interpolation over probes, and add them in.
 *
 * After merge, lower cascade's rays contain:
 *   localRadiance (if hit) OR upper-cascade-contribution (if escaped).
 *
 * §7.1: each lower ray maps to 4 upper rays (2×2 block in oct grid since
 * upper cascade has 2× denser angular grid per axis → 4× more rays).
 * Average the 4 upper contributions before adding.
 *
 * Structure note: wgslFn() expects the code string to start with `fn`.
 * MergeUniforms struct is exported as wgsl() and passed as an include.
 */

import { wgslFn, wgsl } from 'three/tsl';

/** MergeUniforms struct (must match layout in cascadeDispatch.ts). */
export const mergeUniformsStruct = wgsl(/* wgsl */`
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
`);

/** Trilinear sample of upper cascade at a world position for a given ray index.
 *  Takes a pointer to MergeUniforms array to match the storage-buffer approach. */
const trilinearSampleUpper = wgslFn(
  /* wgsl */`
  fn trilinearSampleUpper(
    upper    : ptr<storage, array<vec4f>, read>,
    worldPos : vec3f,
    rayIdx   : u32,
    m        : MergeUniforms,
  ) -> vec3f {
    // Map world pos to upper probe grid float coords.
    let probeUV = (worldPos - m.probeOriginWorld) / m.roomSize;
    let gridF   = probeUV * vec3f(m.upperProbeCount) - vec3f(0.5);
    let gridI   = vec3i(floor(gridF));
    let f       = gridF - floor(gridF);

    // Clamp to valid range.
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

          // Guard: rayIdx might be out of range if upper has fewer rays at boundary.
          if (upperOutIdx >= arrayLength(upper)) {
            continue;
          }

          let rad = (*upper)[upperOutIdx].rgb;
          let wx  = select(1.0 - f.x, f.x, dx != 0);
          let wy  = select(1.0 - f.y, f.y, dy != 0);
          let wz  = select(1.0 - f.z, f.z, dz != 0);
          sum = sum + rad * wx * wy * wz;
        }
      }
    }
    return sum;
  }
  `,
  [mergeUniformsStruct],
);

/** The cascade merge compute kernel. */
export const cascadeMergeKernel = wgslFn(
  /* wgsl */`
  fn cascadeMergeKernel(
    upperCascade : ptr<storage, array<vec4f>, read>,
    lowerCascade : ptr<storage, array<vec4f>, read_write>,
    m_arr        : ptr<storage, array<MergeUniforms>, read>,
    index        : u32,
  ) -> void {
    let uMerge = (*m_arr)[0];
    let totalLower   = uMerge.lowerProbeCount.x * uMerge.lowerProbeCount.y * uMerge.lowerProbeCount.z;
    let totalThreads = totalLower * uMerge.lowerRayCount;
    if (index >= totalThreads) { return; }

    let lowerProbeIdx = index / uMerge.lowerRayCount;
    let lowerRayIdx   = index % uMerge.lowerRayCount;

    let lowerOutIdx = lowerProbeIdx * uMerge.lowerRayCount + lowerRayIdx;
    let local       = (*lowerCascade)[lowerOutIdx];

    // Only merge if THIS cascade's ray escaped (alpha == 0).
    // If alpha == 1, this probe already has a local hit contribution.
    if (local.a > 0.5) { return; }

    // Lower probe world position.
    let pz = lowerProbeIdx / (uMerge.lowerProbeCount.x * uMerge.lowerProbeCount.y);
    let py = (lowerProbeIdx / uMerge.lowerProbeCount.x) % uMerge.lowerProbeCount.y;
    let px = lowerProbeIdx % uMerge.lowerProbeCount.x;
    let lowerUV  = (vec3f(f32(px), f32(py), f32(pz)) + 0.5) / vec3f(uMerge.lowerProbeCount);
    let probePos = uMerge.probeOriginWorld + lowerUV * uMerge.roomSize;

    // The 4 upper-cascade child direction indices for this lower direction.
    // Upper grid is 2x denser: lower grid cell (gx, gy) maps to 4 upper cells.
    let gx = f32(lowerRayIdx % uMerge.lowerRayGridSize);
    let gy = f32(lowerRayIdx / uMerge.lowerRayGridSize);

    var merged = vec3f(0.0);
    for (var ci = 0u; ci < 4u; ci = ci + 1u) {
      let dx = ci % 2u;
      let dy = ci / 2u;
      let childGx = u32(gx) * 2u + dx;
      let childGy = u32(gy) * 2u + dy;
      let childRayIdx = childGx + childGy * uMerge.upperRayGridSize;

      merged = merged + trilinearSampleUpper(
        upperCascade, probePos, childRayIdx, uMerge
      );
    }
    merged = merged * 0.25;

    // Sum: local was (0,0,0,0) for escaped rays; add merged contribution.
    (*lowerCascade)[lowerOutIdx] = vec4f(local.rgb + merged, 1.0);
  }
  `,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [mergeUniformsStruct, trilinearSampleUpper as any],
);
