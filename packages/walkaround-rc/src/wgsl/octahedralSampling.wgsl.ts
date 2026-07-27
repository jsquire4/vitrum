/**
 * Shared octahedral-direction sampling helpers for Radiance Cascades.
 *
 * The probe producer stratifies uniformly in octahedral UV, not uniformly in
 * solid angle. Receivers therefore have to use the same jittered UV sample
 * and multiply by the octahedral map Jacobian. Keeping both operations in
 * this fragment prevents the producer and receiver from silently drifting.
 *
 * References:
 *  - Cigolle et al. 2014, "A Survey of Efficient Representations for
 *    Independent Unit Vectors", JCGT Appendix A.2.
 *  - Veach 1997, Section 2.3, stratified Monte Carlo estimators.
 *
 * Include requirements: `octDecode` and `pcgHashToF32` must be defined by
 * the containing module before these functions are called.
 */

/** Solid-angle-only helpers. Safe to include in shaders that do not use PCG. */
export const RC_OCTAHEDRAL_SOLID_ANGLE_WGSL = /* wgsl */`
// Exact solid angle of a spherical triangle (Van Oosterom-Strackee form).
fn rcSphericalTriangleSolidAngle(a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let numerator = abs(dot(a, cross(b, c)));
  let denominator = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
  return 2.0 * atan2(numerator, denominator);
}

// Solid angle of one N x N octahedral cell. RC grids are powers of two with
// N >= 4, so their fold diagonals coincide with this two-triangle tessellation.
fn rcOctCellSolidAngle(cx: u32, cy: u32, gridSize: u32) -> f32 {
  let cellWidth = 2.0 / f32(gridSize);
  let u0 = -1.0 + f32(cx) * cellWidth;
  let v0 = -1.0 + f32(cy) * cellWidth;
  let u1 = u0 + cellWidth;
  let v1 = v0 + cellWidth;
  let p00 = octDecode(vec2f(u0, v0));
  let p10 = octDecode(vec2f(u1, v0));
  let p01 = octDecode(vec2f(u0, v1));
  let p11 = octDecode(vec2f(u1, v1));
  return rcSphericalTriangleSolidAngle(p00, p10, p01)
       + rcSphericalTriangleSolidAngle(p10, p11, p01);
}

// Jacobian d(omega)/d(u,v) for the normalized octahedral map. oct is in
// [-1,1]^2. Folding changes orientation but not the absolute Jacobian.
fn rcOctahedralSolidAngleDensity(oct: vec2f) -> f32 {
  var p = vec3f(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
  if (p.z < 0.0) {
    let oldX = p.x;
    p.x = (1.0 - abs(p.y)) * select(-1.0, 1.0, oldX >= 0.0);
    p.y = (1.0 - abs(oldX)) * select(-1.0, 1.0, p.y >= 0.0);
  }
  let len2 = dot(p, p);
  return 1.0 / max(len2 * sqrt(len2), 1e-20);
}

// Importance weight for one sample drawn uniformly inside an octahedral cell:
// cell UV area (4/N^2) multiplied by the map's local solid-angle Jacobian.
fn rcStratifiedSampleSolidAngle(rayUV: vec2f, gridSize: u32) -> f32 {
  let grid = f32(gridSize);
  return (4.0 / (grid * grid))
       * rcOctahedralSolidAngleDensity(rayUV * 2.0 - 1.0);
}
`;

/** Producer/receiver-shared stratification plus solid-angle helpers. */
export const RC_OCTAHEDRAL_STRATIFIED_SAMPLING_WGSL = /* wgsl */`
${RC_OCTAHEDRAL_SOLID_ANGLE_WGSL}

fn rcStratifiedRaySeed(
  probeIdx: u32,
  rayIdx: u32,
  frameSeed: u32,
) -> u32 {
  return (probeIdx * 0x9E3779B9u + rayIdx) ^ frameSeed;
}

fn rcStratifiedRayUV(
  probeIdx: u32,
  rayIdx: u32,
  gridSize: u32,
  frameSeed: u32,
) -> vec2f {
  let gx = f32(rayIdx % gridSize);
  let gy = f32(rayIdx / gridSize);
  let jitterSeed = rcStratifiedRaySeed(probeIdx, rayIdx, frameSeed);
  let jitter = vec2f(
    pcgHashToF32(jitterSeed),
    pcgHashToF32(jitterSeed * 7919u + 1u),
  );
  return (vec2f(gx, gy) + jitter) / f32(gridSize);
}
`;
