/**
 * DDGI L2 spherical-harmonics irradiance — shared WGSL helpers.
 *
 * Seam-free replacement for the octahedral irradiance atlas (the octahedral
 * map's square edges put every CARDINAL normal — flat floors/walls — on the
 * fold seam, giving ~33% under-read vs ground truth; diagonals, which land in
 * the interior, were ~6%). L2 SH reconstructs DIFFUSE irradiance to ~1-2% at
 * EVERY normal including cardinals (Ramamoorthi & Hanrahan 2001, "An Efficient
 * Representation for Irradiance Environment Maps" — the cosine kernel
 * annihilates all bands above l=2, so 9 coefficients are near-exact for
 * irradiance), is seam-free (continuous, no border ring), and uses the 192-ray
 * budget far more efficiently (9 coeffs projected from 192 samples is heavily
 * over-determined). Validated CPU-side (wsl-gpu ddgi-sh-irradiance-validate.ts:
 * cardinals 1.4%, diagonals 1.6%) and GPU-side via ddgi-indirect-pi-ab.ts.
 *
 * VISIBILITY stays octahedral — depth is a step function (sharp at occluder
 * silhouettes); a low-order basis rings on it. Only IRRADIANCE migrates to SH.
 *
 * STORAGE (in-texture, zero bind-group churn): the 9 RGB coefficients are
 * written into the first 3x3 interior texels of each probe's existing
 * irradiance-atlas cell (coeff k at interior texel (k%3, k/3)). The atlas
 * stays a texture_2d<f32>; only the cell CONTENTS change (SH coeffs, not
 * octahedral texels). The octahedral border-fill pass is a no-op for SH
 * (continuous; nothing reads the border ring) so it is skipped for irradiance.
 *
 * CONVENTION: the COSINE-CONVOLVED irradiance coefficients E_lm = A_l * c_lm
 * are stored (A_l = {PI, 2PI/3, PI/4} per band), so the receiver eval is just
 * E(n) = sum_k E_lm[k] * Y_k(n) and returns irradiance E directly (matching the
 * old ddgiSample return; the material then applies albedo/PI exactly once).
 */

/** SH band convolution + basis for the probe producer. Receivers inline the
 *  same nine-term evaluation in the single-function `ddgiSample` parser ABI. */
export const DDGI_SH_WGSL = /* wgsl */`
// Real L2 SH basis (9 functions), matching the validated CPU reference.
fn ddgiShBasis(d: vec3f) -> array<f32, 9> {
  return array<f32, 9>(
    0.282095,
    0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
    1.092548 * d.x * d.y,
    1.092548 * d.y * d.z,
    0.315392 * (3.0 * d.z * d.z - 1.0),
    1.092548 * d.x * d.z,
    0.546274 * (d.x * d.x - d.y * d.y),
  );
}

// Cosine-lobe convolution coefficient per SH index (Ramamoorthi-Hanrahan):
// band l=0 -> PI, l=1 -> 2PI/3, l=2 -> PI/4. Applied at WRITE time.
fn ddgiShCosineA(k: u32) -> f32 {
  if (k == 0u) { return 3.14159265359; }
  if (k < 4u)  { return 2.09439510239; }   // 2*PI/3
  return 0.78539816340;                     // PI/4
}
`;
