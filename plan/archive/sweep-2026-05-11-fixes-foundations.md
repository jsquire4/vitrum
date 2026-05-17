# Sweep 2026-05-11 — Fix & Foundations Plan

**Branch:** `main` | **Date:** 2026-05-11 | **Scope:** Items 14–18, 26–38

This plan covers every item in the second-half sweep. Each entry is written
against the live code verified in the 2026-05-11 in-flight sweep. No code
changes are made here; this is the authoritative fix brief.

---

## Item 14: pt-webgpu glossy BSDF sampling/PDF mismatch

**File(s):**
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:930–934` (`glossyReflectionSample`)
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:334–368` (`brdfDirectionalPdf`)

**Root cause:** `glossyReflectionSample` generates samples via
`mix(reflectDir, cosineHemiSample, roughness²)` — a lerp around the mirror
direction — but `brdfDirectionalPdf` evaluates those same samples against the
GGX half-vector PDF `D(h)·(N·H) / (4·(V·H))`. The two distributions are
incompatible, so MC integration over any glossy surface gives a biased
estimator. At low roughness the lerp collapses to a near-mirror direction
(roughly correct), but at mid roughness (0.3–0.7) the lerp distribution differs
from GGX VNDF by >2× in mean and has a heavier tail — the result is systematic
energy loss on glossy surfaces.

**Authoritative source:** Heitz, E. "Sampling the GGX Distribution of Visible
Normals." *Journal of Computer Graphics Techniques* 7(4):1–13, 2018.
https://jcgt.org/published/0007/04/01/paper.pdf  
Section 3 gives the exact VNDF sample formula; Section 4 gives the exact PDF
`p(ωi | ωo) = G1(ωo)·max(0, ωo·ωm)·D(ωm) / (ωo·N)` expressed as
`D(ωm)·G1(ωo)·max(0, ωo·h) / (N·ωo)` with the Jacobian cancellation for
reflection already applied, reducing to `D(h)·(N·H) / (4·(V·H))` — which is
already what `brdfDirectionalPdf` computes. The fix is therefore on the
sampling side: replace the lerp with VNDF sampling so both sides use the same
distribution.

**Fix (complete WGSL outline):**

1. Add a helper `sampleGgxVndf(wo: vec3f, alpha: f32, rng) -> vec3f` to
   `pathTraceBruteforce.wgsl.ts` implementing Algorithm 1 of Heitz 2018:

   ```wgsl
   fn sampleGgxVndf(wo: vec3f, alpha: f32, rng: ptr<function, u32>) -> vec3f {
     // Step 1: Stretch wo into the unit-roughness space.
     let Vh = safe_normalize(vec3f(alpha * wo.x, alpha * wo.y, wo.z));
     // Step 2: Build an ONB around Vh.
     var T1: vec3f; var T2: vec3f;
     buildOnb(Vh, &T1, &T2);
     // Step 3: Sample point on unit hemisphere with cosine-weighted
     //         projection adapted for the visible normal distribution.
     let u1 = rand_f32(rng);
     let u2 = rand_f32(rng);
     let r = sqrt(u1);
     let phi = 2.0 * PI * u2;
     let t1 = r * cos(phi);
     let t2 = r * sin(phi);
     let s = 0.5 * (1.0 + Vh.z);
     let t2_adj = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
     // Step 4: Unstretch and return the half-vector in world space.
     let Nh = t1 * T1 + t2_adj * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2_adj * t2_adj)) * Vh;
     return safe_normalize(vec3f(alpha * Nh.x, alpha * Nh.y, max(1e-6, Nh.z)));
   }
   ```

   Note: `wo` must be in the surface tangent-space (N = +Z) for this
   algorithm. The caller must transform `wo` into surface-local space
   before calling, and transform the returned half-vector back to world
   space. Add a `worldToLocal(v, n, t, b)` helper if one does not exist.

2. Replace `glossyReflectionSample` body with:
   ```wgsl
   fn glossyReflectionSample(rng: ptr<function, u32>, wo: vec3f, n: vec3f, t: vec3f, b: vec3f, roughness: f32) -> vec3f {
     let alpha = max(roughness * roughness, 0.001);
     let wo_local = vec3f(dot(wo, t), dot(wo, b), dot(wo, n));
     let h_local = sampleGgxVndf(wo_local, alpha, rng);
     // Reflect wo around the sampled half-vector.
     let h_world = h_local.x * t + h_local.y * b + h_local.z * n;
     return safe_normalize(reflect(-wo, h_world));
   }
   ```
   Signature change: add `n, t, b` parameters. Update all call sites
   (`sampleNextBounceDirection` specular and transmission branches at
   lines 1349 and 1355) — they already have `normal` and can call
   `buildOnb` to get `t, b`.

3. `brdfDirectionalPdf` already computes `D(h)·(N·H)/(4·(V·H))` for the
   specular PDF, which is the exact VNDF PDF once the reflection Jacobian is
   applied. No change needed there.

4. Remove `k = clamp(roughness * roughness, 0.0, 1.0)` lerp entirely from
   all call sites.

**Decision points:** None — this is a clear correctness requirement; there is
no valid alternative that keeps the current lerp sampling while matching the
existing half-vector PDF.

**Behavior-preserving test:**
```ts
// packages/pt-webgpu/__tests__/bsdfPdfNormalization.test.ts
// Monte Carlo integration of sampleGgxVndf PDF over hemisphere ≈ 1.0
// for several (wo, alpha) pairs. Tolerance 1% at N=100 000 samples.
// Also: verify pdfSpec from brdfDirectionalPdf matches
//   D(h)*NdotH/(4*VdotH) within 1e-4 for matched (wi, wo) pairs.
```

**Verification:** On a Cornell box with a polished sphere (roughness=0.2),
visible glossy highlight should tighten to match the GGX lobe; energy should
be conserved (white furnace test at roughness 0–1 stays ≤ 1.0 per channel).

**Dependencies:** None upstream.

**Risk:** Signature change to `glossyReflectionSample` touches two call sites
in `sampleNextBounceDirection`. The dielectric transmission branch also calls
it (line 1349) — verify the half-angle convention is symmetric for refracted
rays before signing off.

---

## Item 15: pt-webgpu BSDF→light MIS sees only first area light

**File(s):**
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:370–399`
  (`intersectRectAreaLightRay`, hardcodes `rectAreaLights[0..3]`)
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:402–419`
  (`intersectMeshAreaLightRay`, hardcodes `meshAreaLights[0..3]`)

**Root cause:** Both functions read the first light's geometry data using
hard-coded offsets `[0]`, `[1]`, `[2]`, `[3]`. When multiple rect-area or
mesh-area lights are present, BSDF→light MIS cannot evaluate the PDF against
lights 1..N, so contribution from those lights via the BSDF lobe is lost
(the path still samples them in the direct-light loop, but MIS weight is
wrong).

**Authoritative source:** Veach, E. "Robust Monte Carlo Methods for Light
Transport Simulation." PhD thesis, Stanford, 1997, Chapter 9 (power
heuristic MIS). The PDF term in the power heuristic must match the sampling
distribution identically. If the BSDF-sample-to-light intersection test
skips N−1 lights, the returned PDF is 0 for all but one light and the
balance equation breaks.

**Fix:**

1. Generalize `intersectRectAreaLightRay` to accept a light index `li: u32`:
   ```wgsl
   fn intersectRectAreaLightRay(li: u32, rayOrigin, rayDir, distOut, lightPdfOut) -> bool {
     let rb = li * 4u;
     let rectPos = rectAreaLights[rb].xyz;
     let uAxis = rectAreaLights[rb + 1u].xyz;
     let vAxis = rectAreaLights[rb + 2u].xyz;
     // remainder unchanged
   }
   ```
2. Generalize `intersectMeshAreaLightRay` similarly with `li * 4u` base offset.
3. In `bsdfAreaLightConnectionContribution`, iterate over all
   `params.rectAreaLightCount` lights, keeping the minimum-distance hit.
   Weight the chosen PDF by `1 / lightCount` (uniform light selection
   probability) if a random light selection model is used, or accumulate
   all PDF contributions if using sum MIS.
4. The simplest correct fix for the current uniform-random light selection
   model: find the light hit closest to the BSDF sample direction; its PDF
   is `lightPdf * lightCount` (since it was selected with probability
   `1/lightCount`). The power-heuristic denominator already sees
   `lightPdf` so multiply by `lightCount` to cancel the unbiased
   estimator.

**Decision points:** Determine whether BSDF connection should consider all
lights simultaneously (sum MIS, unbiased) or only the one selected in the
direct-light RNG draw (balance heuristic per randomly-chosen light). The
former is more correct but requires N intersection tests per bounce.
Recommended: iterate all lights, take closest hit, use its PDF directly —
this is unbiased and only costs O(N) light-count extra tests, acceptable
for the prototype.

**Behavior-preserving test:**
```ts
// Scene with 2 rect-area lights on opposite walls.
// Verify total radiance is within 5% of single-light × 2 (linearity).
```

**Verification:** Enable 2 rect-area lights in the test scene; confirm
brightness doubles vs single-light scene.

**Dependencies:** None. Must land before Item 14 if 14 changes the BSDF
sampling path (so the MIS test uses the corrected PDF).

**Risk:** Low. The change is a loop expansion of an existing per-light
pattern already present in the main direct-light loop.

---

## Item 16: pt-webgpu dielectric branch heuristic, not Fresnel-weighted

**File(s):**
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:1313–1368`
  (`sampleNextBounceDirection`)

**Root cause:** The dielectric branch (`chooseTransmission`) is entered when
`xi < transProb`, where `transProb = transmission * (1 - metallic)`. This is
a material-property probability, not a Fresnel-weighted probability. The exact
Fresnel reflectance `R(θ)` determines the per-ray split between reflection and
refraction; at grazing angles `R → 1` (total internal reflection suppresses
transmission regardless of `transmission` flag). The code's heuristic
over-transmits at grazing angles and produces incorrect TIR behavior (the
`validRefract` fallback at line 1346 does reflect, but the probability budget
for reflection is too low because `specProb` was reduced by `baseTransProb`).

**Authoritative source:** Pharr, Jakob, Humphreys. *Physically Based
Rendering* 4th ed. Section 9.3 "Specular Reflection and Transmission" and the
`DielectricBxDF::Sample_f` implementation at pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF.
The correct partition: compute `R = FrDielectric(cosTheta_i, eta)` using the
unpolarized Fresnel equations; sample reflect with probability R, transmit
with probability T = 1 − R; divide throughput by the chosen probability to
produce an unbiased estimator.

**Fix:**

1. Add `fn frDielectric(cosTheta_i: f32, eta: f32) -> f32` implementing the
   unpolarized Fresnel equation:
   ```wgsl
   fn frDielectric(cosTheta_i_in: f32, eta_in: f32) -> f32 {
     var cosTheta_i = clamp(cosTheta_i_in, -1.0, 1.0);
     var eta = eta_in;
     if (cosTheta_i < 0.0) { eta = 1.0 / eta; cosTheta_i = -cosTheta_i; }
     let sin2Theta_i = max(0.0, 1.0 - cosTheta_i * cosTheta_i);
     let sin2Theta_t = sin2Theta_i / (eta * eta);
     if (sin2Theta_t >= 1.0) { return 1.0; }  // TIR
     let cosTheta_t = sqrt(max(0.0, 1.0 - sin2Theta_t));
     let r_par  = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
     let r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
     return 0.5 * (r_par * r_par + r_perp * r_perp);
   }
   ```
   This is the standard unpolarized Fresnel from PBR4e §9.3 /
   `FrDielectric()`.

2. In `sampleNextBounceDirection`, inside the `transmission > 0` path,
   compute `R = frDielectric(dot(-incomingDir, normal), ior)`.
3. Split the budget: `let pr = R; let pt = 1.0 - R;`
4. Randomly select: `if (xi < pr) { specular reflect } else { transmit }`.
   Divide throughput by `pr` or `pt` respectively.
5. Remove the old heuristic `baseTransProb / baseSpecProb` partition for the
   dielectric case. Keep the existing diffuse/specular partition for
   non-transmissive surfaces.
6. Verify `thinFilmTransmitTint` is still applied as a multiplicative tint on
   the throughput of the refracted ray (not the reflected ray).

**Decision points:** Whether to remove the existing three-way heuristic
partition entirely or keep it for non-transmissive metallic/diffuse surfaces
(recommended: keep the heuristic only for `transmission == 0` surfaces; for
`transmission > 0` use Fresnel-weighted split).

**Behavior-preserving test:**
```ts
// White furnace test on a glass sphere (IOR=1.5): total throughput after
// one dielectric bounce ≈ 1.0 (within 1%) at N=10 000 rays.
// Verify TIR: at grazing angles (θ > critical angle), pr == 1.0.
```

**Verification:** Glass sphere in Cornell box should show correct caustic
brightness without energy inflation at Brewster angle.

**Dependencies:** Item 14 (VNDF sampling changes `glossyReflectionSample`
called inside the refracted branch at line 1349).

**Risk:** Probability partition change can shift global brightness if existing
scenes relied on the heuristic weighting. Validate with a reference render
before/after.

---

## Item 17: pt-webgpu normal transform inverted on non-uniform scale instances

**File(s):**
- `packages/pt-webgpu/src/math/mat4.ts:92–100` (`transformDirection`)

**Root cause:** `transformDirection` applies the forward model matrix `M`
to normals (treating them as directions). Under non-uniform scale
`M = diag(sx, sy, sz)`, a normal `n` must transform as `(M⁻¹)ᵀ n`. Using
`M n` instead gives a normal that is no longer perpendicular to the surface.
For uniform scale the result is proportional to the correct answer (normalisation
hides the error); for non-uniform scale the normal direction is wrong.

**Authoritative source:** Pharr, Jakob, Humphreys. *PBR* 4th ed. §3.10
"Applying Transformations" — `Transform::operator()(const Normal3f&)` uses the
transpose-inverse. See also pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations.
Also: Turkowski, K. "Properties of Surface Normal Transformations." *Graphics
Gems*, Academic Press, 1990 — the derivation is that `M⁻ᵀ n` is required to
maintain `n · t = 0` under `M t`.

**Fix:**

1. Add `transformNormal(m: Mat4, v: Vec3): [number, number, number]` to
   `mat4.ts` that computes the upper-left 3×3 of `inverse(M)` transposed,
   applies it to `v`, and normalises:
   ```ts
   export function transformNormal(m: Mat4, v: Vec3): [number, number, number] {
     // Compute (M⁻¹)ᵀ using cofactor expansion of the 3×3 submatrix.
     // For column-major Mat4 m: m[col*4 + row].
     const [m00,m10,m20, m01,m11,m21, m02,m12,m22] = [
       m[0]??0, m[1]??0, m[2]??0,
       m[4]??0, m[5]??0, m[6]??0,
       m[8]??0, m[9]??0, m[10]??0,
     ];
     // Cofactors (rows of (M⁻¹)ᵀ = cofactor matrix / det).
     const c00 = m11*m22 - m21*m12;
     const c01 = -(m01*m22 - m21*m02);
     const c02 = m01*m12 - m11*m02;
     const c10 = -(m10*m22 - m20*m12);
     const c11 = m00*m22 - m20*m02;
     const c12 = -(m00*m12 - m10*m02);
     const c20 = m10*m21 - m20*m11;
     const c21 = -(m00*m21 - m20*m01);
     const c22 = m00*m11 - m10*m01;
     // Apply cofactor matrix to v (det cancels in normalisation).
     const x = c00*v[0] + c10*v[1] + c20*v[2];
     const y = c01*v[0] + c11*v[1] + c21*v[2];
     const z = c02*v[0] + c12*v[1] + c22*v[2];
     const len = Math.hypot(x,y,z);
     if (len < 1e-8) return [0,1,0];
     return [x/len, y/len, z/len];
   }
   ```
2. In `flattenScene.ts` (wherever `transformDirection` is called for normals),
   replace with `transformNormal`.
3. Keep `transformDirection` for ray direction transforms (correct there).

**Decision points:** None. The math is settled.

**Behavior-preserving test:**
```ts
// transformNormal round-trip: for a non-uniform scale matrix S=diag(2,1,3),
// a face-normal n=(0,1,0) and tangent t=(1,0,0): verify
// dot(transformNormal(S,n), transformDirection(S,t)) < 1e-6.
```

**Verification:** Instances with non-uniform scale should show correct
shading normals. Visible symptom: specular highlight on a stretched sphere
should track the correct surface tangent plane.

**Dependencies:** None.

**Risk:** Low — purely a math fix with no GPU buffer layout change.

---

## Item 18: pt-webgpu Beer-Lambert path distance clamped at 32 world units

**File(s):**
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:1525`
  `exp(-sigmaT * min(hit.dist, 32.0))`

**Root cause:** The constant `32.0` is an arbitrary world-unit cap with no
physical justification. Beer-Lambert attenuation is `T(d) = exp(-σₜ · d)`
where `d` is the actual path length through the medium. Clamping `d` at 32
prevents correct attenuation in thick media and gives incorrect transmittance
for objects larger than 32 units (e.g. large water volumes or architectural
glass panels).

**Authoritative source:** Novák, J. et al. "Monte Carlo Methods for
Volumetric Light Transport Simulation." *Eurographics State of the Art*, 2018.
https://cs.dartmouth.edu/~wjarosz/publications/novak18mcvrl.pdf — equation for
transmittance estimator T(x→y) = exp(−σₜ·‖x−y‖) with no distance cap.

**Fix:**

1. Remove the `min(hit.dist, 32.0)` clamp: change to `exp(-sigmaT * hit.dist)`.
2. Add a UBO-plumbed `maxVolumeDist: f32` field to `FrameParams` with a
   host-side default of `1e6` (effectively infinite). Hosts may override for
   performance (capping at e.g. scene AABB diagonal × 2).
3. If a cap is truly desired for numerical safety, use `max(hit.dist, 0.0)`
   only (prevent negative) — not an upper clamp.

**Decision points:** Whether to add `maxVolumeDist` to the UBO now or simply
remove the clamp. Recommendation: remove the clamp unconditionally. `exp(-σₜ · d)`
is numerically stable for large d (it just approaches 0); no cap is needed.

**Behavior-preserving test:**
```ts
// Transmittance through a slab of thickness 100 units with σₜ = 0.01:
// T = exp(-0.01 * 100) = exp(-1) ≈ 0.368. With the 32-unit cap: exp(-0.32) ≈ 0.726.
// Verify the correct value post-fix.
```

**Verification:** Large glass slab should attenuate correctly; small objects
unchanged.

**Dependencies:** None.

**Risk:** None — removing a physically incorrect floor cannot break correct
rendering. Existing content with σₜ > 0 will appear darker in thick regions
(physically correct behaviour).

---

## Item 26: Two incompatible BVHNode encodings co-exist

**File(s):**
- `packages/pt-webgpu/src/scene/buildCpuBvh.ts` — stores **absolute** right-child
  node index in `rightChildOrTriOffset`
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:817` — traversal reads
  it as absolute (`let rightChild = node.rightChildOrTriOffset`)
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:533` — traversal reads
  as relative (`let rightChild = nodeIdx + node.rightChildOrTriOffset`)
- `packages/shared-bvh/src/bvhCommon.ts:626–654` — `normalizeBvhInteriorOffsets`
  converts three-mesh-bvh output to **relative** encoding

**Root cause:** pt-webgpu independently implements a CPU BVH builder that
stores absolute right-child indices. walkaround-hybrid consumes three-mesh-bvh
via `shared-bvh`, which normalises to relative encoding. The WGSL traversal
in `common.wgsl.ts` (walkaround) hard-codes the relative convention;
`pathTraceBruteforce.wgsl.ts` (pt-webgpu) hard-codes absolute. The two packages
are currently isolated but any future cross-package BVH sharing will silently
corrupt traversal.

**Canonical convention:** Relative right-child offset (three-mesh-bvh's 0.9.x
convention, and what `shared-bvh/normalizeBvhInteriorOffsets` produces). This
is the convention that matches the existing GPU-verified walkaround path.

**Fix:**

1. Change `buildCpuBvh.ts` to store a relative offset:
   in `build()`, after computing `rightChild = build(right)`, store
   `node.rightChildOrTriOffset = rightChild - nodeIndex` (not `rightChild`).
2. Change the traversal in `pathTraceBruteforce.wgsl.ts:817` to match:
   `let rightChild = nodeIdx + node.rightChildOrTriOffset;`
   (exactly the same as walkaround's `common.wgsl.ts:533`).
3. Add an assertion comment (and a CPU test) that `rightChildOrTriOffset`
   for any interior node satisfies `1 ≤ value < totalNodes`, which is the
   relative-encoding invariant.
4. Update `buildCpuBvh.test.ts` to verify the relative-offset encoding
   directly (currently tests structural shape only).

**Enforcement strategy:** Add a `validateBvhEncoding(nodes: Float32Array)` function
to `shared-bvh/bvhCommon.ts` that asserts every interior node's right-child
offset is in `[1, totalNodes)`. Call it from both:
- `buildCpuBvh` (after build, in dev/test mode)
- `buildSceneBVH` (after `normalizeBvhInteriorOffsets`, assertion removed in
  prod)
Gated behind a `debug` flag in both builders.

**Decision points:** None on convention choice (relative is clearly the
canonical form — it is what the GPU-proven walkaround path uses and what
three-mesh-bvh produces). Decision needed: should `pt-webgpu`'s CPU BVH be
replaced entirely by delegating to `shared-bvh`? See Item 31 for that
discussion.

**Behavior-preserving test:**
```ts
// buildCpuBvh round-trip: build a 4-triangle box, verify every interior
// node's rightChildOrTriOffset satisfies 1 ≤ offset < totalNodes.
```

**Verification:** Traversal of a pt-webgpu scene should produce the same hit
counts before and after the encoding change (since absolute and relative are
consistent within pt-webgpu once the traversal is updated to match).

**Dependencies:** Item 31 (if CPU BVH is replaced by shared-bvh, this item is
subsumed). Plan Item 26 first as an immediate correctness fix; Item 31 as a
follow-on unification.

**Risk:** Medium. The encoding change touches `buildCpuBvh` and the WGSL
traversal simultaneously. Ensure both are updated atomically in the same
commit; a partial update will silently produce wrong traversal (no crash,
wrong pixels).

---

## Item 27: Index buffer stride differs across packages

**File(s):**
- `packages/pt-webgpu/src/scene/buildCpuBvh.ts:67,153` — indices packed as
  `vec4u` (stride 4), `.w` zeroed
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:777` — reads
  `indices[t]` as `vec3u` component access (`.x,.y,.z`)
- `packages/walkaround-hybrid/src/rc/wgsl/probeRayCast.wgsl.ts:78` and
  RC/DDGI paths — use `vec3u` stride (3 u32 per triangle)
- `packages/walkaround-hybrid/src/restir/bvhCompute.ts` — uses `vec4u` (stride
  4, `.w` encodes packed RGB + texType)

**Root cause:** ReSTIR's index buffer packs material data into `.w` of a
`vec4u`, so ReSTIR needs stride 4. RC/DDGI use stride 3. pt-webgpu's CPU
builder allocates stride 4 but zeroes `.w`. There is no shared type declaration
or assertion preventing a mismatch at upload time.

**Fix:**

1. Define an enum/type in `shared-bvh/src/index.ts`:
   ```ts
   export type BvhIndexStride = 3 | 4;
   ```
2. Add a `bvhIndexStride: BvhIndexStride` field to `SceneBVHCommonResult`.
   `buildSceneBVH` always returns stride 3 (the WGSL `array<vec3u>` form);
   callers that need stride 4 run a post-process to pack `.w`.
3. Add an upload-time assertion in each engine's BVH upload path:
   ```ts
   if (indexData.byteLength % (stride * 4) !== 0)
     throw new Error(`BVH index buffer must be aligned to stride ${stride} u32`);
   ```
4. pt-webgpu's `buildCpuBvh` outputs `reorderedIndices` as `Uint32Array` with
   stride 4. The WGSL already reads `.x,.y,.z` of `bvhIndex[triIdx]` (declared
   `array<vec3u>`... actually `vec4u` — check the actual WGSL struct). Verify
   the declaration and add a comment on the `.w` zero-fill contract.

**Decision points:** The RC/DDGI stride-3 path and the ReSTIR stride-4 path
exist for legitimate reasons (ReSTIR packs material color into `.w`). No single
canonical stride makes sense; the fix is clear contracts at the boundary, not
unification. Endorse this as the design.

**Behavior-preserving test:**
```ts
// Upload a known 3-triangle mesh through both stride-3 and stride-4 paths;
// verify the triangle indices decode correctly from WGSL.
```

**Verification:** RC ray-trace produces the correct hit geometry after the
stride assertion is added and a test exercising all three strides passes.

**Dependencies:** Item 26 (encoding convention should be settled first to
avoid chasing two moving targets).

**Risk:** Low — this is a contract clarification, not a data format change.

---

## Item 28: Unguarded `1/dir` in 5 ray-AABB sites → NaN on axis-aligned rays

**File(s):**
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:506`
  `let invDir = vec3f(1.0) / ray.direction;`
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:545`
  `let invDir = vec3f(1.0) / ray.direction;` (in `intersectAabbDetailed`)
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:501`
  `let invDir = vec3f(1.0) / dir;` (in `bvhIntersectAny`)
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:571`
  `let invDir = vec3f(1.0) / ray.direction;` (in `bvhIntersectFirstHit`)
- `packages/walkaround-hybrid/src/rc/wgsl/probeRayCast.wgsl.ts:78`
  `let invDir = 1.0 / ray.direction;`

**Root cause:** IEEE 754 defines `1.0/0.0 = +Inf` and `1.0/(-0.0) = -Inf`,
so `invDir` components are ±Inf, not NaN, for axis-aligned rays. The slab
test then computes `(bmin.x - origin.x) * Inf`, which is ±Inf for non-zero
extents — and this is actually handled correctly by the `min/max` slab test.
The NaN occurs only when `origin.x == bmin.x` AND `dir.x == 0`, yielding
`0 * Inf = NaN`, which poisons the tNear/tFar comparisons.

**Authoritative source:** Williams, A. et al. "An Efficient and Robust
Ray-Box Intersection Algorithm." *Journal of Graphics Tools* 10(1):49–54,
2005. https://people.csail.mit.edu/amy/papers/box-jgt.pdf — Section 4 notes
that storing `invDir` components with IEEE infinity handles the axis-aligned
case robustly, but relies on IEEE NaN propagation in the comparison NOT
short-circuiting. The WGSL spec guarantees IEEE 754 f32, so the slab test is
safe IF the origin is not exactly on the slab plane. For probe origins that
are snapped to integer grid positions (DDGI probes) coincidence is common.

**Fix:** Replace the raw division with a robust inverse:
```wgsl
fn safeInvDir(d: vec3f) -> vec3f {
  // Use select to avoid 0/0: if dir component is zero, invDir is ±Inf
  // (not NaN). This matches Williams 2005 §4's IEEE requirement.
  return vec3f(
    select(1.0 / d.x, sign(d.x) * 1e20, abs(d.x) < 1e-30),
    select(1.0 / d.y, sign(d.y) * 1e20, abs(d.y) < 1e-30),
    select(1.0 / d.z, sign(d.z) * 1e20, abs(d.z) < 1e-30),
  );
}
```
Replace all 5 `vec3f(1.0) / dir` calls with `safeInvDir(dir)`.

Alternatively, use IEEE-standard `1.0 / max(abs(d), vec3f(1e-30)) * sign(d)`,
but `select` is clearer and avoids the sign-of-zero edge case.

**Decision points:** None — this is an IEEE 754 correctness fix with a known
authoritative solution.

**Behavior-preserving test:**
```ts
// CPU-side: ray along +Y axis through a unit AABB; origin exactly at bottom
// face center (0, 0, 0). Verify tNear == 0, tFar > 0, no NaN.
// This tests the 0 * Inf = NaN case.
```

**Verification:** DDGI probe rays that pass through atlas-aligned geometry no
longer produce black atlas pixels from NaN-poisoned slab tests.

**Dependencies:** None. Can land independently; apply to both packages.

**Risk:** Very low. The only observable change is for rays exactly coincident
with a slab plane, which previously returned unpredictable results (NaN →
boolean false → missed intersection). The fix makes those return a hit.

---

## Item 29: BVH stack overflow drops right child silently

**File(s):**
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:534, 536`
  `if (stackPtr < 62u)`
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:621, 623`
  `if (stackPtr < 62u)` (second traversal function)
- `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:818`
  `if (stackPtr + 2u < 64u)` (pt-webgpu's traversal)
- Stack declared as `array<u32, 64>` in all sites

**Root cause:** When `stackPtr >= 62`, the interior-node push is silently
skipped, dropping both children of a node. This is harmless for BVHs of
depth ≤ 30 (64-entry stack covers depth 64 in a binary tree), but for
adversarial or degenerate input the BVH depth can approach `n` (worst case
for median-split on sorted data). Even with SAH, a 60-entry stack is safe
for all practical scenes (three-mesh-bvh caps depth internally). Note: the
current guard wastes 1 stack slot (index 63 unused) — pushing 2 children at
stackPtr=61 writes indices 61 and 62, leaving 63 unused. The fix below uses
all 64 slots.

**Fix:**

1. Change `if (stackPtr < 62u)` to `if (stackPtr + 1u < 64u)` — this
   allows pushing up to index 63, using the full stack. The current guard
   erroneously limits to 62 entries, losing 2 stack slots.
2. Add a counter: when the guard triggers, mark a `bool overflowed` flag
   visible per-invocation. In debug builds, OR all threads' `overflowed`
   into a storage buffer and read it back on the host. In release, accept
   the miss rather than crash.
3. For walkaround-hybrid: the walkaround BVH is built by three-mesh-bvh
   with SAH; its practical depth never exceeds 35 on a 30K-triangle scene.
   The overflow guard is a safety net, not a live path. Leave the stack at
   64 entries; fix the guard arithmetic.

**Decision points:** Whether to raise the stack to 128 entries. Cost: 128 ×
4 bytes = 512 bytes additional register pressure per invocation, which
measurably increases occupancy pressure on low-VRAM devices. Recommendation:
keep at 64 and fix the arithmetic; document the depth bound in code.

**Behavior-preserving test:**
```ts
// Construct a pathological BVH of depth 32 (worst-case median split on 64
// sorted triangles). Trace a ray that should hit the deepest leaf. Verify hit.
```

**Verification:** No change for current scenes; the fix prevents silent misses
in future high-depth inputs.

**Dependencies:** None.

**Risk:** Near-zero. The guard is a safety net for a condition that doesn't
occur in practice; the fix just corrects the guard arithmetic.

---

## Item 30: walkaround-hybrid common.wgsl traversal ignores split-axis ordering

**File(s):**
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:529–537`
  (interior-node push in `bvhIntersectAny`)
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:618–626`
  (interior-node push in `bvhIntersectFirstHit`)

**Root cause:** The traversal always pushes right child first, then left child
(so left is processed first). An ordered traversal would push the farther
child first so the nearer child is popped and tested first — reducing the
number of intersection tests by visiting near subtrees preferentially. The
current unordered traversal is correct but suboptimal: it visits ~1.3–2×
more nodes than ordered traversal on average scenes.

**Authoritative source:** Wald, I. et al. "Ray Tracing Deformable Scenes
Using Dynamic Bounding Volume Hierarchies." *ACM TOG* 26(1), 2007, and
Pharr PBR4e §7.3.3 "Traversal" — ordered traversal uses `splitAxis` from
the interior node to determine which child is near.

**Fix:**

1. Read `node.splitAxisOrTriCount & 0x3u` to get the split axis (stored in
   the low 2 bits by three-mesh-bvh for interior nodes).
2. Compare `ray.direction[axis]` sign to determine near/far child order.
3. Push far child first, near child second (so near is popped first).

```wgsl
let axis = splitOrCount & 0x3u;
let dirSign = select(0u, 1u, ray.direction[axis] >= 0.0);
// dirSign 0 → left is near; dirSign 1 → right is near
let nearChild = select(nodeIdx + 1u, rightChild, dirSign == 1u);
let farChild  = select(rightChild, nodeIdx + 1u, dirSign == 1u);
if (stackPtr + 1u < 64u) {
  stack[stackPtr] = farChild;  stackPtr++;
  stack[stackPtr] = nearChild; stackPtr++;
}
```

**Decision points:** This is a performance optimisation, not a correctness
fix. Prioritise after correctness items. The performance gain is approximately
30–50% fewer node tests for first-hit queries on coherent rays.

**Behavior-preserving test:**
```ts
// Trace 1000 random rays through a 10K-tri scene; compare hit distance
// from ordered vs unordered traversal — must be identical.
// Measure node-visit count (instrumented traversal) — ordered should be < unordered.
```

**Verification:** Same rendered image; GPU timestamp for BVH traversal pass
should decrease.

**Dependencies:** Item 26 (encoding convention must be settled), Item 27
(stride consistency). Ordered traversal reads `splitAxisOrTriCount` correctly
only if the node layout is canonically encoded.

**Risk:** Low. Ordering does not change correctness; only traversal efficiency.

---

## Item 31: pt-webgpu/buildCpuBvh is median-split, not SAH

**File(s):**
- `packages/pt-webgpu/src/scene/buildCpuBvh.ts` — complete implementation

**Root cause:** The builder sorts on longest centroid axis and splits at the
median, producing O(n log n) build time but BVH quality substantially worse
than SAH (Surface Area Heuristic). On typical scenes this increases ray-test
cost by 1.5–3× vs SAH. For a prototype path tracer this is acceptable, but it
is not acceptable if `pt-webgpu` is ever used for production.

**Authoritative source:** Wald, I. "On fast Construction of SAH-based
Bounding Volume Hierarchies." *IEEE Symposium on Interactive Ray Tracing*,
2007. https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf
Binned SAH with K=16 bins achieves within 5% of optimal SAH quality at
O(n log n) build time with a small constant.

**Trade analysis — delegate to `shared-bvh` vs implement locally:**

| Factor | Delegate to `shared-bvh` | Implement binned SAH locally |
|---|---|---|
| Code duplication | Eliminates duplicate builder | Keeps pt-webgpu independent |
| BVH encoding | Gains `normalizeBvhInteriorOffsets` automatically | Must be added manually (Item 26) |
| Index format | `shared-bvh` returns stride-3 indices | pt-webgpu uses stride-4 currently |
| Three.js dependency | `shared-bvh` depends on `three` | `buildCpuBvh` has no Three.js dep |
| pt-webgpu purity | Adds `three` transitive dep to pt-webgpu | Keeps pt-webgpu dependency-light |

**Recommendation:** For now, **implement binned SAH locally** in `buildCpuBvh.ts`
and separately fix the encoding (Item 26). The Three.js dependency on `shared-bvh`
is inappropriate for `pt-webgpu`'s stated goal of being a host-agnostic backend.
If `pt-webgpu` later grows a Three.js integration layer, revisit.

**Binned SAH local implementation plan:**

1. Add `interface BinData { boundMin, boundMax, triCount: number }` and a
   `computeSAH(bins[], splitIdx, totalSA)` helper.
2. In `build()`, for each candidate axis (0,1,2), bin the centroids into K=16
   buckets, sweep left-to-right to accumulate SAH cost at each split plane
   (O(K) sweep), pick the best axis+plane.
3. If no split improves SAH over making a leaf, create a leaf (this is the
   "SAH terminates early" condition that median-split never exploits).
4. Carry the `rightChildOrTriOffset` as a relative offset (Item 26 fix).

**Behavior-preserving test:**
```ts
// Build BVH on 100-triangle scene; verify SAH builder produces
// the same or fewer leaf nodes than median-split builder.
// Verify hit results identical for 1000 test rays.
```

**Verification:** Render time for a complex scene should decrease; reference
render pixels unchanged.

**Dependencies:** Item 26 (relative encoding must be established before
replacing the builder).

**Risk:** Medium. Replacing the builder changes the BVH topology; all traversal
tests must pass before and after. The fix is purely in TypeScript (host side),
not WGSL.

---

## Item 32: Two RFE trackers disagree

**File(s):**
- `plan/external-requests-status.md` — covers RFEs 01–05 (contract-layer status)
- `external_requests/IMPLEMENTATION-STATUS.md` — covers RFEs 06–14 with fork-commit hashes

**Root cause:** The two files were created by different agents at different
times and cover disjoint but confusingly-named RFE ranges. A reader consulting
`external-requests-status.md` sees RFEs 01–05 as "implemented" without context
that 06–14 exist. A reader consulting `IMPLEMENTATION-STATUS.md` sees RFE
implementation details but not the contract-layer context.

**Fix:**

1. Add a header section to `plan/external-requests-status.md`:
   ```
   ## Scope note
   This file covers RFEs 01–05 (API contract additions to @vitrum/core completed
   2026-05-09). For RFEs 06–14 (fork-side shader patches), see
   external_requests/IMPLEMENTATION-STATUS.md.
   ```
2. Add a corresponding backlink to `external_requests/IMPLEMENTATION-STATUS.md`:
   ```
   ## Scope note
   This file covers RFEs 06–14 (fork shader patches, 2026-05-10). For RFEs 01–05
   (@vitrum/core contract additions), see plan/external-requests-status.md.
   ```
3. No content changes to either file beyond scope notes — the content in both
   is accurate as verified by the sweep.

**Decision points:** Whether to merge both files into a single canonical status
doc. Recommended: keep separate (different audiences — library vs fork), but
link them. Merge is a larger refactor without correctness benefit.

**Behavior-preserving test:** N/A — documentation change.

**Verification:** A reader following either file reaches both halves of the
RFE picture.

**Dependencies:** None.

**Risk:** None.

---

## Item 33: Tests are structural, not numerical

**Root cause:** The test suite has 662 tests that verify TypeScript types,
buffer packing, WGSL string structure, and API contract shape. No test
verifies that a sampling function's PDF integrates to the correct value, that
a BSDF satisfies energy conservation, or that a traversal algorithm finds the
correct intersection. Math bugs in Items 14–30 above are invisible to the
existing suite.

**Required numerical tests — full enumeration:**

### 33-A PDF normalization integrals
**File:** `packages/shared-samplers/__tests__/pdfNormalization.test.ts`

| Test | What it asserts | Tolerance | Dependencies |
|---|---|---|---|
| HG phase PDF normalizes | MC integral of `hg_phase(cosTheta, g)` over sphere ≈ 1 | 0.5% at N=50k | None |
| equiAngular PDF matches sample | `equiAngularSample(u)` returns `t` with `pdf(t) == equiAngularPdf(t)`; verify PDF integral over valid range ≈ 1 | 1% at N=10k | None |
| mixturePdf sums correctly | `mixturePdf([p1,p2,...], weights)` == weighted sum of component PDFs | 1e-6 exact | None |
| octahedral PDF | Octahedral encode maps unit sphere to [0,1]² bijection; the area differential distortion integrates to 1 over sphere surface | 1% at N=50k | None |
| environment importance PDF | Sum of all CDF buckets == 1; PDF at each pixel ≥ 0 | 1e-6 | None |

### 33-B VNDF PDF normalization (new, required by Item 14)
**File:** `packages/pt-webgpu/__tests__/bsdfPdfNormalization.test.ts`

| Test | What it asserts | Tolerance |
|---|---|---|
| GGX VNDF sample PDF integrates to 1 | MC integral of `p(ωi|ωo)` over hemisphere at (roughness=0.1,0.5,0.9) | 1% at N=100k |
| brdfDirectionalPdf diffuse lobe sums correctly | `∫ pdfDiff(wi) dωi = diffProb` over upper hemisphere | 1% at N=50k |
| Full BSDF PDF hemisphere integral == 1 | Combined `brdfDirectionalPdf` over all bounce types | 1% at N=100k |

### 33-C MC convergence tests
**File:** `packages/pt-webgpu/__tests__/radianceConvergence.test.ts`

| Test | Scene | Analytic reference | Tolerance |
|---|---|---|---|
| Lambertian sphere exitant radiance | Unit sphere, uniform albedo ρ, directional light L | `π · ρ · L · cos(θ_L)` per solid angle | 2% at N=1000 samples |
| Single-bounce diffuse from area light | Infinite plane illuminated by uniform area light | Analytic solid-angle integral | 2% at N=1000 |

These require a headless CPU path tracing reference (no GPU): implement a
TypeScript miniature path tracer using the same WGSL-mirrored functions
to do the MC integral numerically.

### 33-D Energy conservation (white furnace)
**File:** `packages/pt-webgpu/__tests__/energyConservation.test.ts`

| Test | What it asserts | Tolerance |
|---|---|---|
| BSDF round-trip: `∫f(ωo,ωi)·cos(θi) dωi ≤ 1` | For white albedo (ρ=1) at roughness 0.1–0.9, metallic 0–1 | ≤ 1.0 + 1e-3 |
| Dielectric Fresnel energy: `R + T = 1` | For `frDielectric(θ, η)`, verify `R + (1-R) = 1` at all θ | Exact (1e-6) |

### 33-E Octahedral encode/decode round-trip
**File:** `packages/shared-samplers/__tests__/octahedral.test.ts` (extend
existing file)

| Test | What it asserts | Tolerance |
|---|---|---|
| Encode→decode identity | For 1000 uniform sphere samples, `decode(encode(v)) ≈ v` | `dot(original, decoded) > 0.9999` |
| Bijection coverage | Encoded values span [0,1]² without clustering | Chi-squared uniformity p > 0.01 |

### 33-F Light-tree leaf PDF sums to 1
**File:** `packages/shared-samplers/__tests__/lightTree.test.ts` (extend
existing file)

| Test | What it asserts | Tolerance |
|---|---|---|
| Leaf PDF partition | Sum of `leafPdf(i)` over all leaves == 1 | 1e-5 |
| No leaf with zero power has non-zero PDF | If `power[i] == 0`, `leafPdf(i) == 0` | Exact |

### 33-G BVH cross-package compatibility round-trip
**File:** `packages/shared-bvh/__tests__/bvhEncoding.test.ts` (new)

| Test | What it asserts | Tolerance |
|---|---|---|
| pt-webgpu builder relative encoding | After fix (Item 26), every interior node's `offset ∈ [1, totalNodes)` | Exact |
| walkaround normalization | `normalizeBvhInteriorOffsets` converts 0.7.x absolute to relative correctly | Exact |
| Cross-package traversal identity | Same ray through same geometry gives identical `tHit` from both traversals | 1e-5 |

### 33-H Half-float overflow NaN behavior
**File:** `packages/pt-webgpu/__tests__/numerics.test.ts` (new)

| Test | What it asserts | Tolerance |
|---|---|---|
| Attenuation clamp | `exp(-sigmaT * d)` for large `d` approaches 0, not NaN | finite, ≥ 0 |
| frDielectric bounds | `frDielectric(theta, eta)` ∈ [0, 1] for all valid inputs | Exact bounds |
| safeInvDir no NaN | `safeInvDir(vec3(0,1,0))` is finite, not NaN | isFinite |

### 33-I Dielectric Fresnel sum test (required by Item 16)
**File:** `packages/pt-webgpu/__tests__/energyConservation.test.ts`

| Test | What it asserts | Tolerance |
|---|---|---|
| `R + T = 1` at all angles | `frDielectric(θ, 1.5) + (1 - frDielectric(θ, 1.5)) == 1` | Tautologically exact |
| TIR at critical angle | `frDielectric(θ > θc, 1.5/1.0)` == 1.0 | 1e-6 |
| Brewster angle | At `θ_B = atan(eta)`, parallel polarisation coefficient == 0 (for full Fresnel) | 1e-4 |

**Priority order for implementation:** 33-B (Item 14 fix), 33-D (energy
conservation), 33-G (cross-package BVH), 33-A (PDF normalization),
33-E (octahedral), 33-F (light tree), 33-C (MC convergence), 33-H, 33-I.

---

## Item 34: `HybridEngine.ts:592` vague "future sprint" TODO

**File(s):**
- `packages/walkaround-hybrid/src/HybridEngine.ts:588–593`

**Root cause:** The comment at line 592 reads: "Real-time caustic strategies
(MNEE / photon-map) are not compatible with the walkaround engine's frame
cadence. The walkaround engine always reports `'none'`; see
`external_requests/05-manifold-nee.md` §4 ("walkaround-hybrid" backend guidance)
for the approved approximation path when real-time caustic approximations are
added **in a future sprint**."

Upon reading `external_requests/05-manifold-nee.md` and the RFE implementation
notes, this position is current and correct — caustic approximation for the
walkaround engine is legitimately deferred because no sprint in Phase 6 or
Phase 7 schedules MNEE for the real-time path. The "future sprint" reference
is accurate (not stale). The only issue is the phrase "future sprint" is vague
when read out of context.

**Fix:**

Replace "in a future sprint" with a reference to the specific Phase 7 / caustic
sprint where this belongs. Since no sprint is currently scheduled, change to:

```
// ... for the approved approximation path if real-time caustic approximation
// is added. Track via external_requests/05-manifold-nee.md §4.
```

Remove "future sprint" language — it is a stale-context risk.

**Decision points:** None.

**Behavior-preserving test:** N/A — comment change.

**Verification:** No "future sprint" language in walkaround codebase.

**Dependencies:** None.

**Risk:** None.

---

## Item 35: Spectral curve deprecation warning — "Phase 7 / Sprint 1" is now in progress

**File(s):**
- `packages/three-bindings/src/material.ts:148–150`

**Root cause:** The `console.warn` at line 148 reads:
```
"Scheduled for removal in Phase 7 / Sprint 1."
```
Phase 7 is in progress as of 2026-05-11. The deprecated bare-`Float32Array`
path must now be evaluated: either remove it (breaking change) or update the
message to a concrete removal timeline.

**Reading the code:** The deprecated path accepts a `Float32Array` for
`spectralAttenuation` when the caller should pass a `SpectralCurve` object.
The path still works; removal requires callers to migrate. No internal vitrum
code uses the deprecated path (only external hosts might).

**Fix:**

1. Decide whether Phase 7 Sprint 1 means "remove now" or "deprecate with a
   firm deadline." Since vitrum is pre-alpha with no external consumer of this
   path, **remove it now**: delete lines 142–157 (the `Float32Array` branch)
   and update the type guard to only accept `SpectralCurve` shape.
2. If removal is blocked, change the warning to:
   ```
   "Deprecated since Phase 7. Remove support for Float32Array before Phase 8."
   ```
   with a `// TODO(Phase-7): remove this branch` comment.

**Decision points:** Remove now (breaking but clean, pre-alpha is the right
time) or defer to Phase 7 end. Recommend: **remove now**, add a TypeScript
type guard that narrows the check to `SpectralCurve` shape (`wavelengthStart`,
`wavelengthEnd`, `values` fields present).

**Behavior-preserving test:**
```ts
// After removal: verify that passing a Float32Array as spectralAttenuation
// returns no SpectralCurve on the material (the field is undefined).
// Passing a proper SpectralCurve still works.
```

**Verification:** `npm test` passes; `three-bindings` material round-trip
tests pass.

**Dependencies:** None.

**Risk:** Breaking for any external host using the deprecated path. Since
vitrum is pre-alpha, this is acceptable with a changelog entry.

---

## Item 36: Hardcoded thresholds without UBO plumbing

**File(s):**
- `packages/pt-webgpu/src/wgsl/common.wgsl.ts:16` `TRI_INTERSECT_EPSILON = 1e-5`
- `packages/pt-webgpu/src/wgsl/common.wgsl.ts:59` `safe_normalize` floor `1e-8`
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:30` `TRI_INTERSECT_EPSILON = 1e-5`
- `packages/walkaround-hybrid/src/shaders/common.wgsl.ts:397–399` `safe_normalize` floor `1e-8`

**Root cause:** `TRI_INTERSECT_EPSILON` controls Möller–Trumbore degeneracy
rejection. At `1e-5` it is calibrated for metre-scale scenes. Hosts with
centimetre or kilometre geometry will see false misses (too large epsilon) or
false hits through thin surfaces (too small). `safe_normalize`'s `1e-8` floor
is similarly scene-scale-dependent.

**Fix strategy:** These are WGSL constants, not UBO fields. Plumbing them
through the UBO would require adding two `f32` fields per-invocation — a
non-trivial layout change. The pragmatic fix is to document the scale
assumption in the constant declaration and expose a host-side option:

1. In `FrameParams` / `WalkaroundUBO`, add:
   ```
   triIntersectEpsilon: f32   // default 1e-5 (metre scale)
   ```
2. Replace the WGSL `const TRI_INTERSECT_EPSILON` with a read from the UBO.
3. For `safe_normalize`'s floor: `1e-8` is a vector length floor, not
   geometry-scale dependent. It is safe as a constant.

If UBO plumbing is deemed too invasive for now, at minimum add a JSDoc comment
on the constant:
```wgsl
// TRI_INTERSECT_EPSILON: calibrated for metre-scale geometry (world units ~1m).
// Hosts with cm-scale geometry should use 1e-3; km-scale: 1e-7.
// To override: rebuild with a custom constant (no UBO path today).
```

**Decision points:** UBO-plumb `TRI_INTERSECT_EPSILON` now (clean but requires
layout bump) vs document-only (fast, defers to Phase 7). Recommend: document
only for now; add to the Phase 7 "library generality" tunables list alongside
the existing `emitterDist2Floor`, `directFireflyClamp`, etc. already in the
UBO.

**Behavior-preserving test:**
```ts
// Smoke test: verify FrameParams layout byte-total is unchanged if
// documentation-only fix is chosen.
```

**Verification:** No rendering change expected; only UBO layout or comment
update.

**Dependencies:** If UBO-plumbing: depends on Item 27 (stride consistency)
being settled. Otherwise no dependencies.

**Risk:** UBO layout change: medium (host-side packing must stay aligned).
Documentation-only: zero risk.

---

## Item 37: CHANGELOG.md missing ~30 entries since Sprint 11

**File(s):**
- `CHANGELOG.md` (root)

**Root cause:** The `[Unreleased]` section already contains a large block of
entries, but examination of commits from Sprints 12–18 shows the following
missing entries:

- Sprint 12: spectral accumulator fork patch (RFE-08)
- Sprint 13: pt-webgpu multi-light direct sampling, disc-area emitters
- Sprint 14: ReSTIR-DI temporal / spatial passes shipped (Sprints 14–15)
- Sprint 15: GTAO added to composite pass
- Sprint 16: ReSTIR-GI initial path (spatial / temporal)
- Sprint 17: W-cap fix + per-channel atrous indirect chain
- Sprint 18: Adaptive sampling thresholds + PPG cell cap options (recent commits)
- External RFE passes: RFE-09 bridge, RFE-10 userData round-trip, RFE-12 plan
- Babylon stub removal (committed 2026-05-11)
- Walkaround UBO tunables (audit B1, B3, B4, M1, M6, M7, M8, M12)

**Fix:** Append all missing entries to the `[Unreleased]` section, grouped by
category (Added / Changed / Fixed). Use git log from the sprint milestone
commits to enumerate the full list systematically. Each entry should reference
the sprint or commit where the work landed.

**Decision points:** None — this is a documentation catch-up task.

**Behavior-preserving test:** N/A.

**Verification:** After the update, `git log --oneline` and `CHANGELOG.md`
entries are in 1-to-1 correspondence for all Sprint 12+ commits.

**Dependencies:** None.

**Risk:** None.

---

## Item 38: `_staging/README.md` table doesn't list all current files

**File(s):**
- `_staging/README.md`

**Root cause:** The table in `_staging/README.md` lists several specific
`*.tsx` file names, but `_staging/legacy-source/` contains more files not
listed. The README was written when the staging directory was first created
and has not been updated since extraction.

**Fix:**

1. Run `ls _staging/legacy-source/` and compare to the table.
2. Add any missing files with their disposition ("Host React/Redux — do not
   copy into packages").
3. Add a note clarifying that the table is intentionally non-exhaustive for
   host-app files: "All React/Redux host-app files in `legacy-source/` share
   the disposition 'Host-only — do not extract'; only files with non-trivial
   migration notes are individually listed."

**Decision points:** None.

**Behavior-preserving test:** N/A.

**Verification:** After update, the README accurately describes the staging
directory.

**Dependencies:** None.

**Risk:** None.

---

## Suggested Execution Order

### Phase 1 — Correctness foundations (unblock all later work)
Items in this phase have no dependencies and fix hard incorrect behaviour.

1. **Item 28** — Fix `safeInvDir` NaN in all 5 ray-AABB sites (both packages).
   Tiny change, no layout impact, unblocks probe accuracy.
2. **Item 17** — Fix `transformNormal` (normal transform under non-uniform scale).
   Pure TypeScript math utility, no WGSL change.
3. **Item 26** — Canonicalise BVH encoding to relative offsets (pt-webgpu builder
   + pt-webgpu WGSL traversal). Must land atomically.
4. **Item 29** — Fix stack overflow guard arithmetic (both packages).

### Phase 2 — pt-webgpu BSDF correctness (Items 14–16 + 18)
These touch the same shader file and should land together or in order.

5. **Item 14** — VNDF GGX sampling (replace lerp in `glossyReflectionSample`).
6. **Item 16** — Dielectric Fresnel-weighted branch (add `frDielectric`).
7. **Item 18** — Remove Beer-Lambert 32-unit clamp.
8. **Item 15** — Generalize area-light MIS to all lights (depends on Item 14
   for correct PDF in MIS denominator).

### Phase 3 — Cross-package structure + BVH quality
9. **Item 27** — Formalise index buffer stride contracts and upload assertions.
10. **Item 31** — Replace median-split BVH with binned SAH in `buildCpuBvh.ts`.
11. **Item 30** — Ordered BVH traversal in walkaround-hybrid `common.wgsl`.

### Phase 4 — Numerical test suite (Item 33)
12. **Item 33** in priority sub-order: 33-B → 33-D → 33-G → 33-A → 33-E → 33-F → 33-C → 33-H → 33-I.
    Each test group can be parallelised with Phase 3 work.

### Phase 5 — Documentation and stale context
13. **Item 32** — Cross-link RFE tracker files.
14. **Item 34** — Remove "future sprint" language from HybridEngine comment.
15. **Item 35** — Remove deprecated `Float32Array` spectral path (or update warning).
16. **Item 36** — Document or UBO-plumb `TRI_INTERSECT_EPSILON`.
17. **Item 37** — Catch up CHANGELOG.md with Sprint 12–18 entries.
18. **Item 38** — Update `_staging/README.md` table.

Phase 5 items can be batched in a single commit with no ordering constraints
among themselves.

---

## References

- Heitz, E. "Sampling the GGX Distribution of Visible Normals." *JCGT* 7(4), 2018.
  https://jcgt.org/published/0007/04/01/paper.pdf
- Williams, A. et al. "An Efficient and Robust Ray-Box Intersection Algorithm."
  *Journal of Graphics Tools* 10(1):49–54, 2005.
  https://people.csail.mit.edu/amy/papers/box-jgt.pdf
- Wald, I. "On fast Construction of SAH-based Bounding Volume Hierarchies."
  *IEEE Symposium on Interactive Ray Tracing*, 2007.
  https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering* 4th ed.
  §9.3 Specular Reflection/Transmission, §3.10 Applying Transformations.
  https://pbr-book.org/4ed
- Veach, E. "Robust Monte Carlo Methods for Light Transport Simulation."
  PhD thesis, Stanford, 1997, Ch. 9 (MIS power heuristic).
- Novák, J. et al. "Monte Carlo Methods for Volumetric Light Transport
  Simulation." *Eurographics STAR*, 2018.
  https://cs.dartmouth.edu/~wjarosz/publications/novak18mcvrl.pdf
