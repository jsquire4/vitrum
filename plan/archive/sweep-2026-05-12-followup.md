# Sweep follow-up — verification + closeout + future-sprint revival

**Date:** 2026-05-12 | **Branch:** off `feat/sweep-2026-05-11-fixes` (or `main` after merge)

This plan covers everything called out in the post-sweep
"what was skipped / what needs verification" review. It is **tiered** —
Tier 1 is closeout work (1–2 weeks); Tier 2 is future-sprint revival
(months, each phase a major milestone in its own right).

If you want to scope-cut, drop Tier 2 entirely — the future-sprint
placeholders already capture each one's design intent for whenever they
are scheduled.

---

## Tier 1 — Closeout (verification + skipped-mid-sweep + sub-agent re-verify)

Each phase below is self-contained. Phases A, B, C are large-ish (days
each). Phases D–G are smaller (hours each). All are sized so they could
ship as a follow-up branch (`feat/sweep-2026-05-12-closeout`) in
roughly the order given.

---

### Phase A — GPU reference renders for M5/M7/M8/M9

The repo already has a benchmark-runner harness:
`tools/benchmark-runner/scenario-presets.mjs` +
`capture-adapter-playwright.mjs` + `tools/reference-renders/baseline/`.
Currently 7 RFE scenarios are defined; one baseline PNG exists
(`rfe03-layered-front-back.png`). Extend, don't rebuild.

**A1. Define new verification scenarios (one per algorithmic change):**

Add to `scenario-presets.mjs`:
- `m5-glass-fresnel-grazing` — glass sphere on a textured floor; camera at
  45° above horizon to expose grazing-angle Fresnel highlight. Verifies
  Item 16 (frDielectric branch).
- `m5-multi-light-cornell` — Cornell variant with 2 rect-area lights on
  opposite walls. Verifies Item 15 (sum-MIS over all lights). Brightness
  should ≈ 2× single-light at converged spp.
- `m5-glossy-roughness-sweep` — same scene at roughness 0.1, 0.3, 0.5,
  0.7. Verifies Item 14 (Heitz VNDF) — highlight tightness should match
  GGX VNDF lobe shape.
- `m7-ddgi-grey-vs-white-cornell` — Cornell with grey wall (albedo=0.5)
  variant and white wall (albedo=1.0) variant. Indirect contribution on
  the grey wall should be ~half the white wall (verifies the corrected
  receiver math).
- `m7-ddgi-uniform-environment` — sphere in a constant-irradiance
  environment. With Halton SO(3) (Item 6) + Lambertian cosine kernel
  (Item 20), all probe atlas values should converge to a uniform value;
  the rendered sphere should appear uniformly lit. Pre-fix: pow(8) kernel
  + frozen rotation produced visible directional banding.
- `m8-ddgi-no-seam-darkening` — surface with smooth normals seen at a
  glancing angle to expose probe-atlas seams. Should show no
  cell-grid darkening rings after M8 border-fill landed.
- `m9-rc-uniform-environment` — same as the m7 variant but with the RC
  cascade pyramid as the GI source. Verifies #22 (per-bin Ω) + #21
  (merge integral).
- `m9-gtao-corner-shadows` — scene with a 90° corner. Should show the
  Jiménez 2016 slice integral producing measurably darker contact
  shadows than the pre-fix `(h1+h2)/π` HBAO approximation.
- `m9-albedo-edge-preservation` — checkerboard-floor Cornell with
  alternating black/white tiles under indirect DDGI. Material edges
  should stay crisp through the atrous-variance chain (no bleeding).
- `m17-stretched-sphere-shading` — sphere with `scale(2,1,1)`
  transform; observe specular highlight tracks the stretched surface
  normal correctly (M4 #17 transformNormal).
- `m18-thick-glass-attenuation` — glass slab of thickness 100 wu.
  Transmittance through the slab should ≈ `exp(-σ·100)` not
  `exp(-σ·32)` (M4 #18 Beer-Lambert clamp removed).

For each scenario, add `{ scenarioId, seed, resolution, bounces, spp }`
in `scenario-presets.mjs`. Use seeds that make MC noise reproducible.

**A2. Generate baseline PNGs against `main` (pre-sweep):**
```bash
git checkout main
VITRUM_GPU_CAPTURE=1 \
  VITRUM_ALLOW_BASELINE_GEN=1 \
  VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
  npm run benchmark:gap-closure
# Move generated PNGs to `tools/reference-renders/baseline/<scenarioId>.png.pre-sweep`
```

**A3. Generate post-sweep PNGs against `feat/sweep-2026-05-11-fixes`:**
Same procedure. Move to `…/<scenarioId>.png.post-sweep`.

**A4. A/B compare:**
For each scenario, eyeball + numeric diff (PSNR / SSIM). Acceptance:
- Visual changes must be in the direction the math predicts
  (e.g., m9-gtao-corner-shadows post-sweep should be DARKER in corners,
  unchanged in open-sky pixels).
- No NEW artifacts (e.g., m7 should not introduce flicker; m8 should
  not over-brighten seams).
- File a `tools/reference-renders/sweep-2026-05-11-diff-report.md`
  documenting each scenario's outcome.

**A5. Adopt post-sweep PNGs as new baselines:**
After A/B sign-off, replace the baseline PNGs with the post-sweep
captures. Future regressions will diff against these.

**Effort estimate:** 2 days of GPU-machine time + 1 day of
scenario-authoring + diff analysis. Requires the GPU machine the user
has access to.

**Decision point — VITRUM_STRICT_GAP_CLOSURE behavior:** the existing
runner has a strict-mode flag (`tools/benchmark-runner/README.md`
documents it). Decide whether the new scenarios participate in the
gap-closure exit-status gate or are advisory-only. Recommend advisory
until the post-sweep baselines are stable (~2 cycles).

---

### Phase B — CPU mini-ray-tracer for 33-C (MC convergence tests)

Land the M3 / 33-C tests that were deferred. The 33-C convergence
asserts catch any future BSDF / integrator regression at the
system-output level — orthogonal to the per-component PDF/energy tests
already in M3.

**B1. New package or in-package?**
Two options:
- **(a)** Add to `packages/pt-webgpu/__tests__/`, mirroring the WGSL
  in TS test-only helpers. Pro: collocated. Con: large per-test mirror
  duplication.
- **(b) Recommended** Create a small `packages/cpu-mini-tracer/` (or
  `packages/test-fixtures/cpu-tracer.ts` inside an existing package).
  Provides a clean reference implementation that tests in any package
  can import.

**B2. Mirror scope (test-only utility):**
- Cosine-hemisphere sample + PDF (already in shared-samplers — re-export)
- GGX VNDF sample + PDF (mirror Heitz 2018 Algorithm 1 once; use across
  pt-webgpu + future BSDF tests)
- frDielectric (mirror PBR4e §9.3)
- Schlick Fresnel
- Möller–Trumbore triangle test
- Williams-2005-style ray-AABB with safeInvDir
- Power-heuristic MIS (β=2)
- Russian roulette with throughput compensation

All in pure TypeScript with deterministic LCG seeded RNG.

**B3. Integrator skeleton (≤200 lines):**
```ts
function integratePath(scene, ray, rng, opts): Vec3 {
  let throughput = vec3(1);
  let radiance   = vec3(0);
  for (let bounce = 0; bounce < opts.maxBounces; bounce++) {
    const hit = traverseBvh(scene.bvh, ray);
    if (!hit) { radiance = add(radiance, mul(throughput, scene.envSample(ray.dir))); break; }
    // NEE direct light:
    const direct = sampleDirectLight(scene, hit, rng);
    radiance = add(radiance, mul(throughput, direct));
    // Sample next bounce:
    const sample = sampleBsdf(hit, rng);
    if (sample.pdf <= 0) break;
    throughput = mul(throughput, mul(sample.f, sample.cosTheta / sample.pdf));
    if (bounce > 2) {
      const survival = clamp(maxComp(throughput), 0.1, 0.95);
      if (rng() > survival) break;
      throughput = scale(throughput, 1 / survival);
    }
    ray = { origin: hit.point, dir: sample.wi };
  }
  return radiance;
}
```

**B4. Test fixtures (4 tests, in `packages/pt-webgpu/__tests__/mcConvergence.test.ts`):**
- **Lambertian sphere → analytic exitance**: unit sphere with `albedo = 0.7`, directional light `L = 1, θ_L = 30°`. Analytic: `L_exitant_per_sr = π · ρ · L · cos(θ_L) / π = ρ · L · cos(θ_L)`. Tolerance ±1% at N=2000 spp.
- **Single-bounce diffuse from area light**: infinite plane illuminated by uniform unit-area light. Analytic from solid-angle integral. ±2% at N=2000 spp.
- **White furnace (ρ=1, environment L=1)**: any scene path-trace should converge to L=1 at every pixel (energy conservation). ±0.5% per channel at N=5000 spp; this is the strict catch-everything regression test.
- **Mirror reflection test**: perfect mirror, environment with single bright pixel, expected outgoing radiance is the reflected-direction environment value. ±0.1% at N=100 spp (mirror has low variance).

**B5. CI gating:**
These tests are slow (4 × ~5 sec each). Mark with `it.concurrent` and
gate on a separate `npm run test:slow` script if they exceed normal
suite latency. Or always-on if total adds <30s.

**Effort estimate:** 3–4 days. The TS mirrors are mechanical; integrator
correctness needs care.

**Decision point — BVH traversal scope:** the tracer needs *some* BVH
traversal to work on real geometry. Two options:
- (a) Use the same binned-SAH builder from M6 in pt-webgpu. Same
  encoding; tests catch encoding drift.
- (b) Use a brute-force "hit nearest tri" inner loop. Simpler; fast
  enough at small triangle counts.
Recommend (b) for the tracer's own test fixtures (≤50 tris); the BVH
correctness is already covered by 33-G.

---

### Phase C — DDGI behavior tests (CPU emulation)

The entire DDGI pipeline (producer → blend → border → receiver) has
zero behavior tests. Pass-layout structural tests and the M8 mirror
formula tests exist; nothing measures atlas content vs analytic.

**C1. CPU mirror of the DDGI pipeline (`packages/walkaround-hybrid/__tests__/ddgiPipeline.test.ts`):**
- Mirror `probeUpdateRays.wgsl.ts` ray accumulation (post-M7: stores
  raw L_i — no albedo/π baking).
- Mirror `probeUpdateBlend.wgsl.ts` cosine-weighted hemisphere
  averaging (post-M7: `max(0, n·d)` Lambertian, not `pow(8)`).
- Reuse the M8 border-mirror formula function.
- Mirror `applyDDGIShading.ts` receiver math: `L_o = (albedo/π) · E`.

**C2. Behavior assertions (4 tests):**
- **Uniform white room → atlas converges to L_in**: scene = closed
  white box (ρ=1, all six faces) lit by uniform interior emission
  L=1. After 50 frames of Halton SO(3) ray sampling + EMA blend, the
  per-probe atlas E should converge to `π · L = π` (irradiance from
  uniform environment over the upper hemisphere). Tolerance ±2%.
- **Receiver multiply produces Lambertian outgoing**: given E from
  the test above, receiver `L_o = (albedo/π) · π = albedo`. For
  albedo=0.5, output should be 0.5 ± 1e-3. Confirms the M7 energy
  model end-to-end.
- **Halton SO(3) decorrelation**: 192 rays/probe at frame 0 vs frame
  1 should have <50% direction overlap (verifies the rotation actually
  rotates).
- **Cosine-kernel non-negativity**: for any blended atlas value, no
  per-channel value is negative (basic sanity; pre-M7 `pow(8)` could
  produce edge-case negatives at low ray counts).

**Effort estimate:** 2 days. The CPU mirror is straightforward; the
"converges to π" test needs careful sample-count selection to balance
runtime vs MC noise.

---

### Phase D — Test-gap fills (small surgical adds)

**D1. VNDF "integrates to 1" via importance sampling.**

The M3 / 33-B test pivoted to property tests. Add the explicit
normalization test using importance sampling (avoids the variance
problem of uniform-hemisphere MC):

```ts
// Importance-sample VNDF, weight = 1, sum should equal hemisphere area
// for a normalized PDF. With p(ω) = VNDF(ω | wo), the importance-sampled
// estimator of `1` is `(1/N) · Σ 1/p(ω_i) · p(ω_i) = 1` — but trivially.
// The right test: importance-sample VNDF, weight by the GGX D distribution
// directly, the integral should converge to `1` (since VNDF is normalized).
```

Specifically: integrate `D(h) · max(0, ω_o · h) / (ω_o · n)` over the
hemisphere — this IS the VNDF PDF, integrated against itself with
importance sampling gives 1.0 ± noise. N=10000 samples, ±1% tolerance.

File: `packages/pt-webgpu/__tests__/energyConservation.test.ts` (extend).

**D2. 2-light sum-MIS correctness.**

For Item 15 (multi-light area MIS), add a CPU test:
- Scene: 2 point lights at known positions, flat surface, evaluate
  the BSDF→light MIS contribution.
- Expected: brightness ≈ 2× single-light variant; per-light contributions
  sum within MC tolerance.

File: `packages/pt-webgpu/__tests__/multiLightMis.test.ts` (new).

**D3. BVH cross-package full hit-result round-trip on a complex scene.**

Extend `packages/shared-bvh/__tests__/bvhEncoding.test.ts` with one more
test that builds a BVH on a 100-triangle test mesh (load a small `.obj`
fixture or generate procedurally), then traces 1000 random rays through
both pt-webgpu and walkaround-hybrid traversal mirrors. Hit `tHit`
must agree within 1e-5 for every ray.

**Effort estimate:** 1 day total for D1–D3.

---

### Phase E — Surgical deferred items

**E1. GTAO multi-bounce term (Jiménez 2016 §5.2 / Eq. 16):**

Currently deferred because GTAO bind group has no albedo G-buffer.

Steps:
1. Wire the existing `hdrAlbedoOut` (now produced by M9.C shade pass)
   into the GTAO bind group. Update `bindGroupLayouts.ts` and
   `bindGroupBuilders.ts`.
2. In `gtao.wgsl.ts`, add the multi-bounce factor:
   ```wgsl
   let a = 2.0404 * albedo - vec3f(0.3324);
   let b = -4.7951 * albedo + vec3f(0.6417);
   let c = 2.7552 * albedo + vec3f(0.6903);
   let aoMb = ((a * vis + b) * vis + c) * vis;
   ```
3. Replace the scalar AO output with the per-channel multi-bounce
   variant. Composite consumer multiplies by GI/ambient signal.
4. Test: white surface (ρ=1) should have `a_mb ≥ a_raw` (multi-bounce
   brightens). Add to `sprint15-gtao.test.ts`.

**E2. M4 #36 — `TRI_INTERSECT_EPSILON` in `rc/wgsl/probeRayCast.wgsl.ts`:**

That shader binds `CascadeUniforms`, not `WalkaroundUBO`. Two options:
- **(a)** Add `triIntersectEpsilon: f32` to `CascadeUniforms` struct
  (touches `cascadeBuffers.ts` + `cascadeDispatch.ts` + the WGSL).
  Plumb through `HybridEngine.triIntersectEpsilon` option that already
  exists from M4.A.
- **(b)** Document the local constant explicitly:
  ```wgsl
  // CASCADE-LOCAL constant (different UBO than walkaround). Bound to
  // 1e-5 (metre-scale geometry); host-override requires extending
  // CascadeUniforms — currently no host need.
  const TRI_INTERSECT_EPSILON: f32 = 1e-5;
  ```

Recommend (a) for consistency with the project's UBO-plumb pattern.
Effort: 2 hours.

**Effort estimate:** 1 day for E1 + E2.

---

### Phase F — Independent re-verification of 5 sub-agent claims

Per CLAUDE.md "sub-agent reports are hypotheses". Spot-checked the
highest-impact claims during the audit; these 5 weren't independently
re-verified:

**F1. M5 `bsdfAreaLightConnectionContribution` rewrite:**
Read `pathTraceBruteforce.wgsl.ts` `bsdfAreaLightConnectionContribution`
end-to-end. Verify:
- Loop iterates `params.rectAreaLightCount` AND
  `params.meshAreaLightCount` (not just one or the other).
- Closest-hit selection picks the light at min `t`.
- PDF multiplied by `(rectAreaLightCount + meshAreaLightCount)` to
  cancel uniform light-selection.
- TIR/refraction interaction at glass surfaces unchanged.

Add a CPU test (D2 above) that exercises 2-light MIS. If the loop
math is wrong, the brightness test fails.

**F2. M6 binned SAH builder:**
Read `packages/pt-webgpu/src/scene/buildCpuBvh.ts` `build()` end-to-end.
Verify:
- SAH cost formula: `cost = traversal + leftSA·leftCount + rightSA·rightCount`
  (not just left+right counts without SA weighting).
- Leaf-cost vs split-cost early-exit threshold sane.
- Bins iterated for all 3 axes (not just 1).
- Stride-4 index packing preserved.

Add a 100-triangle scene SAH-quality assertion: total tree node count
should be ≤ median-split's count + 50% (binned SAH typically reduces
by 30%; allow margin). This is informational, not strict.

**F3. M7 Halton axis-angle conversion:**
Read `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:670–718`.
Verify:
- Shoemake quaternion form correct (sigma1/sigma2 split is right).
- `sin(θ/2) = sqrt(1 - qw²)` correct (Shoemake gives a unit
  quaternion, so `qw² + qx² + qy² + qz² = 1`).
- Axis-angle output matches the WGSL Rodrigues consumer (does the
  consumer expect `axis * angle` packed in vec3, or separate axis +
  scalar angle?).

Add a unit test (TS): for 1000 frame indices, the rotation matrix
constructed from Halton should be uniformly-distributed-on-SO(3).
Verify by Frobenius distance to identity is reasonable; verify
det(R) = 1 always; verify R · R^T = I.

**F4. M9.A RC merge solid-angle weighting integration:**
Read `cascadeMerge.wgsl.ts` `octCellSolidAngle` + the merge formula.
Verify:
- `octCellSolidAngle(cx, cy, N)` returns the same value as the
  Float32Array from `computeOctahedralSolidAngles(N)[cy*N + cx]`.
- Merge formula `Σ child · Ω_child / Σ Ω_child` is the weighted
  average (not weighted sum without normalization).

Add an integration test: render a uniform-environment cascade,
verify per-cascade brightness is consistent across cascades (Phase A
m9-rc-uniform-environment scenario covers this on GPU; this is a CPU
mirror that runs on CI).

**F5. M9.C albedo demodulation pipeline plumbing:**
Read the 10 files M9.C touched. For each:
- `shade.wgsl.ts`: confirm `Lo_indirect` no longer multiplies by
  `albedo`; `hdrAlbedoOut` written; `hdrTotalOut` re-applies albedo.
- `indirectCombine.wgsl.ts`: confirm `output = direct + filtered_lighting · albedo`.
- `resourceManager.ts` / `bindGroupLayouts.ts` / `bindGroupBuilders.ts`:
  confirm new bindings allocated + bound.
- `WalkaroundGPUPipeline.ts`: confirm dispatch order unchanged.
- `atrousVarianceWebGPU.ts`: confirm optional `albedoRgb` parameter
  routes correctly.
- Tests: confirm checkerboard albedo with uniform L produces uniform
  L/albedo demodulated buffer (already in M9.C tests).

If anything reads "wrong shape" — file as a defect.

**Effort estimate:** 2 days for F1–F5.

---

### Phase G — Stray cleanup + audit follow-ups

**G1. Investigate commit `0f09b63`:**
Background: `0f09b63` (M9.A self-commit) covers Items 21+22; `9f5e9f1`
(M9 squash) covers 21+22+23+24. Diff between them shows `9f5e9f1`
contains all of `0f09b63`'s changes plus the additional GTAO/atrous
work. Conclusion: history is merely incremental, not duplicated.
Action: `git log --reverse 0f09b63..9f5e9f1 -- packages/walkaround-hybrid/src/rc/`
to confirm all RC content from `0f09b63` survives into `9f5e9f1`. Document
finding in the closeout retrospective.

**G2. Investigate `external_requests/09-runtime-lighting-updates.md`:**
This file is a real RFE-09 proposal ("Runtime Lighting Updates Without
Pipeline Rebuild") — substantive content, not garbage. It was added
during the sweep without explicit attribution in any milestone. Action:
- Confirm it's the RFE proposal, not an artifact.
- If keeping, add a one-line entry to `external_requests/IMPLEMENTATION-STATUS.md`
  noting RFE-09 is now Proposed (currently shown as Applied per M10
  CHANGELOG entry — there's a contradiction).
- File status: keep, but cross-reference correctly.

**G3. Verify M4.D's backtick-template-literal fix:**
M4.D agent claimed they fixed a "pre-existing build blocker" in
`walkaround-hybrid/src/shaders/common.wgsl.ts` by changing backtick-
quoted code examples in WGSL comments to single quotes. Verify:
- The change is comments-only (no shader semantics affected).
- The `git blame` on the affected lines shows the backticks pre-dated
  the sweep.
- esbuild parses the file cleanly post-fix (test harness loads it).

If the change accidentally modified actual WGSL code, file as a defect.

**Effort estimate:** ½ day for G1–G3.

---

### Tier 1 totals

- Phase A: 3 days (incl. 2 days GPU machine time).
- Phase B: 4 days.
- Phase C: 2 days.
- Phase D: 1 day.
- Phase E: 1 day.
- Phase F: 2 days.
- Phase G: ½ day.

**Tier 1 total: ~13.5 days, possibly compressible to 1.5–2 weeks with
parallelism (B/C/D/E independent of F/G; A serializes on GPU access).**

---

## Tier 2 — Future-sprint revival (large; user discretion)

These were explicitly DELETED or RENAMED-AS-STUB in the sweep because
the existing implementations were not paper-faithful. Each is a
multi-week project. Spec preserved in dedicated plan docs.

If you want any of these now, treat each as its own sprint; do NOT
bundle into the closeout branch.

### Phase H1 — Real Schied 2017 SVGF

**Doc:** `plan/sprint-svgf-real-future.md` (created in M1).

Adds: motion-vector reprojection pass, history-length texture
(persistent r16uint), momentsHistory texture (persistent rg32float
M1+M2), disocclusion test on depth+normal+objId, variance-guided
α-clamp, paper Eq. 4 edge-stop form, 7×7 spatial fallback.

Replaces the `'atrous-variance'` denoiser mode (or adds a new `'svgf'`
mode that survives alongside).

**Effort:** 2–3 weeks. Requires GPU iteration.

### Phase H2 — Neural denoiser

**Doc:** `plan/sprint-neural-denoiser-future.md` (created in M1).

Builds: real `train.py` PyTorch pipeline; ONNX export; WGSL kernels
fixed (8 enumerated scaffold bugs from sprint13's deletion); wire as
`'neural'` denoiser mode in HybridEngine. Compare against OIDN bridge.

**Effort:** 4–8 weeks (training + integration). Requires GPU + a
training dataset.

### Phase H3 — PPG paper-faithful rebuild

**Doc:** `plan/sprint-ppg-rebuild-future.md` (created in M1).

Builds: adaptive sTree (Müller §3.1, sample-count split), adaptive
dTree (§3.2, flux-fraction directional refinement), train on incoming
radiance L_i (not outgoing L_o), MIS with BSDF (§3.4), per-leaf
solid-angle weights from leaf area.

Requires atomic-update GPU storage for the trees (compute pass
splits cells mid-frame).

**Effort:** 3–4 weeks.

### Phase H4 — Full Veach §10.3 BDPT

**Doc:** `plan/sprint-bdpt-veach-full-future.md` (created in M1).

Replaces `bdptConnectionMIS_partial` with full strategy enumeration:
recursive `p_{s+1}/p_s` ratio per PBR4e Eq. 16.16, specular-vertex
zero-weight handling, area↔solid-angle G factor, camera/light endpoint
PDFs.

Wires fork-side BDPT dispatch (currently gated, was Sprint 10c).

**Effort:** 2–3 weeks.

### Phase H5 — Sprint 14 layered BSDF fork patch

**Doc:** `plan/sprint-14-layered-bsdf-fork-patch.md` (existing, gated).

Fork patch to `~/projects/three-gpu-pathtracer/`. Adds layered BSDF
support to the WebGL2 path tracer.

**Effort:** 2–4 weeks (depends on fork's current state).

### stainedGlass-driven adds (audit 2026-05-12)

Added per the gap audit at `plan/stainedGlass-gap-audit-2026-05-12.md`.
Folded into this in-flight branch since the user explicitly authorized
"get it in now while we're doing this work."

- **SG.A — anisotropy round-trip (Gap 5):** add `anisotropy` +
  `anisotropyRotation` to `@vitrum/core` Material; read in
  `convertMaterial`; write in `vitrumSceneToThree`. Ripple/waterglass
  cells gain correct anisotropic highlight in PT. ~½ day.
- **SG.B — three.js peer dep relaxation (Gap 4):** vitrum requires
  `three ^0.184`; stainedGlass at 0.171. Either (a) relax peerDeps to
  `>=0.171 <0.190` after API audit, or (b) hard-document the upgrade
  requirement. Eliminates `as any` casts at every host bridge call.
  ~½ day.
- **SG.C — `HybridEngine.updateLighting()` (Gap 1):** runtime light
  update API. Re-uploads sun UBO + invalidates DDGI probe cache +
  resets temporal accumulator. Eliminates the engine-recreation
  workaround documented at `useVitrumWalkaroundEngine.ts:34` (which
  blocks smooth time-of-day scrubbing). Medium scope, ~2 days. Land
  after H3 (PPG) commits to release HybridEngine.ts.
- **SG.D — per-material spectral pipeline through fork (Gap 2):** THE
  load-bearing find. `forkUniformBridge.ts` uploads only global CMF
  tables; per-material `spectralAttenuation` / `thinFilmStack` /
  `scatteringCoefficient` are dropped at the bridge. Cobalt, iron,
  Se/Cd, gold-ruby glass all render via RGB approximation in current
  PT renders. Fix requires:
    1. pt-webgl: per-material spectral upload path
       (`forkUniformBridge.ts` extension).
    2. Fork: extend `MaterialsTexture` packing to carry per-material
       spectral data (Sprint 12 was "partial" — this completes it).
    3. Fork: BSDF spectral consumer reads from MaterialsTexture
       instead of global tables.
  Naturally folds into the H6 chain since both are fork-side work
  and Sprint 5's MRT G-buffer infrastructure is the structural
  prerequisite. Schedule as **H6.6 — Sprint 12 completion** after
  Sprint 10c (H6.5). Multi-day fork patch.

**Surprising finding NOT in scope here** — surprise #3 in the audit:
walkaround engine bypasses `sceneFromThreeJS` and reads raw THREE.Scene.
All `userData.vitrum*` stamps are available but only `surfaceTextureId`
is read. The contract is asymmetrically enforced (pt-webgl honors it;
walkaround doesn't). That's a **Tier 3 architectural concern** — folds
into T3.A (unified `createEngine` + canonical Scene flow) when Tier 3
ships.

### Phase H6 — BDPT chain (Sprints 4 → 5 → 6 → 10c) — EXPANDED 2026-05-12

**Originally:** "Sprint 10c BDPT fork dispatch."

**Re-scoped after discovery:** Sprint 10c (BDPT integrator) requires the
Sprint 5 MRT G-buffer infrastructure (`WebGLMultipleRenderTargets` with
gColor + gNormalDepth + gAlbedo). The fork team intentionally skipped
Sprints 4/5/6 (going 3 → 7 → 8 → 12 → 14 directly), so 10c never had
its prerequisite. To ship H6, the prerequisite chain must land.

**Tangible benefit (justifies the scope expansion):**
For stainedGlass specifically — the canonical app for vitrum — caustic
patterns through colored glass onto interior surfaces are the
load-bearing visual feature. Walkaround can't render them
(ReSTIR-GI restricted to diffuse hits; no specular-vertex caustics).
PT mode renders them but slowly. BDPT cuts the final-render time
30min → 5min for stained-glass scenes. That's the difference between a
usable client-facing render workflow and a "render overnight" workflow.

**Sub-phases (serial):**
- **H6.1** — Apply Sprint 4 fork patch per `plan/archive/sprint-4-pt-fork-patch.md`. Small.
- **H6.2** — Apply Sprint 5 MRT G-buffer per `plan/archive/sprint-5-pt-fork-patch.md` + `sprint-5-mrt-gbuffer-spec.md`. STRUCTURAL fork change.
- **H6.3** — Regression-test Sprints 7 (volume scatter, fork commit `260c432`), 8 (chromatic dispersion `7ffd15d`), 12 (hero spectral `8917492`), 14 (layered BSDF `ee379dc`) against the new MRT base. Forward-port any broken integrator paths.
- **H6.4** — Apply Sprint 6 fork patch per `plan/archive/sprint-6-pt-fork-patch.md`. Builds on Sprint 5.
- **H6.5** — Apply Sprint 10c BDPT integrator (per the original spec, was at `plan/sprint-10c-pt-fork-patch.md` — now likely moved to `plan/archive/`). GLSL port of H4's `bdptConnectionMIS_full` power heuristic. Vertex pdf storage uses the Sprint 5 MRT ping-pong infrastructure.
- **H6.6 — Sprint 12 completion (per-material spectral)** — per stainedGlass audit Gap 2. Extend `MaterialsTexture` in the fork to carry per-material `spectralAttenuation` (32 bins × 4 bytes), `thinFilmStack` (variable; 8-layer cap), `scatteringCoefficient`. BSDF reads from MaterialsTexture for these. pt-webgl `forkUniformBridge.ts` uploads per-material spectral data alongside the existing global CMF tables. Cobalt/iron/Se-Cd/gold-ruby glass start rendering with TRUE per-wavelength absorption instead of RGB Beer-Lambert approximation. ~3-5 days fork work.
- **Vitrum-side bridge** (small): pt-webgl exposes a `bdpt: boolean` option in `createPTEngine_WebGL2`; the fork interprets it.

**Effort:** 3–4 weeks. The risk is H6.3 — Sprints 7/8/12/14 may not adapt cleanly to MRT and need re-authoring.

**Verification gates:**
- After H6.2: existing fork build + smoke render must succeed (catches obvious regressions).
- After H6.3: each of Sprints 7/8/12/14 must produce visually-equivalent output to its pre-MRT version on a reference scene.
- After H6.5: BDPT vs unidirectional-PT on a glass+caustic scene must show measurable variance reduction at fixed sample count (the whole reason BDPT exists).

**Why earlier "BLOCKED" was wrong:** The dispatched agent looked in `plan/` for prerequisite specs, found nothing, declared blocked. Specs exist in `plan/archive/`. The deeper truth (deliberate fork-team skip; multi-week scope) is real, but the user explicitly authorized the expanded scope; H6 is back in.

### Tier 2 totals

If all 6 land: ~3–6 months of single-developer work, not counting GPU
iteration time. Each phase has a hard prerequisite of GPU access for
verification.

**Recommend:** schedule individually as separate sprints. Don't bundle.

---

## Recommended scoping

If you want a pragmatic close on this work without committing to Tier 2:

1. **Run Tier 1 in full** as a single closeout branch
   (`feat/sweep-2026-05-12-closeout`). ~2 weeks.
2. **Defer Tier 2** items to the future-sprint backlog (their
   placeholder docs are already in `plan/`).
3. **Decision point per Tier 2 item** — schedule individually when
   the motivating use case is concrete. Don't pre-commit.

If you want everything: budget ~4 months total.

If you want to scope-cut Tier 1: the highest-value subset is
**A + C + F** — GPU verification + DDGI behavior tests + sub-agent
re-verify. This catches the highest-risk items (visual regressions in
M7/M8/M9 + the 5 unverified agent claims). ~1 week.

---

## Single decision the user must answer before execution

**Tier 2 scope:** are H1–H6 in or out of this follow-up?

- **Recommendation: out** — defer to per-sprint scheduling. The Tier 1
  closeout is enough to call the original sweep "fully verified
  except for the explicitly future-sprint items".
- If Tier 2 is in: pick which subset and in what order; each is a
  multi-week project.

Everything else in Tier 1 is mechanical or has a recommended approach;
no further decisions needed before execution.
