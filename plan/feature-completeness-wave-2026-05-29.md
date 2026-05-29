# Feature-Completeness Wave — Implementation Plan (2026-05-29)

> **Goal:** bring every major rendering algorithm / decision-of-consequence to a genuinely
> complete, wired, *consumed* state — a coherent feature-complete stopping point. This is
> **extension of unfinished portions**, not remediation. Branch: `feat/feature-completeness-wave`.

Authoritative live state is always `git log` + `CLAUDE.md` + `HARDWARE-VALIDATION-NEEDS.md`. This
plan was produced from a deep, code-verified audit (docs were found stale and corrected first).

---

## 0. Decisions locked (user sign-off)

| # | Decision | Choice |
|---|---|---|
| Scope | Workstreams in the wave | #1 cleanup ✅, #2 smooth normals, #3 many-light, #4 volumetric SSS, #5 diff-RT, #6 NRC encode-backward |
| Diff-RT backend | Adjoint integrator | **In-browser WGSL adjoint** (not export to PyTorch/dr.jit) |
| #2 buffer budget | How to free a storage slot | **Reclaim a slot** — move `bvh_beer` to a texture (textures don't count against `maxStorageBuffersPerShaderStage`), freeing a storage slot for `bvh_normal`; smooth normals work on all adapters |
| #3 scope | Which backends | **pt-webgpu only** — pt-webgl's fork already does power-weighted selection (`randomLightSample`, `Light.power`) |
| #4 BDPT interaction | First-pass scope | **Gate the volumetric walk off when BDPT is enabled** (energy-conservation mismatch otherwise); fix caustic-chain extinction; document |
| #5 phases | This wave | **Phase 0 (finite-difference) + Phase 1 (path-replay BSDF adjoint, albedo+roughness)**; Phases 2–5 out of scope |

---

## 1. Verified starting state (the audit)

- **#1 done** (committed `3a2e12e`): corrected stale claims. NRC forward query + record + per-frame MLP `trainStep` ARE wired (comments said "next phase" — false). GPU skin computes inverse-transpose normals — but they're dropped (`applyGpuSkinnedRefit` ignores them) and the whole real-time primary path shades **faceted geometric** normals.
- **#2:** primary passes (`shade`/`ris`/`risGi`/`risGiNrc`) use geometric `result.normal`; `IntersectionResult` already carries `.barycoord` + `.indices`. DDGI precedent does the smooth blend (`probeUpdateRays.wgsl.ts:443-454`). Blocker: scene group at the 16-storage-buffer floor (`bindGroupLayouts.ts:86-88`).
- **#3:** pt-webgpu picks one emitter uniformly (`kernel.wgsl.ts:275`) + `*lightCount` compensation (`:429`). pt-webgl already power-weighted. Tested `sampleLightTree` WGSL exists in `shared-samplers` + walkaround.
- **#4:** `kernel.wgsl.ts:234-258` is Beer-Lambert + a `0.02` fudge — no distance sampling, no phase function, no multiple scattering, no MIS.
- **#5:** no implementation (only `plan/differentiable-rt.md`). Optimizer/grad substrate exists (NRC fused-MLP Adam). Readback (`rgba16fReadback.ts`), incremental material/emitter updates, deterministic RNG all exist.
- **#6:** NRC trains the MLP but the hash-grid **tables are frozen at random init** (`nrcSubsystem.ts:202` write-once); `nrcEncodeBackwardWgsl` emitted but never dispatched; the MLP backward stops at the input layer (`fusedMlp.wgsl.ts:333` `l>0` guard) so `dL/dX` isn't produced.

---

## 2. Per-workstream plans

### WS1 — Smooth normals (walkaround-hybrid)

**Approach.** Move `bvh_beer` (per-tri, read-only `u32`) to a `texture_2d<u32>` (frees a storage slot). Add a per-vertex `bvh_normal: array<vec4f>` (from the already-built `shared.normals`) to the scene bind group. Barycentric-blend the shading normal in the 4 primary passes; **keep the geometric normal for ray offset / backface / `side` flip** (DDGI precedent). Redirect the GPU-skin normal write into the shared `bvh_normal` at `baseVertex+vi`. Merged-BVH normals are world-space (no transform); TLAS-mode smooth normals deferred (documented) since the per-instance transform isn't carried out of `traceTlasFirstHit`.

**Files.** `restir/bvhCompute.ts` (expose `bvhNormals`), `pipeline/BvhBufferHost.ts` (+`_bvhNormalBuffer`, `_bvhBeerTexture`, re-upload on refit/rebuild), `pipeline/bindGroupDescriptors.ts` + `bindGroupBuilders.ts` + `bindGroupLayouts.ts` (swap beer→texture, add normal), `shaders/{shade,ris,risGi,risGiNrc}.wgsl.ts`, `skin/GpuSkinningSubsystem.ts` + `skin/gpuSkinBvh.wgsl.ts` (write `baseVertex+vi`), `pipeline/__tests__/bindGroupDescriptorParity.test.ts`.

**Tests.** Barycentric-blend CPU oracle (`normalize(w·n0+u·n1+v·n2)`, unit-length over random triples); codegen pins (geometric normal used for offset; `bvh_beer` storage-ref gone); descriptor-parity passes; scene-group storage count assertion; GPU-skin normal == CPU `mat3InverseTranspose` result; **V21** reference A/B (low-poly curved mesh, faceted→smooth, no acne).

**DoD.** typecheck+vitest green; parity test green; all 4 passes reference `bvhNormal`; V21 captured on real GPU (tracked).

### WS2 — Many-light importance sampling (pt-webgpu)

**Approach.** Import `sampleLightTree` + `LIGHT_TREE_WGSL` from `@vitrum/shared-samplers` (do NOT copy from walkaround). Add per-light power to `emitterPacking.ts` (`luminance·area` for area lights, `luminance` for delta). Build/pack via `buildLightTree`/`packLightTreeForGPU`. Add a **dedicated group-3 storage buffer** for the tree (full-tier only — lite keeps the uniform pick). In `kernel.wgsl.ts`, replace the uniform pick (`:263-275`) and the `*f32(lightCount)` compensation (`:429`) with `sampleLightTree` + `/lt.pdf`. Fallback path when `nodeCount==0`.

**Files.** `scene/emitterPacking.ts`, `scene/uploadSceneBuffers.ts`, `gpuResources.ts` (group-3 BGL, full-tier-gated), `scene/frameParamsLayout.ts` (`lightTreeEnabled`/`nodeCount`), `wgsl/pathTrace/material.wgsl.ts` (group-3 binding), `wgsl/pathTrace/kernel.wgsl.ts`, `wgsl/pathTraceBruteforce.wgsl.ts` (compose), `__tests__/multiLightMis.test.ts`.

**Tests.** Per-type power formula (CPU); partition-of-unity (`Σpdf=1`); **unbiasedness MC convergence** (power-weighted == uniform in the mean) + variance-reduction (`var_pw < 0.8·var_uniform` for ≥5× power spread); power-field packing round-trip; consumer grep (no uniform-pick left in NEE); **V22** reference A/B (8 lights, 10:1 spread, variance reduction at equal mean).

**DoD.** typecheck+vitest green; lite tier untouched (no group-3); V22 tracked.

### WS3 — NRC hash-grid encode-backward (walkaround-hybrid)

**Approach.** Extend the MLP backward to emit layer-0 `dL/dX` (drop the `l>0` guard for input-delta; add a `gradInputFx` atomic buffer + finalize). Store raw query `pos` per record (`_recordStride += 3`; append in `nrcWriteRecord`). New encode-backward `@compute` kernel (inline the 8-corner trilinear scatter — WGSL forbids passing module-scope storage as a `ptr` arg) that recomputes `nrm` from `pos`, reads `dL/dX`, scatters into `_gradTablesFx`. Table Adam step (reuse `ADAM_WGSL`; separate lower LR per Instant-NGP `lr_embed≈0.1` vs `lr_mlp≈0.01`). Wire into `trainFromRecords` after the MLP step.

**Files.** `neural/nrc/wgsl/fusedMlp.wgsl.ts` (`:333`), `neural/nrc/fusedMlpTrainer.ts` (+`gradInputFx/F`, `readInputGrads`), `neural/nrc/nrcSubsystem.ts` (+`_gradTablesFx/F`, `_mTables/_vTables`, stride, encode-backward pipeline + bind group, `tableAdamStep`, `trainFromRecords`), `neural/nrc/wgsl/nrcQuery.wgsl.ts` (record append), new `neural/nrc/wgsl/nrcEncodeBackward.wgsl.ts` (standalone entry), `neural/nrc/__tests__/nrcGateBitIdentity.test.ts`.

**Tests.** `dL/dX` analytic == finite-difference (≤1e-4, extends the existing `nrcEncoding.test.ts` FD pattern); chained encode→forward→backward table grads == FD; **tables change after N steps** (liveness — guards the silent no-write failure); Adam step formula; **gate-OFF bit-identity preserved** (UBO bytes unchanged; record stride is data-only); lavapipe FD smoke (extend `fusedMlpHarness.ts`). NRC is BIASED → no mean-equality assertion. **V20** (extended): self-training stability, no divergent loss.

**DoD.** typecheck+vitest green; `nrcStructuralGate` + `nrcGateBitIdentity` green; tables provably learn; V20 tracked.

### WS4 — Volumetric SSS (pt-webgpu)

**Approach.** Add `inMedium`/`mediumSigmaT`/`mediumSigmaS`/`mediumG` path state; add `enteredMedium`/`exitedMedium` to `BounceSample` (set on the refraction branch). Replace the Beer-Lambert+fudge block with a homogeneous random walk: free-flight distance `t=-ln(1-ξ)/σ_t`; if `t<hit.dist` scatter (throughput `×σ_s/σ_t`), HG phase sampling, in-medium NEE with HG↔light **MIS**; else attenuate `exp(-σ_t·dist)` and fall to the surface BSDF. Hero-wavelength σ_t in spectral mode. **Gate the walk off when `params.bdptEnabled`** (compile-time / structural). Add medium extinction to the caustic specular chain. Derive `σ_a_rgb` from `attenuationColor`/`attenuationDistance` in `materialPacking.ts` (no new `@vitrum/core` field required). Mirror in `kernelLite`.

**Files.** `wgsl/pathTrace/bsdf.wgsl.ts` (`BounceSample`, refraction branch), `wgsl/pathTrace/kernel.wgsl.ts` (`:140-148`, `:234-258`, `:504-507`), `wgsl/pathTrace/kernelLite.wgsl.ts` (`:211-234`), `wgsl/pathTrace/caustic.wgsl.ts` (chain extinction), `wgsl/pathTrace/material.wgsl.ts` (`decodeMaterial` σ_t), `scene/materialPacking.ts` (σ_a derive; bump stride if needed — keep TS/WGSL constants in sync), new `__tests__/volumetricSss.test.ts`.

**Tests.** HG normalizes to 1 over the sphere (MC); HG sampler matches its pdf (IS weight ≈1); single-scatter homogeneous slab == analytic transmittance/albedo; Beer's-law anchor (no-scatter, high precision); MIS partition-of-unity; **structural gate** (SSS WGSL symbols absent when BDPT on; UBO/layout byte-stable); free-flight CDF inversion; **V23** reference A/B (translucent slab, exit-face chrominance within 5%, BDPT-ON path unchanged).

**DoD.** typecheck+vitest green; `frameParamsLayout`/`sharedPipelineLayout` green; BDPT-on path provably unchanged; V23 tracked.

### WS5 — Differentiable RT (core + pt-webgpu)

**Approach.** New `@vitrum/core` `inverse.ts`: `InverseParam`/`InverseSessionOptions`/`InverseStepResult`/`InverseSession` + optional `createInverseSession?` on `Engine`; forward through the `createEngine` proxy conditionally. **Phase 0:** finite-difference loop in pt-webgpu — perturb a material/emitter param (`updatePrimitive`/`updateEmitter`), `reset()`, render N samples, read back via `rgba16fReadback`, L2 loss, CPU Adam on the tiny param vector. **Phase 1:** new `pathTraceAdjoint.wgsl` — path-replay (fixed `frameSeed`) + analytic BSDF partials `dL/d·albedo`, `dL/d·roughness` (sampled `wi` frozen, so no diff through sampling); reuse the NRC Adam (extract `recordAdam`+`ADAM_WGSL` into a small `GpuAdam` helper). Add optional `frameSeed`/`deterministic` to `FrameInput` (default = current behavior). **Validate the RNG-replay path against a CPU reference first** (riskiest part).

**Files.** new `core/src/inverse.ts`, `core/src/index.ts`, `core/src/engine/index.ts`, `core/src/frame.ts` (optional `frameSeed`), `engine/src/createEngine.ts` (proxy), `pt-webgpu/src/index.ts` (`createInverseSession`), `pt-webgpu/src/scene/frameParamsLayout.ts` + `wgsl/pathTrace/kernel.wgsl.ts` (seed RNG), new `pt-webgpu/src/wgsl/pathTrace/pathTraceAdjoint.wgsl.ts` (Phase 1), new `__tests__/inverseSession.test.ts`.

**Tests.** **BSDF adjoint analytic == FD of `evaluateBrdf`** (≤1e-4 — the critical Phase-1 gate; do not ship without it); Phase-0 loss decreases + gradient sign correct on a 1-sphere-1-light fit; analytic Lambert `dL/dρ = L_i·cosθ/π` cross-check; RNG-replay determinism (same seed → same hit; different param → different contribution); `InverseSession` contract shape + idempotent dispose; **V24** GPU A/B (Phase-0 loss descent; Phase-1 adjoint within 5× of GPU FD; dispose clean).

**DoD.** typecheck+vitest green; new core types exported; `createEngine` proxy forwards; default frame path unchanged; V24 tracked. Phases 2–5 explicitly deferred (roadmap).

---

## 3. Execution order, waves & audit checkpoints

```
Wave A (parallel, walkaround-hybrid, no contention):  WS1 ‖ WS3
   └─ AUDIT CHECKPOINT A
Wave B (sequential within pt-webgpu — all touch kernel.wgsl):  WS2 → WS4 → WS5
   └─ AUDIT CHECKPOINT B
Wave C (validation):  capture V21–V24 reference renders; record V-items
```

- **WS1 ‖ WS3** are entirely walkaround-hybrid with zero file overlap.
- **WS2 → WS4 → WS5** share `kernel.wgsl.ts`/`material.wgsl.ts`/`frameParamsLayout.ts`; do them in series to avoid merge churn (WS2 purely additive light-pick; WS4 edits the Beer-Lambert block; WS5 adds 3-line seed init + new files).
- Each WS lands as its own commit (typecheck + vitest green; TDD tests written first per §2).

**Audit checkpoints (non-negotiable, per plan-implementation discipline).** After **each wave**, run `/audit` on the changed files. Fix any structural debt (God files, mixed concerns, export sprawl, stale comments in touched files) **before** the next wave. Specifically re-verify: no computed-but-unconsumed buffers (grep WGSL consumers); MIS partition-of-unity unit tests present; compile-time (not runtime-UBO) gating for WS2/WS4/WS5 feature flags; bind-group parity intact.

---

## 4. Error handling & cross-cutting constraints

- **Capability gates**, NConsistent with NRC's pattern: WS2 group-3 and any feature requiring above-floor limits must fail early + legibly (clear throw), and gate **at compile-time WGSL composition** for structural layout changes — never an unconditional `@group` add (the GRIS black-frame lesson, `f8df9a4`).
- **Storage-texture / layout traps:** no `texture_storage_2d<...,read_write>` (not core WebGPU); use explicit shared `GPUPipelineLayout` for shared bind groups (not `layout:'auto'`).
- **Degradation:** lite tier keeps uniform light pick + no volumetric walk; non-deterministic frame path keeps existing RNG; BDPT-on keeps the old absorption path.
- **Contract:** `@vitrum/core` public types are fixed; host owns device/lifecycle/cadence; cite prior work in source comment + package README + `CREDITS.md` (Conty-Estevez/Kulla 2018, Shirley 1996, Müller 2021/2022 NRC + Instant-NGP, Henyey-Greenstein 1941, PBRT vol. transport, Vicini 2021 path-replay backprop, Nimier-David 2020 radiative backprop).
- **Validation reality:** GPU A/B is pending hardware. Unit-pin + structural-test everything possible; add V21–V24 to `HARDWARE-VALIDATION-NEEDS.md`; never claim radiometric correctness from lavapipe/SwiftShader or screenshots (telemetry only). NRC acceptance is perceptual/variance, not mean-equality (it is biased).
- **No** npm publish, remote push, or upstream PRs.

---

## 5. Definition of done (wave)

- [ ] All six workstreams landed on `feat/feature-completeness-wave`, each its own commit.
- [ ] `npm run typecheck` clean; `npm test` green across workspaces.
- [ ] New tests per §2 present and green (TDD tests written first where flagged).
- [ ] Bind-group parity + NRC gate bit-identity + structural gate tests green.
- [ ] No computed-but-unconsumed buffers (WGSL consumer grep clean for every new buffer/uniform).
- [ ] V21–V24 added to `HARDWARE-VALIDATION-NEEDS.md`; reference-render dirs created for GPU A/B.
- [ ] Stale comments in every touched file corrected in the same commit.
- [ ] `CLAUDE.md` / `plan/state-of-the-pipeline-2026-05-29.md` / `CREDITS.md` updated to reflect the new completed state.
- [ ] Audit checkpoints A & B run; structural debt cleared.

---

## 6. Risks (top)

1. **WS1 buffer budget** — the bvh_beer→texture swap is load-bearing; verify exact scene-group entry count against `bindGroupDescriptors.ts` + parity test before/after.
2. **WS5 RNG replay** — error-prone; validate against a CPU reference before trusting Phase-1 gradients; MC variance needs fixed seed + high SPP + tiny scene.
3. **WS4 ↔ BDPT/caustics** — energy non-conservation if only the eye path loses energy in the medium; the BDPT gate + caustic-chain extinction mitigate; full BDPT-over-volumes deferred.
4. **WS3 record stride** — re-read `nrcGateBitIdentity.test.ts` before changing stride; confirm OFF path writes no records.
5. **WS4 material stride bump** — keep TS `MATERIAL_VEC4_STRIDE` and the WGSL constant in lockstep; only pt-webgpu's WGSL is affected.
