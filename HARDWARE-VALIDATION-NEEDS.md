# Hardware (GPU) Validation Needs — vitrum

**Audience:** an agent/engineer on a machine with a **real hardware WebGPU adapter** (Chrome 120+ on a discrete GPU, or any host where `chrome://gpu` shows hardware-backed WebGPU). The dev box this was authored on is **WSL2 with SwiftShader only**, which cannot validate any of the items below.

This file lists every change from the 2026-05-28 complexity-remediation sweep (and its follow-ups) whose **correctness is already pinned by unit tests + code audits** but whose **visual / GPU-runtime confirmation could not be run in WSL**. Each item has: what changed, why it needs a GPU, the test scene, the expected delta, acceptance criteria, and the exact commands.

---

## 0. Why WSL can't do this (verified)
`tools/benchmark-runner` adapter probe (`VITRUM_PROBE_START_SERVER=1 npm run benchmark:pt-webgpu-adapter-probe`) reports, for BOTH the "hardware" and "swiftshader" Playwright profiles in WSL2:
```
maxStorageBuffersPerShaderStage: 10   maxStorageTexturesPerShaderStage: 4
ptWebgpuFullTier: false   ptWebgpuLiteTier: true   hybridCanRun: false
```
Consequences:
- **walkaround-hybrid backend won't initialize** (`hybridCanRun: false` — needs ≥8 storage textures; SwiftShader gives 4). → all DDGI / ReSTIR / shade / stained-glass validation is blocked.
- **pt-webgpu runs lite tier only** (full tier needs ≥23 storage buffers). → BDPT light-subpath + full-tier intersection paths can't run.
- The README warns SwiftShader renders are "slow and likely incorrect" for these scenarios.

**First step on a real GPU:** run the probe and confirm `ptWebgpuFullTier: true` + `hybridCanRun: true` before trusting any capture.

## 1. Harness reference
- Capture adapter: `tools/benchmark-runner/capture-adapter-playwright.mjs` (writes a PNG to `VITRUM_OUTPUT_PNG`).
- Diff/gate runner: `npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner` (compares candidate vs baseline, computes PSNR).
- Image diff helper: `tools/reference-renders/diff-baselines.mjs`.
- WSL→Windows-Chrome bridges (if running WSL vite + Windows GPU Chrome): the `tools/benchmark-runner/run-*-win-chrome.mjs` variants.
- Acceptance-metric JSON generation from PNG pairs: `npm run benchmark:acceptance-metrics` (see `tools/benchmark-runner/README.md` §"Acceptance metrics artifacts").

### Pre/post A/B baseline
Everything in this sweep is now on `main`. The **pre-sweep baseline commit is `bbd32c8`** ("docs(context-sweep): purge stale comments…", the commit `main` sat at before the sweep). The general A/B procedure for any radiometric item below:
```bash
# baseline (pre-sweep)
git stash; git checkout bbd32c8
VITRUM_GPU_CAPTURE=1 VITRUM_ALLOW_BASELINE_GEN=1 \
  VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
  npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
# (move the generated PNGs to *.pre-sweep.png per the benchmark-runner README §"Sweep verification capture")
# candidate (post-sweep)
git checkout main
VITRUM_GPU_CAPTURE=1 \
  VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
  npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```
Acceptance (per `tools/benchmark-runner/README.md` §"Sweep verification capture"): the visual change must be in the direction the math predicts; no NEW artifacts vs the pre-sweep image; a PSNR regression is acceptable only if visually justified (physically-correct darkening/colour-shift).

---

## 2. Items needing GPU validation

> Status legend: **[done-needs-render]** = code shipped + unit-pinned, only visual confirm outstanding.

### V1 — A3: BDPT bounce-0 tangent fix  [done-needs-render]
- **What changed:** `wi = t*x + b*y + n*z` (was `b*x`; `y` was dead) in `pt-webgl/src/bdpt/bdptSceneEmittersCpu.ts` + `pt-webgpu/src/bdpt/bdptEmitterPickCpu.ts`. The CPU oracle now matches the GPU `cosineHemisphereSample` shader.
- **Why GPU:** the BDPT light-subpath integrator is full-tier pt-webgpu / pt-webgl; lite tier omits it.
- **Test scene / plan:** a BDPT-enabled scene with an area or mesh light (e.g. a Cornell box with BDPT on) at fixed seed, both backends. Capture pre-sweep (`bbd32c8`) vs `main`.
- **Expected delta:** corrected cosine-weighted light-subpath sampling → slightly different (correct) indirect distribution; **no fireflies/bias** introduced. Converged images should be very close (the bug biased low-sample distributions).
- **Acceptance:** converged BDPT render is artifact-free and matches the analytic expectation; A/B shows no systematic darkening/brightening beyond MC noise at high SPP.

### V2 — T16: DDGI emitter radiometry (chroma + area)  [done-needs-render]
- **What changed:** DDGI light collection now maps core emitters directly (`coreEmittersToDDGILights.ts`) instead of round-tripping through THREE — recovers per-emitter **chroma** (a red rect-area light was bleeding **white**) and uses the correct emissive area `4·|u×v|` (was `width·height`). Wired on both the init path and the incremental `updateEmitter` path.
- **Why GPU:** DDGI runs in walkaround-hybrid (`hybridCanRun: false` in WSL).
- **Test scene / plan:** a colour-bleed scene — a saturated (e.g. red) rect-area light near a white wall, hybrid backend, DDGI on. Capture `bbd32c8` vs `main`. Add a sheared-axis rect-area variant (non-orthogonal u/v) to exercise the area-metric fix.
- **Expected delta:** indirect/probe lighting near the coloured light now carries the **light's colour** (red bleed, not white); sheared-rect emitters change brightness by the area ratio (`|u×v|` vs `|u|·|v|`).
- **Acceptance:** the colour bleed visibly matches the emitter colour; brightness change matches the area-ratio math; no probe artifacts.

### V3 — T5: stained-glass shade module (opt-in, default OFF)  [done-needs-render]
- **What changed:** `lo_sun_caustic` + `lo_sky_aperture` extracted from `shade.wgsl` into an opt-in `stainedGlassShade.wgsl` module gated by a `stainedGlassFlags` UBO field; default OFF. `HybridEngineOptions.stainedGlass?: { sunCaustic?, skyAperture? }`.
- **Why GPU:** walkaround-hybrid shade pass.
- **Test scene / plan:** (a) a Cornell-stained-glass scene with `stainedGlass: { sunCaustic: true, skyAperture: true }` — capture `main` vs `bbd32c8`; (b) a generic (non-stained-glass) scene with the flags OFF (default).
- **Expected delta:** (a) with flags ON, output must be **bit-for-bit / visually identical** to pre-sweep (the flag-ON WGSL is already proven byte-identical to the old inline shader — this is a confirmation, the least-risky item); (b) with flags OFF, the sun-caustic + sky-aperture terms must drop to **exactly zero** vs pre-sweep (the intended fix: generic scenes no longer get stained-glass physics).
- **Acceptance:** (a) PSNR ≈ ∞ / pixel-identical; (b) the two terms are gone, no other change.

### V4 — directional → DDGI sun  [implemented this session — needs-render]
- **What changed:** a scene `directional` emitter now drives the DDGI sun with its **real direction** (was hardcoded `(0,-1,0)`) and is single-counted (resolved the setSunIntensityMultiplier-vs-fixture double-count); `directional` re-declared in walkaround's `supportedEmitterKinds` + ledger.
- **Why GPU:** DDGI sun lighting, walkaround-hybrid.
- **Test scene / plan:** a scene with a non-downward `directional` emitter (e.g. dir `(1,-1,0)` normalized), hybrid backend. Capture `main` vs `bbd32c8`.
- **Expected delta:** DDGI probe/indirect sun lighting now comes from the **correct direction** (previously lit as if from straight down); intensity unchanged (single-counted, not doubled).
- **Acceptance:** sun-lit surfaces face the emitter's actual direction; no doubling/over-brightening; a `(0,-1,0)` directional is unchanged vs pre-sweep.

### V5 — DDGI IRR border-texel fill  [implemented this session — needs-render]
- **What changed:** the irradiance border pass now fills all border texels of each 10×10 probe-atlas cell (previously left the 4 bottom-edge texels `lx∈{6,7,8,9}, ly=9` unfilled).
- **Why GPU:** DDGI probe atlas, walkaround-hybrid.
- **Test scene / plan:** any DDGI scene; inspect probe-atlas edges / look for probe-seam artifacts at cell boundaries. Capture `main` vs `bbd32c8`.
- **Expected delta:** probe octahedral border bilinear-filtering is now correct at the previously-unfilled edge → subtle removal of edge-seam artifacts in indirect lighting.
- **Acceptance:** no probe-seam artifacts; indirect lighting at probe-cell boundaries is continuous.

### V6 — pt-webgl instanced-mesh support  [code landed + unit-pinned — OUTSTANDING: needs GPU render A/B]
- **What changed:** pt-webgl now renders `instanced-mesh` primitives. The shared `vitrumSceneToThree` still builds ONE `THREE.InstancedMesh` per primitive (unchanged — walkaround-hybrid's TLAS path needs the single InstancedMesh + per-instance transforms). pt-webgl's `setScene` now calls `expandInstancedMeshesInScene(threeScene)` (new file `packages/pt-webgl/src/expandInstancedMeshes.ts`) AFTER `vitrumSceneToThree` and BEFORE the fork's `WebGLPathTracer.setScene` → BVH/geometry generator. Each `THREE.InstancedMesh` is replaced by N standalone `THREE.Mesh` objects, each baking `parent.matrixWorld · instanceMatrix[i]` into its `matrixWorld` (with `matrixAutoUpdate = false`), sharing the InstancedMesh's single geometry + material. This is required because the absorbed three-gpu-pathtracer fork's `convertToStaticGeometry.js` bakes only `mesh.matrixWorld` (line 283) and never reads `instanceMatrix`, so an unexpanded InstancedMesh would collapse all N instances to one copy at the local origin. `instanced-mesh` re-declared in pt-webgl's `supportedPrimitiveKinds` (capability getter) + the core `promiseLedger.ts` pt-webgl row, so `partitionSceneBySupport` now KEEPS instanced-mesh primitives (flows them through to the converter) instead of warn-skipping them.
- **Code/test status (verified, non-GPU):** `pt-webgl` + `core` `tsc --noEmit` clean; full-workspace typecheck clean; new `expandInstancedMeshes.test.ts` (7 tests) pins N→N expansion at correct per-instance world transforms (incl. scaled + composed-parent-transform cases), shared geometry/material, no-op on non-instanced scenes, and the capability/ledger re-declaration; `capabilitiesPartition.test.ts` updated to assert instanced-mesh now flows through (not warn-skipped); full `npm test` green (all workspaces).
- **Why GPU:** the actual render is the pt-webgl path tracer; WSL SwiftShader can't produce a trustworthy capture.
- **Test scene / plan:** a scene with an `instanced-mesh` of N≥3 instances at distinct transforms, pt-webgl backend. Capture `main`; compare against a hand-authored equivalent of N separate meshes (the reference). Add a scaled-instance variant to exercise the non-translation matrix bake.
- **Expected delta:** all N instances render at their correct per-instance world transforms (was: one copy at the origin, or — before this change — warn-skipped entirely). The render must match the N-separate-meshes reference.
- **Acceptance:** instance count + transforms match the N-separate-meshes reference; no missing / origin-collapsed instances; scaled instances render at the right size.

### V7 — Möller-Trumbore unify (pt-webgpu)  [code landed — OUTSTANDING: needs GPU render A/B]
- **What changed:** pt-webgpu's triangle-intersection is now single-sourced on the canonical shared-bvh Möller-Trumbore math. The divergent local copy in `packages/pt-webgpu/src/wgsl/common.wgsl.ts` (`h = cross(dir, e2); det = dot(e1, h)`, strict `u<0||u>1` / `v<0||u+v>1` barycentric tests) is gone; both pt-webgpu's f32-returning `intersectTriangle` wrapper AND the canonical struct-returning `intersectTriangle` in `BVH_INTERSECT_WGSL` now delegate to the new shared `mollerTrumboreCore` fragment (`MOLLER_TRUMBORE_WGSL` in `packages/shared-bvh/src/wgsl/bvhIntersect.wgsl.ts`), which uses `n = cross(e1,e2); det = -dot(dir,n)` with **triEps-tolerant signed barycentric tests** (`u/v/w < -triEps`, triEps = `params.triIntersectEpsilon` = 1e-5). The `__tests__/cpuTracer.ts` oracle was updated to mirror the same canonical core. This is a deliberate **edge-case intersection numerics** change (NOT behaviour-preserving at triangle edges): the hit distance `t` for interior hits is algebraically unchanged, but hits grazing a shared edge by less than triEps are now **accepted** instead of rejected.
- **Code/test status (verified, non-GPU):** workspace `tsc --noEmit` clean; shared-bvh (74) + pt-webgpu (138 + mcConvergence oracle) vitest green; full `npm test` green (608 pt-webgpu pkg tests etc.). Composed pt-webgpu kernel verified to contain exactly one `mollerTrumboreCore` / `TriHit` / `intersectTriangle` / `safeInvDir` (no WGSL duplicate definition) and zero `IntersectionResult` (canonical struct stays out of pt-webgpu, avoiding the struct collision).
- **Why GPU:** full-tier pt-webgpu path tracer — the edge/tolerance delta only manifests at runtime on shared triangle edges / t-junctions.
- **Test scene / plan:** a mesh-heavy scene with grazing/edge-on triangles (where barycentric tolerance matters), pt-webgpu full tier. Capture **`bbd32c8` (pre-change baseline, has the old divergent formula)** vs the post-change build at high SPP.
- **Expected delta:** near-identical except at triangle edges / shared edges (t-junctions), where the canonical formula's tolerance differs — should be **equal or fewer** edge-crack artifacts, not more.
- **Acceptance:** no new edge cracks / light leaks at triangle boundaries; high-SPP A/B within MC noise except at intended edge cases.

### V8 — T9-stepC per-pass WGSL narrowing  [implemented this session — needs shader-compile validation]
- **What changed:** each walkaround pass's `requires` was narrowed from the full `common` aggregate to its minimal WGSL module set.
- **Why GPU:** a wrongly-narrowed pass (missing a needed declaration) only fails at `createShaderModule` — typecheck and the byte-composition test won't catch a genuinely-missing symbol that the pass references.
- **Test plan:** run the hybrid pipeline on a real GPU and confirm **every pass's shader compiles** (no `createShaderModule` errors) — i.e. the existing GPU smoke / hybrid init path. If any pass fails to compile, its `requires` is missing a module; widen it.
- **Acceptance:** full hybrid pipeline initializes + renders a frame on hardware WebGPU with zero shader-compile errors across all passes.

### V9 — real Jakob-Hanika RGB→spectrum upsampling (pt-webgl spectral)  [implemented — needs render]
- **What changed:** `shared-samplers/src/jakobHanika.ts` replaced the placeholder with the genuine Gauss-Newton sigmoid-coefficient solve (round-trip CPU-pinned to ~1e-7 interior / <5e-3 saturated); pt-webgl's `forkUniformBridge` uploads real `u_jakobCoeffs` from `vitrum.ptWebgl.spectralAlbedo` when spectral rendering is on. Capability tag `spectral-jakob-hanika-placeholder` → `spectral-jakob-hanika`. **UPDATE (this session): the fork integrator now actually CONSUMES the coefficients** — before, `evalSpectrum(u_jakobCoeffs,λ)` had ZERO call sites (uploaded-but-dead). A gated `mediumAlbedoHero()` now drives the volume single-scatter (`u_scatterAlbedo`) + SSS single-scatter (`u_sssAlbedo`) albedo through `evalSpectrumAtHero` when spectral upsampling is active (`uSpectralRendering && u_jakobCoeffs != 0`); off-gate it returns the legacy `heroScalarFromRgb` smoothstep, bit-identical. So the A/B below is now actually meaningful (previously the coeffs had no render effect).
- **Why GPU:** the fork's `evalSpectrum(u_jakobCoeffs, λ)` reflectance weighting only manifests in a live spectral render (hero-wavelength accumulation), not exercisable under SwiftShader.
- **Test scene:** a dispersive/transmissive glass scene, `spectralRendering: true` + a saturated `spectralAlbedo` (e.g. `[0.5,0.1,0.7]`), fixed seed, pt-webgl. A/B `bbd32c8` vs `main`.
- **Expected delta:** the medium picks up the correct chromatic reflectance per hero wavelength (vs the prior flat S≡½); no fireflies; converged image consistent with the CPU round-trip (<5e-3).
- **Acceptance:** chroma shift in the math-predicted direction; no new artifacts.
- **NOTE (capture-harness gap, from `plan/fidelity-promotion-playbook.md`):** pt-webgl is WebGL2 — the lavapipe WebGPU device can't validate it; this row needs a real-browser GL capture. And the lavapipe *PNG render-capture* adapter for WebGPU rows is not yet wired (only probe + compute-smoke exist) — building it is a prerequisite for hands-free WSL fidelity captures.

### V10 — BMFR denoiser (Koskela 2019)  [implemented this session — OUTSTANDING: needs real-GPU denoise-quality A/B]
- **What changed:** the previously-unimplemented `denoiser: 'bmfr'` mode (the `@vitrum/core` `EngineOptions.denoiser` union always advertised it, but `walkaround-hybrid` THREW on it) is now a **real BMFR denoiser** — Koskela, Immonen, Mäkitalo, Foi, Viitanen, Jääskeläinen, Kultala, Takala, "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing Reconstruction," ACM TOG 38(5), 2019. New `@vitrum/shared-denoisers` modules: `wgsl/bmfr.wgsl.ts` (per-32×32-block compute kernel), `bmfrRegression.ts` (CPU regression core / oracle), `bmfrBindings.ts` (UBO), `bmfrConstants.ts`, `bmfrWebGPU.ts` (one-shot host pipeline). New `walkaround-hybrid` registry entry `pipeline/denoisers/bmfr.ts`; the HybridEngine `'bmfr'` throw is removed. **Design:** 32×32 block, feature set `[1, p.xyz, n.xyz, p².xyz]` (10 cols), Householder-QR solve of the normal equations per color channel, reconstruct `color = T·α`, then temporal EMA (α=0.2, reset on camera motion). The realtime entry uses a **screen-space position proxy** (`(pixelX, pixelY, depth)` from `gNormalDepth.w`) so no dedicated world-position G-buffer is needed; the one-shot host path uses a real world-position buffer.
- **Code/test status (verified, non-GPU + lavapipe):** shared-denoisers + walkaround-hybrid `tsc --noEmit` clean; new `bmfrRegression.test.ts` (7 CPU tests: Householder-QR solve, exact-linear recovery, noisy-constant → mean denoise, gradient preservation, rank-deficient stability); `denoiserRegistry.test.ts` + `passLayout.test.ts` updated for the 7th denoiser; full shared-denoisers (146) + walkaround-hybrid (601) vitest green. **Lavapipe compute smoke (Deno + LLVMpipe, headless WSL): the real `BMFR_WGSL` kernel compiles + dispatches and denoises a noisy 1-spp constant block over a smooth surface with 99.66% variance reduction (outVar 0.00018 vs inputVar 0.05302), output finite, block mean 0.598 vs true 0.600.** This proves kernel correctness (incl. a real UBO-size-mismatch bug caught + fixed during validation) but NOT real-pipeline visual quality — lavapipe is a CPU rasteriser.
- **Why GPU:** BMFR runs inside walkaround-hybrid (`hybridCanRun: false` under WSL SwiftShader); the denoise QUALITY (firefly suppression, edge/gradient preservation, temporal ghosting under motion, block-seam visibility) only manifests on a live hybrid render at real SPP.
- **Test scene / plan:** a noisy hybrid scene (e.g. Cornell with low SPP / a moving camera), hybrid backend, `denoiser: 'bmfr'`. A/B against `denoiser: 'svgf-real'` and `'atrous-variance'` on the SAME scene/seed. Also capture a static-camera converged frame and a camera-in-motion sequence (to check ghosting + the motion-reset path).
- **Expected delta:** vs the raw (`'none'`) signal — substantial noise reduction with **preserved geometric gradients** (BMFR fits the spatial features, so it is gradient-preserving, not a box blur) and **crisp material edges** (block-local fit + screen-space position discontinuities at depth edges). vs `svgf-real` — comparable-or-better spatial smoothness on smooth surfaces; watch for block-boundary seams (32×32 grid) and temporal ghosting on motion (mitigated by the camera-move history reset).
- **Acceptance:** (a) clear noise reduction vs `'none'` with no new fireflies; (b) no hard 32×32 block seams in the final composite; (c) gradients + material edges preserved (not over-blurred); (d) under camera motion, no persistent ghost trails (history resets on `isMoving`); (e) a static converged frame is artifact-free. If block seams ARE visible, enable the half-block-offset overlap (blockStride < blockSize, the `blockStride` UBO field) and/or widen the temporal window.

### V11 — GPU-skin bindMatrix fallback  [implemented this session — needs functional GPU check]
- **What changed:** the GPU skinning path (`GpuSkinningSubsystem`) now falls back to the CPU `solveSkin` for a skinned mesh with a **non-identity `bindMatrix`** and no morphs (it was taking the GPU path, whose kernel uses `combineSkinMatrices(bones, boneInverses)` WITHOUT the bind wrapping the CPU solver applies → wrong positions + normals). Identity/absent bind still uses the GPU fast path.
- **Why GPU:** skinning runs in walkaround GPU compute.
- **Test scene / plan:** an animated `SkinnedMesh` authored with a non-identity `bindMatrix` (no morphs), hybrid backend; compare against the CPU `solveSkin` result.
- **Expected delta:** the mesh deforms correctly (matches CPU reference) — previously distorted under non-identity bind.
- **Acceptance:** non-identity-bind skinned mesh matches the CPU reference; an identity-bind mesh is unchanged (still GPU path).

### V12 — duplicate DDGI sun dedup  [implemented this session — needs-render]
- **What changed:** when a scene `directional` (→ DDGI sun) AND a host `opts.lights` `sun` are both present, only ONE sun now reaches DDGI (scene directional wins, host sun dropped + one-time warn). Applied on BOTH the init path (`HybridEngineLifecycle`) and the incremental re-sync path (`HybridEngine._syncDdgiLightsFromThreeRoot`).
- **Why GPU:** DDGI direct lighting, walkaround.
- **Test scene / plan:** a scene with a `directional` emitter AND a host `opts.lights` sun, hybrid backend. A/B `main` vs `bbd32c8`.
- **Expected delta:** sun lighting at the correct SINGLE intensity (was ~2× from double-injection).
- **Acceptance:** brightness matches a single sun; no doubling; scene-directional precedence honored.

### V13 — RC merged-instance refit + filter parity  [implemented this session — needs-render]
- **What changed:** RC merged-mode moving-instance refit is now wired into `propagateBvhToGiSubsystems` (`refitMergedInstance` — in-place positions + node-AABB refit, no teardown). Also `RCSubsystem.setScene` now builds with a permissive all-meshes filter (was PBR-only `DEFAULT_FILTER`) so RC's merged vertex layout matches ReSTIR's by construction (non-PBR meshes were diverging — caught the moment the refit was wired).
- **Why GPU:** RC merged-mode GI, walkaround (`rcEnabled`).
- **Test scene / plan:** RC merged mode (`rcEnabled`, merged BVH) with a moving instance + at least one non-PBR mesh (e.g. MeshBasic), hybrid backend. Move the instance; observe RC GI.
- **Expected delta:** after the instance moves, RC GI tracks the new geometry without a full rebuild; non-PBR meshes are included in RC's BVH (matching ReSTIR).
- **Acceptance:** GI stays correct after instance moves; no RC/ReSTIR geometry divergence; no teardown stutter.

### V14 — resolutionFactor composite upscale (C1)  [implemented this session — needs-render]
- **What changed:** the composite blit is now resolution-independent — the vertex shader emits a `[0,1]²` screen UV and the frag indexes `denoisedTex` via `uv × textureDimensions` (nearest). Previously it indexed the internal-res texture with raw swap-chain `fragCoord`, so `resolutionFactor < 1` rendered into the top-left and left the rest of the canvas BLACK.
- **Why GPU:** composite pass, walkaround.
- **Test scene / plan:** any hybrid scene at `resolutionFactor` 0.5 and 0.75 on a real GPU.
- **Expected delta:** the ENTIRE canvas is covered (no black border/region); nearest-neighbour upscale; `factor == 1` is bit-identical to before.
- **Acceptance:** corner-pixel telemetry confirms full coverage; upscale quality acceptable vs a `factor = 1.0` reference. Worked pixel: 1920×1080@0.5 → bottom-right maps to internal (959,539).

### V15 — DDGI cadence load-bearing + 2→32 preset spread (H1)  [implemented this session — needs-render]
- **What changed:** `ddgiUpdateDivisor` now drives the REAL probe-update round-robin stride (was a dead UBO field; real cadence was a hardcoded stride 8). Preset spread is now ultra=2 / high=4 / medium=8 / low=32. **IMPORTANT: omitting `qualityTier` applies the ultra preset, so the no-preset DEFAULT cadence is now stride 2 — 4× the old hardcoded stride-8.** Deliberate fidelity-forward change.
- **Why GPU:** DDGI probe update, walkaround.
- **Test scene / plan:** a step lighting change (`updateLighting` toggling the sun); capture DDGI convergence over N frames at `ddgiUpdateDivisor` = 2, 8, 32. Telemetry: `window.__DDGI__.activeCount` per frame + atlas-mean.
- **Expected delta:** GI converges proportionally faster at lower divisors (divisor 2 ≈ 4× the per-frame active-probe count of divisor 8 ≈ 8× of divisor 32); the new default (stride 2) shows visibly faster GI response than the old stride-8.
- **Acceptance:** GI response rate scales inversely with divisor; per-frame `activeCount` matches the stride; no probe artifacts/instability at stride 2; the default 8→2 change is faster without temporal-blend breakdown.

### V16 — RC ⊕ ReSTIR-GI per-pixel confidence MIS  [implemented this session — needs-render]
- **What changed:** the indirect RC/ReSTIR-GI blend (`shade.wgsl`) is now a per-pixel confidence-ratio balance heuristic — `c_restir = clamp(Meff/restirGiMClamp)`, `c_rc = rcWeight·(1−m)`, `w_rc = c_rc/(c_rc+c_restir)` — replacing the old fixed host scalar. Unbiased (convex blend of two estimators of the same integral).
- **Why GPU:** shade pass, walkaround (`rcEnabled`).
- **Test scene / plan:** `rcEnabled: true`, moderate `rcWeight` (≈0.5), a disocclusion-heavy scene; capture the first ~10 frames after a fast camera pan. A/B vs the pre-change fixed-scalar build.
- **Expected delta:** less indirect noise in freshly-disoccluded (low-M) regions (RC fills in); converged (high-M) regions match the old result closely (RC fades out).
- **Acceptance:** disoccluded regions visibly cleaner than the fixed-scalar blend; converged regions ≈ unchanged; `rcEnabled: false` is bit-identical to a pre-change build.

### V17 — PPG guided sampling (gi-ris consumes the learned dTree)  [implemented this session — needs-render]
- **What changed:** PPG was train-only (learned a dTree but gi-ris sampled plain cosine; the guide output had no reader). gi-ris now draws GI candidate directions from the learned dTree with a defensive, UNBIASED MIS mixture: with prob α from the dTree else cosine, and the RIS source pdf is `α·p_guide + (1−α)·p_cos` evaluated for the chosen direction. New `ppg/ppgPdf.wgsl.ts` (`ppgEvalPdf` mirrors CPU `dTreePdf`); the 3 PPG buffers packed into the hybridLayers bind group (group 3, bindings 6/7/8, respecting maxBindGroups=4); `ppgEnabled`/`ppgMixAlpha` UBO gate. ppg-OFF (α=0) is bit-identical to the pre-PPG cosine kernel (no extra RNG draw; literal `w = luminance(Lo)`).
- **Why GPU:** walkaround GI; PPG is opt-in (`ppgEnabled`, forbidden on lite tier).
- **Test scene / plan:** a guiding-friendly scene (cornell-box / indirect-dominated, small-aperture lighting), `ppgEnabled` on vs off, fixed seed. Capture per-pixel Welford indirect-channel variance (NOT screenshots) at matched frame counts.
- **Expected delta:** PPG-on shows LOWER indirect variance / faster convergence; the converged indirect-channel MEAN must MATCH PPG-off (unbiasedness — same fixed point). PPG-off telemetry must be bit-identical to a pre-change build.
- **Acceptance:** equal converged mean (unbiased) + measurable variance reduction with PPG on; PPG-off identical to pre-change.

### V18 — Full Veach §10.3 BDPT connection MIS (pt-webgpu + pt-webgl fork)  [implemented this session — OUTSTANDING: needs GPU shader-compile + variance A/B]
- **What changed:** the GPU connection-MIS weight was upgraded from the 2-strategy approximation to the full Veach §10.3 sweep over all flipped transfer indices. The WGSL/GLSL port mirrors `@vitrum/shared-samplers`' `buildBDPTStrategyPDFs_full` / `bdptConnectionMIS_full` oracle 1:1 (PBRT-v4 `MISWeight` recurrence; destination-cosine-only ConvertDensity). A new TS reimplementation `bdpt/bdptConnectionMisFull.ts` is asserted equal to the oracle to ≤1e-12 (pdf vector + per-strategy weight + partition-of-unity + pRef scale-invariance). The hardcoded `eyePdfFwd=1.0` is replaced by real `brdfDirectionalPdf` scatter densities threaded through a per-pixel eye-vertex scratch stack; the baked `*G` in the light-subpath shader is dropped (both lanes now store bare SA pdfs, Jacobian applied once at connection). A latent primary-hit double-count in pt-webgpu is fixed (now `bounce>0`, mirroring the fork's `!state.firstRay`).
- **Why GPU:** path-trace kernel (pt-webgpu) + `PhysicalPathTracingMaterial` (pt-webgl fork). Full-tier-only, opt-in via `params.bdptEnabled` / `FEATURE_BDPT`; BDPT-off shading math is bit-identical.
- **Three things needing real hardware:**
  1. **WGSL/GLSL shader-compile** — the §10.3 sweep is structurally reviewed but unverified by a real shader compiler here. Run the opt-in `wgslSmoke.gpu.test.ts` (SwiftShader/Dawn) + the fork's smoke on a GPU host.
  2. **Variance A/B** — render a caustic/glossy-bounce scene (BDPT's regime) with full §10.3 vs the old 2-strategy build at matched SPP. Telemetry: per-pixel Welford on the converged image.
  3. **Storage-buffer layout bump** — the eye-stack adds an unconditional `@group(2) @binding(6)` storage buffer, raising the kernel's per-stage storage-buffer count **23→24** *even when BDPT is off* (single pipeline, runtime-gated). Validated full-tier hardware (RTX 4090 / Chrome-class, where 23 already worked) reports buffer limits far above 24 so this is expected-immaterial — but confirm pipeline creation succeeds, and note a theoretical regression risk for any full-tier adapter reporting `maxStorageBuffersPerShaderStage` in the narrow [10,23] band (a boundary that predates BDPT).
- **Memory ceiling (already enforced, confirm the fallback fires):** the eye-stack is W·H·depth·32 B. 1080p×depth-8 = **506 MiB** exceeds the 384 MiB `GpuResources.BDPT_EYE_STACK_MAX_BYTES` ceiling, so BDPT is **skipped that frame with a `console.warn` and falls back to unidirectional** (a 32-byte placeholder keeps the layout valid). 1080p works up to ~depth-6; full-depth 1080p BDPT needs the deferred cap/tile decision.
- **Expected delta:** equal converged MEAN vs the 2-strategy build (both unbiased — same integral) with LOWER variance on caustic/specular-bounce paths where the extra strategies carry weight; BDPT-off bit-identical to a pre-change build.
- **Acceptance:** shaders compile on a real device; converged means match (unbiasedness) with measurable variance reduction on the BDPT-favourable scene; the 384 MiB fallback fires loudly (never silently) at 1080p×depth-8; BDPT-off identical to pre-change.

### V19 — GRIS reconnection-shift reuse on ReSTIR-GI spatial + temporal (walkaround-hybrid)  [implemented this session — OUTSTANDING: needs GPU shader-compile + converged-mean + variance A/B]
- **What changed:** the ReSTIR-GI spatial (`spatialGi.wgsl`) and temporal (`temporalGi.wgsl`) reuse passes gained an UNBIASED GRIS reconnection-shift path (Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai — "Generalized Resampled Importance Sampling: Foundations of ReSTIR", SIGGRAPH 2022, §5 + Eq. 12), opt-in via `HybridEngineOptions.restirPtReuse` (UBO gate `restirPtReuse` at offset 412, the former `_regirPad`). When ON, each accepted neighbour's reconnection sample is re-rooted onto this pixel's primary vertex via the reconnection shift T, re-weighted by the change-of-variables Jacobian `G(shifted)/G_base` (destination-cosine half-G, base half-G recovered from the Phase-0 cache `cosReconOut/distRecon²`), gated by a **reconnection-visibility ray** through the scene BVH/TLAS (the spatial + temporal pipelines now bind the shared scene group at `@group(1)`), and combined with the canonical sample under a **pairwise generalized-balance MIS** (defensive `1/(K+1)` canonical weight + per-neighbour pairwise weights). The GRIS finalise uses `W = w_sum / p̂` (the MIS weights replace the legacy `1/M` averaging). The WGSL geometry-term + Jacobian + pairwise-MIS arithmetic mirrors the CPU oracle `@vitrum/shared-samplers/reconnectionShift.ts` 1:1 via the TS mirror `pipeline/grisReuseMis.ts`, asserted equal to ≤1e-12 (oracle parity) + partition-of-unity (Σ mᵢ(y) = 1 for a fixed sample). The gate-OFF path is the verbatim Sprint-17 clamped-Jacobian reuse — **OFF is BIT-IDENTICAL** (unit-pinned: every UBO byte unchanged when the gate is 0; only u32[103] flips when ON).
- **Why GPU:** the reuse passes are WebGPU compute kernels; the unbiasedness claim and the variance/disocclusion-bias improvement can only be measured on a real device with a converged accumulation. **Do NOT fabricate renders** — the static gate (`wgslCompose.test.ts` ident-resolution) only proves the WGSL composes with all identifiers declared, not that it produces correct radiance.
- **Three things needing real hardware:**
  1. **WGSL shader-compile** — the spatial + temporal GRIS branch (reconnection-visibility `traceSceneAny`, `grisShiftJacobian`, pairwise-MIS denominators) is structurally reviewed + statically ident-resolved but unverified by a real shader compiler. Compile both pipelines on a GPU host (SwiftShader/Dawn or real adapter) with `restirPtReuse` on; the spatial/temporal pipelines now declare a 2nd bind group (scene BVH/TLAS) — confirm `createComputePipelineAsync` succeeds.
  2. **Converged-mean A/B (unbiasedness)** — render a walkaround GI scene (Cornell-box or a room with strong indirect color-bleed + a disocclusion edge, e.g. a moving occluder or a fresh camera pan) to convergence with `restirPtReuse` ON vs OFF at matched accumulation. **Telemetry:** per-pixel Welford mean on the converged indirect channel (`hdrIndirect`), NOT eyeballed screenshots (temporal pipeline → screenshots capture transient states). The converged MEANS must be EQUAL within noise (both unbiased estimators of the same GI integral) — a mean shift would indicate the Jacobian / MIS / visibility is wrong.
  3. **Variance / disocclusion-bias A/B** — at matched low accumulation (or right after a camera move), ON must show LOWER variance and LESS disocclusion bias than OFF (the legacy reuse leaks neighbour radiance through occluders + over-/under-weights without the proper Jacobian; the visibility ray + exact Jacobian fix both). Telemetry: per-pixel variance map + the disocclusion-edge error vs the converged reference.
- **Storage-buffer budget (confirm, do not assume):** the spatial/temporal GI pipelines now bind the shared scene group, so they reference at most 2 reservoir buffers (group 0) + the scene BVH subset they traverse — **fewer scene buffers than the already-shipping `shade`/`ris` pipelines** (which declare the full 11-entry scene group and work on the full tier). In merged-BVH mode the visibility ray touches only the 3 BVH buffers (nodes/index/position); TLAS slots are declared-but-untraversed (same discipline as shade). Confirm pipeline creation succeeds on the full tier; lite tier forbids the heavy GI features anyway but the GI reuse passes themselves stay within budget. **No budget overflow was introduced** — but verify on a real device.
- **Expected delta:** equal converged MEAN ON vs OFF (unbiased), LOWER variance + less disocclusion bias ON; gate-OFF bit-identical to a pre-change build (already unit-pinned on the UBO bytes).
- **Acceptance:** both pipelines compile with the gate on; converged means match (unbiasedness) with measurable variance / disocclusion-bias reduction; gate-OFF identical to pre-change (byte-level UBO assertion already green).

---

### V20 — Neural Radiance Caching (Müller et al. 2021) — STAGED gate + encoding + spread-termination (walkaround-hybrid)  [implemented this session — STAGED SUBSET; OUTSTANDING: query/record passes + GPU quality/perf A/B]
- **HONEST BIAS CAVEAT (read first):** NRC is a **BIASED estimator** — the cache is a *learned approximation* of the path suffix, NOT an unbiased Monte-Carlo estimate. Unlike V18 (BDPT MIS) / V19 (GRIS) which preserve the converged MEAN, NRC does **NOT** — its value is variance reduction / faster convergence / lower noise with **bounded bias**. The acceptance is therefore *perceptual closeness to the no-NRC reference within tolerance, with measurably faster convergence / lower noise*, **not** "equal converged mean." Do not assert mean-equality for NRC. NRC is opt-in / full-tier only — **it cannot run on lite / SwiftShader** (the hash-grid feature tables + fused-MLP weight/Adam buffers exceed the lite budget; the ctor throws on `tier:'lite' + nrcEnabled`).
- **What landed this session (the VERIFIABLE, gated-OFF-INERT subset):**
  1. **Gate plumbing, end-to-end + OFF-bit-identity-pinned.** `HybridEngineOptions.nrcEnabled` (default `false`) → parsed config → `_nrcEnabled` → frame deps → `PipelineFrameInputs.nrcEnabled` → UBO `nrcEnabled` at **offset 364** (the former `_ppgPad2` pad slot — **no UBO size change**, still 416 bytes). Lite-tier forbid in the ctor (mirrors rcEnabled / ppgEnabled / denoiser:'neural'). `nrcGateBitIdentity.test.ts` pins: omitting the field ≡ `nrcEnabled:0` byte-for-byte (416 bytes), and turning NRC **on flips ONLY u32[91]** — every other UBO byte unchanged. **OFF is BIT-IDENTICAL** to the pre-NRC GI.
  2. **The trainable input encoding (the hard "full"-version piece).** Multiresolution hash-grid positional encoding (Instant-NGP, Müller et al. 2022) — L levels of hashed feature grids, trilinearly interpolated, concatenated — with the **trainable backward** (gradient scatter into hashed table rows; collisions accumulate, §4) + its own fixed-point grad atomics matching the fused-MLP kernel; plus one-blob direction encoding (Müller et al. 2019 §4.3, fixed kernel) and raw surface features (normal/roughness/albedo). CPU oracle `nrcEncoding.ts`; WGSL forward + backward `wgsl/nrcEncoding.wgsl.ts` hand-verified line-for-line. `nrcEncoding.test.ts` pins: Instant-NGP hash primes ≡ WGSL multipliers, trilinear partition-of-unity, forward = weighted corner sum, backward **EXACT-ANALYTIC** (scatter = weight×dOut, collisions accumulate) + a **finite-difference cross-check** of a downstream ½‖feature‖² loss (smooth — no ReLU kink, unlike the MLP-internal FD), one-blob L1-normalisation, full-input assembly width/layout.
  3. **The path-spread cache-termination predicate (Müller 2021 §5).** `a(x) > c·a₀` with the per-segment spread term `sqrt(d²/(p·|cosθ|))`. CPU oracle `spreadTermination.ts`; WGSL `wgsl/spreadTermination.wgsl.ts`. `spreadTermination.test.ts` pins: per-segment term + degenerate-denom clamp, accumulated spread = (running Σ)² + monotonicity, a₀ = (first-segment term)², never-terminate-at-primary, fires-at-first-crossing, larger-c-defers (less bias).
- **What is NOT yet wired (the documented next phase):** the actual **cache-query** + **record-gather** compute passes are NOT registered. The integration SITE is identified and reasoned-about: `risGi.wgsl` `risGiMain`'s reconnection-vertex loop, where the suffix outgoing radiance `Lo` at the reconnection vertex `xs` is computed today by DDGI-atlas sampling (`Lo = irrAtXs * xsMat.rgb * INV_PI`). The query (terminate-into-cache when the spread heuristic fires → MLP forward at the encoded `xs` features → predicted `Lo`) and the record-gather (accumulated outgoing radiance at recorded suffix vertices = self-training target → one `trainStep` per frame on a batch) hook in THERE. Because those passes are unregistered, **even with the gate at 1 today the suffix still uses the DDGI estimate** — i.e. the current tree is gated-OFF-INERT and green regardless of the flag.
- **Why GPU (when the next phase lands):** the cache query + per-frame self-training are WebGPU compute kernels; the *biased*-cache quality (perceptual match within tolerance) and the perf/variance win can only be measured on a real device with a converged accumulation. The fused-MLP kernel itself is **already GPU-validated** (lavapipe: GPU == CPU-analytic ~9.5e-7), so the next phase's GPU validation is about the query/train INTEGRATION (does the cache converge to a useful suffix predictor + reduce noise), not the kernel arithmetic.
- **Three things needing real hardware (next phase):**
  1. **WGSL shader-compile** — the new query/train passes (hash-grid encode forward in the query path, encode forward+backward + MLP forward+backward+Adam in the train path) compile on a GPU host with `nrcEnabled` on.
  2. **Biased-cache QUALITY A/B (NOT mean-equality)** — render a walkaround GI scene (Cornell-box / strong indirect color-bleed room) with NRC ON vs OFF at matched accumulation. **Telemetry:** perceptual error (e.g. FLIP / relative-MSE) of the NRC-ON image vs the **NRC-OFF converged reference** must be within a stated tolerance; NRC-ON must reach a given noise floor in FEWER frames (faster convergence) and/or show lower per-pixel variance at matched frame count. **Do NOT assert equal converged means** — a small bounded mean shift is expected and acceptable within tolerance.
  3. **Self-training stability** — the per-frame `trainStep` must keep the cache stable (no divergence / oscillation) across a camera pan + a lighting change; telemetry: training loss trajectory + the cache prediction error on held-out records over time.
- **Storage-buffer budget (confirm, do not assume):** the next phase adds the hash-grid feature tables + the fused-MLP weight/Adam buffers + record-gather buffers. These are **full-tier-only** (lite forbids NRC). Confirm the query/train pipelines fit the full-tier 16-buffer / 8-texture budget when registered.
- **Expected delta:** gate-OFF **bit-identical** to a pre-NRC build (already unit-pinned on the UBO bytes); gate-ON (once the passes land) perceptually close to the NRC-OFF reference within tolerance with faster convergence / lower noise — **biased, not mean-preserving**.
- **Acceptance:** OFF byte-identical (green now); the encoding forward/backward + spread predicate match their CPU oracles to ~1e-6 f32 (green now); the next-phase query/train passes compile + the NRC-ON image is within the perceptual tolerance of the NRC-OFF reference with measurable convergence/variance improvement.

---

## 3. Suggested session order
Run the **probe** (confirm full tier + hybrid). Then V8 (cheap: does everything still compile?), then the render A/Bs V1–V7, V9, V10 (BMFR), and the later-wave items V11–V17 (skin bind, sun dedup, RC merged-refit, resolutionFactor coverage, DDGI cadence/2→32 default, RC⊕ReSTIR-GI MIS, PPG guided sampling). The cheapest functional checks (V8 compile, V11 skin-routing, V14 canvas-coverage) gate the rest. Adopt the post-sweep PNGs as new baselines per the benchmark-runner README §5 once each passes visual sign-off. **Note:** several V11–V16 items default-ON now (e.g. V15's stride-2 default cadence, V16's confidence MIS), so a plain `main` capture already exercises them.
