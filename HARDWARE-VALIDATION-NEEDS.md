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

### V6 — pt-webgl instanced-mesh support  [implemented this session — needs-render]
- **What changed:** pt-webgl now renders `instanced-mesh` primitives (expands the InstancedMesh into N baked meshes pt-webgl-side, since the fork's geometry generator ignores `instanceMatrix`); `instanced-mesh` re-declared in pt-webgl's `supportedPrimitiveKinds` + ledger.
- **Why GPU:** pt-webgl path tracer.
- **Test scene / plan:** a scene with an `instanced-mesh` of N≥3 instances at distinct transforms, pt-webgl backend. Capture `main`; compare against a hand-authored equivalent of N separate meshes.
- **Expected delta:** all N instances render at their correct per-instance transforms (was: one instance at origin, or warn-skipped).
- **Acceptance:** instance count + transforms match the N-separate-meshes reference; no missing/origin-collapsed instances.

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

---

## 3. Suggested session order
Run the **probe** (confirm full tier + hybrid). Then V8 (cheap: does everything still compile?), then the render A/Bs V1–V7. Adopt the post-sweep PNGs as new baselines per the benchmark-runner README §5 once each passes visual sign-off.
