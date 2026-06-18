# Fidelity-Promotion Playbook (hero stack)

**Status:** active playbook (executable process). **Date:** 2026-05-28.
**Audience:** the `wsl-gpu` sibling agent and future sessions executing roadmap
`plan/roadmap.md` §6.1 "Fidelity promotion program."
**Scope:** the per-row process to promote each `plan/renderer-fidelity-matrix.md`
row from `experimental` → `supported`.

This doc is **process + per-row table + sequencing**, not new code. It threads
together three already-committed pieces:

1. `plan/renderer-fidelity-matrix.md` — the rows + the strict promotion rule.
2. `tools/gpu-env/` — the new headless WebGPU validation env (Deno+lavapipe) that
   clears the hybrid/full-tier limits in WSL (`FINDINGS.md`, `run-gpu-validation.sh`).
3. `tools/benchmark-runner/` — the capture / acceptance-metrics / scenario-preset
   harness (`README.md`) plus `HARDWARE-VALIDATION-NEEDS.md` (the A/B procedure).

---

## 0. The correctness-vs-perf distinction (READ FIRST)

lavapipe (Mesa LLVMpipe) is a **CPU rasteriser** that runs WGSL on the CPU via
LLVM. It is the only backend in this WSL2 box that **clears the hybrid storage
limits** (`maxStorageBuffers/Textures` advertised at 1e6 via wgpu-native, 16/8 via
Node-Dawn — both ≥ the required `16/8`), so it gives `hybridCanRun: true` and
`ptWebgpuFullTier: true` (`tools/gpu-env/FINDINGS.md:27-36, 99-105`).

Consequence — **two orthogonal kinds of evidence**:

| Evidence | What lavapipe gives you | What lavapipe CANNOT give you |
|----------|-------------------------|-------------------------------|
| **Correctness** (right pixels, right shader compile, right compute readback, full-tier bind layout) | ✅ YES — correct pixels at a fixed seed/SPP/resolution; deterministic reference hash; PSNR vs a committed baseline | — |
| **Performance** (`msPerSample`, `p95FrameMs`, ms/frame) | ❌ NO — any wall-clock number reflects the CPU rasteriser, not a real GPU | ✅ requires a **real adapter**: native Windows Chrome on the RTX 4090 (`VITRUM_USE_WIN_CHROME=1` / `run-gpu-host-windows.mjs`) |

> **The strict promotion rule** (`plan/renderer-fidelity-matrix.md:33-39`):
> a row is not `supported` until the matching acceptance scenario produces
> **non-null hashes, perf fields, and PASS**. The fidelity matrix's `supported`
> legend (`:12`) literally reads "Implemented + unit tests + **captured runtime
> evidence**." Runtime evidence has two halves; lavapipe satisfies the
> correctness half for free, the perf half only on real hardware.

**Practical reading of the rule for this playbook:** most hero-fidelity rows are
*about whether the pixels are right* (does thin-film shift hue with angle? does a
red light bleed red?). Those are **correctness claims** — lavapipe-validatable
NOW. A row's promotion is **only perf-gated** when the claim itself is about
throughput/interactivity (e.g. caustic-strategy "validated quality **and perf**"
per the legacy acceptance rule retained in `plan/roadmap.md` §6.1 and the
per-row table below). The per-row table (§2)
marks which is which.

### A known harness gap (do not skip)

The committed capture adapters are **browser-driven**: `capturePtWebgpu.mjs`
(`:20`) and `capture-adapter-playwright.mjs` both go through
`launchWebGpuBrowser.mjs` → Chromium → Dawn → **bundled SwiftShader 10/4** in WSL
(`FINDINGS.md:54-67`). They **cannot reach full tier in WSL**. The lavapipe path
is **native Deno/Node, not browser** — and per `FINDINGS.md:138-141` a
native-WebGPU capture entrypoint that `benchmark-runner` can target **does not
exist yet**. So step 2 of the process below has a one-time prerequisite: a native
capture adapter, or use the `VITRUM_USE_WIN_CHROME=1` real-GPU path. See §1.0.

---

## 1. The promotion process (repeatable, per row)

### 1.0 One-time prerequisite — pick a capture transport

You have three transports for producing a PNG at a fixed seed/SPP/resolution.
Choose per the row's evidence needs (§2):

| Transport | Reaches full tier? | Gives correct pixels? | Gives real perf? | How |
|-----------|--------------------|-----------------------|------------------|-----|
| **A. lavapipe native** (Deno/wgpu-native or Node-Dawn) | ✅ (1e6 / 16/8) | ✅ | ❌ CPU | `tools/gpu-env/run-gpu-validation.sh` proves the adapter; **a native PNG-capture adapter must be wired** (`FINDINGS.md:138-141`) — until then use the smoke renders the env can already produce, or transport C |
| **B. Playwright/Chromium in WSL** | ❌ SwiftShader 10/4 | ⚠️ lite-tier only, "slow and likely incorrect" at the matrix's 1280×720/512+ SPP (`benchmark-runner/README.md:404-406`) | ❌ | existing `capture-adapter-playwright.mjs` — **not usable for full-tier rows in WSL** |
| **C. Windows Chrome on the RTX 4090** | ✅ real GPU | ✅ | ✅ | `VITRUM_USE_WIN_CHROME=1` / `run-gpu-host-windows.mjs`; WSL-vite→win-chrome bridges `run-*-wsl-vite-win-chrome.mjs` |

**Rule:** for **correctness-only** rows, transport **A** (lavapipe) is sufficient
and is the point of the WSL env. For **perf-gated** rows, you additionally need
transport **C** for the `msPerSample`/frame-time number. Transport B is only good
for the `lite`-tier smoke and the already-passing `ptwgpu-parity` baseline.

### 1.1 The steps (mirrors roadmap §6.1 steps 1–4)

For each row:

1. **Confirm the adapter.** Run the limit probe and require full tier:
   ```bash
   tools/gpu-env/run-gpu-validation.sh probe   # expect hybridCanRun:true, ptWebgpuFullTier:true
   ```
   (On a real-GPU host instead:
   `VITRUM_PROBE_START_SERVER=1 npm run benchmark:pt-webgpu-adapter-probe` — require
   `ptWebgpuFullTier:true`, `hybridCanRun:true`; `HARDWARE-VALIDATION-NEEDS.md:10-20`.)
   Do **not** trust any capture until the probe passes.

2. **Mechanical gate (already green; re-run to pin).**
   `npm run typecheck && npm test`; use `npm run shader-gate` for WGSL pass
   graph coverage and `npm run shader-gate:glsl` for native pt-webgl2 GLSL
   coverage. The matrix's "Mechanical evidence" column names the exact test
   file per row — that test must be green.

3. **Capture at the row's fixed scenario.** Use the scenario's
   seed/resolution/bounces/SPP from `scenario-presets.mjs` (the canonical values
   are tabulated in §2). The pt-webgpu scenarios are wired in
   `gapClosurePtWebgpuMap.mjs`. Capture command (correctness, lavapipe/real-GPU):
   ```bash
   VITRUM_GPU_CAPTURE=1 \
   VITRUM_CAPTURE_CMD="<native-lavapipe-adapter OR capture-adapter-playwright.mjs under VITRUM_USE_WIN_CHROME=1>" \
   VITRUM_GAP_SCENARIOS=<scenarioId> \
     npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
   ```
   The runner diffs the candidate PNG against the committed baseline under
   `tools/reference-renders/baseline/<scenarioId>.png` and computes a hash + PSNR
   + mean-abs delta (`benchmark-runner/README.md:97-105, 284-322`).

4. **Acceptance criteria.** A row PASSes when:
   - **Reference hash** is non-null (capture actually ran — not a SwiftShader
     skip) AND, for A/B against `bbd32c8`, the change is **in the direction the
     math predicts with no NEW artifacts** (`benchmark-runner/README.md:507-513`;
     `HARDWARE-VALIDATION-NEEDS.md:44`).
   - **PSNR vs baseline** ≥ the threshold. Global default **28 dB**; per-scenario
     overrides via `VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO` (e.g. caustics
     relaxed to 26: `benchmark-runner/README.md:390-391`). For the strict pt-webgpu
     TLAS-style rows, mean/p95/max delta thresholds (`0.02 / 0.06 / 0.2`,
     `benchmark-runner/README.md:342-347`).
   - **`msPerSample`** (hero) or **`p95FrameMs`** (hybrid) is present and within
     the tier budget — **only for perf-gated rows (§2), captured on a real GPU**.
     The runner stores `<scenario>.png.json` beside the baseline as
     `perfBaselineMsPerSample` (`benchmark-runner/README.md:299-301`).
   - Flip the strict gate on once baselines are stable:
     `VITRUM_STRICT_GAP_CLOSURE=1` (`benchmark-runner/README.md:303-304`) /
     `VITRUM_PTWEBGL_FIDELITY_STRICT=1` (`:392`).

5. **Record the signoff.** Three places, in this order:
   - Commit the candidate PNG as the new baseline under
     `tools/reference-renders/baseline/` (adopt per `benchmark-runner/README.md:537-555`).
   - Flip the row in `plan/renderer-fidelity-matrix.md` from `experimental` →
     `supported` and update its "Runtime evidence" cell with the artifact path +
     hash + (where applicable) ms/sample.
   - Append a dated line to the matrix's revision history and to a new
     `plan/fidelity-signoff-<date>.md` (mirror the `PR/WG-signoff-2026-05-26.md`
     convention) capturing scenario id, seed, resolution, hash pair, perf, PASS.
     The acceptance-artifact contract is the §6.1 promotion process in
     `plan/roadmap.md` plus the benchmark-runner artifact guidance.

---

## 2. Per-row promotion table (ALL fidelity-matrix rows)

Source rows: `plan/renderer-fidelity-matrix.md:19-31`. "Implemented?" was
**verified by reading the code**, not by trusting the matrix.

| # | Feature | Backend | Implemented? (verified) | Correctness-validatable on lavapipe NOW? | Needs real-GPU perf? | Scenario (seed / res / spp) | Promotion blocker |
|---|---------|---------|-------------------------|-------------------------------------------|----------------------|------------------------------|-------------------|
| 1 | Hero-λ + CMF accumulation | pt-webgpu | ✅ `HERO_WAVELENGTH_WGSL` real Wilkie 2014 MIS+CMF→RGB (`shared-samplers/src/wgsl/heroWavelength.wgsl.ts:1-60`) | ✅ | No | `rfe08-13-spectral-payload` / `ptwgpu-spectral-hero` | ✅ CLOSED for pt-webgpu: renderer matrix cites dzn spectral ON/OFF A/B plus committed `baseline/ptwgpu-spectral-hero.png`; `npm run renderer-fidelity-proof-check` verifies the matrix text, PNG, and dzn full-tier status. |
| 2 | Spectral Beer–Lambert (packed μ) | pt-webgpu | ✅ `sampleMaterialSpectralMu` packed accessor + 32-bin grid (`pt-webgpu/.../material.wgsl.ts:206, 423`) | ✅ | No | `rfe08-13-spectral-payload` / spectral status | ✅ CLOSED for pt-webgpu: matrix records the dzn μ-curve present-vs-absent A/B and the proof check pins the dzn spectral full-tier status. |
| 3 | Multi-layer thin-film TMM | pt-webgpu | ✅ `thinFilmTmmRt` real Belcour & Barla transfer-matrix solver (`pt-webgpu/.../material.wgsl.ts:229-285`) | ✅ (hue-vs-angle is a pixel claim) | No | `rfe14-thinfilm-angle-shift` / `ptwgpu-thinfilm-angle` | ✅ CLOSED for pt-webgpu: matrix cites the dzn hue-vs-angle A/B plus committed `baseline/ptwgpu-thinfilm-angle.png`; proof check pins both. |
| 4 | Cauchy dispersion | pt-webgpu | ✅ `cauchyIorAtLambda` real (`pt-webgpu/.../material.wgsl.ts:340-351`); gated on spectral ext + `dispersionAbbeNumber` | ✅ | No | `rfe08-13-spectral-payload` / `ptwgpu-cauchy-dispersion` | ✅ CLOSED for pt-webgpu: matrix cites dzn Abbe-set A/B plus committed `baseline/ptwgpu-cauchy-dispersion.png`; proof check also pins dzn spectral full-tier status. |
| 5 | Layered front/back + transmission MIS | pt-webgpu | ✅ `activeLayerWeightRgb` + η² PDF (matrix mech: `wgslContract.test.ts`); WG-4 landed | ✅ | No | `rfe03-layered-front-back` / `ptwgpu-layered-front` | ✅ CLOSED for pt-webgpu: matrix cites dzn front/back A/B plus committed `baseline/ptwgpu-layered-front.png`; proof check pins the artifact. |
| 6 | SSS / translucent panels | pt-webgpu | ✅ derived `isTranslucent` gate (mech: `wgslContract.test.ts`) | ✅ (mixed-panel "SSS only where flagged" is a pixel claim) | No | `rfe07-11-sss-mixed-panels` / `ptwgpu-sss-mixed-panels` | ✅ CLOSED for pt-webgpu: matrix cites dzn mixed-panel A/B plus committed `baseline/ptwgpu-sss-mixed-panels.png`; proof check pins the artifact. |
| 7 | Multi-emitter direct lighting | pt-webgpu | ✅ bounded emitter arrays (mech: `scenePack.test.ts`, `wgslContract.test.ts`) | ✅ (≈2× floor irradiance is a pixel claim; cf. `m5-multi-light-cornell`) | No | `rfe09-bridge-global-cmf` / `cornell-manylights` | ✅ CLOSED for pt-webgpu: matrix cites committed `baseline/cornell-manylights.png`; proof check also pins point/disc/spot dzn full-tier status. |
| 8 | Material-fields parity (cornell) | pt-webgpu | ✅ (pt-webgl2 is tracked in the fidelity matrix separately); pt-webgpu side WG-0 baseline committed | ✅ | No | `ptwgpu-parity-material-fields` (777 / 1280×720 / 512) | ✅ CLOSED for pt-webgpu: matrix records byte-for-byte strict-hash re-capture (PSNR 999 dB) against `baseline/ptwgpu-parity-material-fields.png`; proof check pins the artifact. |
| 9 | Caustic strategies | pt-webgpu | ✅ strategy plumbing (mech: `factoryCapabilities.test.ts`); full tier only | ✅ correctness | No for current fidelity row; real-GPU throughput remains a separate performance track | `rfe05-caustic-strategy` / `mnee-glass-slab` | ✅ CLOSED for pt-webgpu fidelity: matrix cites deterministic MNEE GPU validation plus committed `baseline/mnee-glass-slab.png`; proof check also pins dzn caustic full-tier status. |
| 10 | SVGF-real denoiser | pt-webgpu | ❌ **`unsupported` — intentional regime mismatch, NOT promotable** (wiring removed; mech: `unsupportedDenoiserDegrade.test.ts` asserts warn + degrade-to-no-denoise) | n/a | n/a | n/a | pt-webgpu is a CONVERGED progressive tracer; SVGF is a real-time 1-spp spatiotemporal filter. The converged denoiser is **`oidn-final`**. SVGF stays in `shared-denoisers` for the realtime walkaround stack only. Do not "promote" — nothing to capture. |
| 10b | SVGF-real denoiser | **pt-webgl2** | ❌ **`unsupported` — same regime mismatch** (converged tracer; only `oidn-final` is wired in `ptEngineWebGL2.ts`) | n/a | n/a | n/a | Same as #10: SVGF-real is real-time-only and intentionally unsupported on this converged backend. Use `oidn-final`. Not a code gap to fill. |
| 11 | BDPT (eye↔light) | pt-webgpu | ✅ GPU light-subpath shipped per roadmap §0.5 (`bdptExtendLightSubpath` @compute); CPU fill + kernel eval (mech: `bdptPlumbing.test.ts`) | ✅ converged A/B is a pixel claim | No (correctness); perf is a separate throughput win (roadmap §6.2) | Cornell-box BDPT-on / `cornell-bdpt-on` | ✅ CLOSED for pt-webgpu fidelity: matrix cites V18/V25 GPU validation plus committed `baseline/cornell-bdpt-on.png`; proof check also pins dzn BDPT full-tier status. Multi-vertex BDPT remains a separate research-mode Road tail, not this matrix row. |
| 11b | BDPT (eye↔light) | pt-webgl2 | ✅ native WebGL2 path (mech: `bdptDriver.test.ts`, `composeTraceGlsl.test.ts`) | ⚠️ pt-webgl2 is **WebGL2** — native lavapipe here is a **WebGPU** device, so this row still needs a browser GL capture path. | No | same Cornell BDPT scene, `backend: pt-webgl2` | pt-webgl2 needs a real-browser GL capture; lavapipe-WebGPU does not cover WebGL2 rows |

**Already promoted for pt-webgpu:** #1, #2, #3, #4, #5, #6, #7, #8, #9, and
#11 are no longer queued adapter work. `plan/renderer-fidelity-matrix.md` is the
active truth table, and `npm run renderer-fidelity-proof-check` verifies that
its pt-webgpu `supported` rows still cite committed runtime evidence.

**Additionally need real-GPU/browser (transport C):**
- Any pt-webgl2/WebGL2 row (#11b plus pt-webgl2 spectral/caustic fidelity rows)
  — pt-webgl2 is **WebGL2**, and the
  native lavapipe device is **WebGPU**; WebGL2 capture still needs a real browser.
- Real-GPU caustic throughput remains a performance-program item, not a blocker
  for the current pt-webgpu fidelity-matrix support grade.

**Not promotable at all (intentional regime mismatch):** #10 and #10b SVGF-real
are `unsupported` on **both** converged backends (pt-webgpu and pt-webgl2) by
design — SVGF is a real-time 1-spp spatiotemporal filter, and a converged
progressive tracer's denoiser is **`oidn-final`**. This is not a code gap to fill;
the real SVGF impl stays in `shared-denoisers` for the realtime walkaround stack.

---

## 3. Sequencing (current)

### Closed for pt-webgpu fidelity

Rows #1-#9 and #11 are closed for the pt-webgpu column of the renderer-fidelity
matrix. Keep `npm run renderer-fidelity-proof-check` green when editing the
matrix or moving baselines.

### Still queued

1. **pt-webgl2/WebGL2 fidelity rows** — WebGL2 path; needs real-browser GL
   capture. The WSL native lavapipe device is a WebGPU adapter and cannot prove
   pt-webgl2 runtime A/B rows.
2. **Research/performance tails** — multi-vertex BDPT radiometric promotion and
   caustic strategy throughput remain Road/performance-program work. They should
   not be represented as missing pt-webgpu fidelity-matrix proof.

(SVGF-real, formerly Tier-3 #10, is no longer in the promotion queue — it is
`unsupported` on both converged backends by design; see "Not in the promotion
queue" below.)

### Not in the promotion queue — real code work (roadmap §0.5, north-star fidelity)

- **#10 / #10b SVGF-real (pt-webgpu AND pt-webgl2)** — `unsupported` by **design**,
  not a missing-feature gap. Both are converged progressive tracers; SVGF is a
  real-time 1-spp spatiotemporal filter and the wrong regime. The converged
  denoiser is **`oidn-final`**. Do **not** implement or "promote" — the real SVGF
  impl stays in `shared-denoisers` for the realtime walkaround stack only.
- **pt-webgl2 spectral fidelity** — current source consumes the spectral payloads
  called out in the renderer matrix; the remaining blocker is browser/adapter A/B
  capture, not retired fork placeholder work.
- **BMFR denoiser** — SHIPPED (roadmap §0.5 item 1): real Koskela-2019 blockwise
  Householder-QR regression in `shared-denoisers` (`wgsl/bmfr.wgsl.ts`,
  `bmfrRegression.ts`) + `BmfrDenoiser` in walkaround. Not a fidelity-matrix row.

### Dependency summary

```
pt-webgpu fidelity rows (#1-#9, #11) ──► CLOSED; guarded by renderer-fidelity-proof-check
real GPU / Win-Chrome (transport C) ───► pt-webgl2/WebGL2 browser A/B rows + perf tails
SVGF-real (#10 / #10b) ──► NOT in queue: unsupported by design on both converged backends (oidn-final is the converged denoiser)
```

---

## 4. Quick-reference commands

```bash
# 0. Prove the WSL adapter clears full tier (do this once per session)
tools/gpu-env/run-gpu-validation.sh probe       # want hybridCanRun:true, ptWebgpuFullTier:true
tools/gpu-env/run-gpu-validation.sh smoke        # WGSL compile + compute readback PASS

# 1. Mechanical gate (per-row test named in the matrix's "Mechanical evidence" col)
npm run typecheck && npm test
npm run shader-gate                               # WebGPU/WGSL pass graph
npm run shader-gate:glsl                          # pt-webgl2 GLSL programs

# 2. Correctness capture + diff for one row (transport A lavapipe, once adapter wired; else C)
VITRUM_GPU_CAPTURE=1 \
VITRUM_CAPTURE_CMD="<native-lavapipe-adapter | playwright under VITRUM_USE_WIN_CHROME=1>" \
VITRUM_GAP_SCENARIOS=rfe14-thinfilm-angle-shift \
  npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner

# 3. Perf number (PERF-GATED rows only — REAL GPU, Windows Chrome on the dGPU)
VITRUM_USE_WIN_CHROME=1 ... node tools/benchmark-runner/run-gpu-host-windows.mjs

# 4. pt-webgl2 fidelity artifact (WebGL2 rows; real browser)
# Use the browser capture adapter for the relevant pt-webgl2 scenario; the
# retired fork-specific fidelity benchmark script no longer exists.

# 5. Strict gate ON once baselines stable
VITRUM_STRICT_GAP_CLOSURE=1 ...                   # gap-closure rows
# WebGL2 browser rows should use their lane-specific proof/check command once
# a committed browser capture lane exists.
```

---

## 5. Related documents

| Document | Role |
|----------|------|
| `plan/renderer-fidelity-matrix.md` | The rows + strict promotion rule (flip rows here) |
| `plan/roadmap.md` §6.1 | The program this playbook executes |
| `tools/gpu-env/FINDINGS.md` | What lavapipe can/can't do (correctness yes, perf no) |
| `tools/gpu-env/run-gpu-validation.sh` | One-command headless WSL probe + compute smoke |
| `tools/benchmark-runner/README.md` | Capture, acceptance-metrics, scenario presets, strict gates |
| `tools/benchmark-runner/scenario-presets.mjs` | Canonical per-scenario seed/res/spp |
| `tools/benchmark-runner/gapClosurePtWebgpuMap.mjs` | pt-webgpu scenario → capture URL/options |
| `HARDWARE-VALIDATION-NEEDS.md` | A/B procedure, baseline commit `bbd32c8`, V1–V8 GPU items |
| `plan/road-to-100.md` | Current Road status and proof-gate ownership |
