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
per `plan/archive/gap-closure-acceptance-matrix.md:24`). The per-row table (§2)
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
   `npm run typecheck && npm test` and, for fork rows, `npm run fork-shader-smoke`
   (`plan/renderer-fidelity-matrix.md:35`). The matrix's "Mechanical evidence"
   column names the exact test file per row — that test must be green.

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
     The acceptance-artifact contract is `roadmap §10.3` +
     `plan/archive/gap-closure-acceptance-matrix.md:40-49`.

---

## 2. Per-row promotion table (ALL fidelity-matrix rows)

Source rows: `plan/renderer-fidelity-matrix.md:19-31`. "Implemented?" was
**verified by reading the code**, not by trusting the matrix.

| # | Feature | Backend | Implemented? (verified) | Correctness-validatable on lavapipe NOW? | Needs real-GPU perf? | Scenario (seed / res / spp) | Promotion blocker |
|---|---------|---------|-------------------------|-------------------------------------------|----------------------|------------------------------|-------------------|
| 1 | Hero-λ + CMF accumulation | pt-webgpu | ✅ `HERO_WAVELENGTH_WGSL` real Wilkie 2014 MIS+CMF→RGB (`shared-samplers/src/wgsl/heroWavelength.wgsl.ts:1-60`) | ✅ | No | `rfe08-13-spectral-payload` (4242 / 1280×720 / 1024) | Native lavapipe capture adapter not yet wired (§1.0) |
| 2 | Spectral Beer–Lambert (packed μ) | pt-webgpu | ✅ `sampleMaterialSpectralMu` packed accessor + 32-bin grid (`pt-webgpu/.../material.wgsl.ts:206, 423`) | ✅ | No | `rfe08-13-spectral-payload` (4242 / 1280×720 / 1024) | Same as #1 (shares the spectral preset + extension flag) |
| 3 | Multi-layer thin-film TMM | pt-webgpu | ✅ `thinFilmTmmRt` real Belcour & Barla transfer-matrix solver (`pt-webgpu/.../material.wgsl.ts:229-285`) | ✅ (hue-vs-angle is a pixel claim) | No | `rfe14-thinfilm-angle-shift` (9001 / 1280×720 / 1024) | Capture adapter (§1.0); needs the angle-sweep variant captured |
| 4 | Cauchy dispersion | pt-webgpu | ✅ `cauchyIorAtLambda` real (`pt-webgpu/.../material.wgsl.ts:340-351`); gated on spectral ext + `dispersionAbbeNumber` | ✅ | No | `rfe08-13-spectral-payload` (shares spectral preset) | Same as #1/#2; no dedicated dispersion scenario yet — may need a prism/fan scene |
| 5 | Layered front/back + transmission MIS | pt-webgpu | ✅ `activeLayerWeightRgb` + η² PDF (matrix mech: `wgslContract.test.ts`); WG-4 landed | ✅ | No | `rfe03-layered-front-back` (1337 / 1280×720 / 512) | Capture adapter (§1.0); baseline PNG already committed (`baseline/rfe03-layered-front-back.png`) |
| 6 | SSS / translucent panels | pt-webgpu | ✅ derived `isTranslucent` gate (mech: `wgslContract.test.ts`) | ✅ (mixed-panel "SSS only where flagged" is a pixel claim) | No | `rfe07-11-sss-mixed-panels` (2027 / 1280×720 / 512) | Capture adapter (§1.0); baseline PNG committed |
| 7 | Multi-emitter direct lighting | pt-webgpu | ✅ bounded emitter arrays (mech: `scenePack.test.ts`, `wgslContract.test.ts`) | ✅ (≈2× floor irradiance is a pixel claim; cf. `m5-multi-light-cornell`) | No | `rfe09-bridge-global-cmf` (31415 / 1024×1024 / 256) or `m5-multi-light-cornell` (6121) | Capture adapter (§1.0); no emitter-count-only baseline committed yet |
| 8 | Material-fields parity (cornell) | pt-webgpu | ✅ (pt-webgl already `supported`); pt-webgpu side WG-0 baseline committed | ✅ | No | `ptwgpu-parity-material-fields` (777 / 1280×720 / 512) | **Closest to done** — baseline `baseline/ptwgpu-parity-material-fields.png` committed; strict-hash re-capture on full-tier adapter is the only step |
| 9 | Caustic strategies | pt-webgpu | ✅ strategy plumbing (mech: `factoryCapabilities.test.ts`); full tier only | ✅ correctness; ⚠️ **also a perf/quality claim** | **YES** (quality **and perf** per `gap-closure-acceptance-matrix.md:24`) | `rfe05-caustic-strategy` (27182 / 1280×720 / 1024), 3 variants `none / manifold-nee / photon-map` | Real-GPU perf number required; PSNR relaxed to 26 |
| 10 | SVGF-real denoiser | pt-webgpu | ❌ **`unsupported` — intentional regime mismatch, NOT promotable** (wiring removed; mech: `unsupportedDenoiserDegrade.test.ts` asserts warn + degrade-to-no-denoise) | n/a | n/a | n/a | pt-webgpu is a CONVERGED progressive tracer; SVGF is a real-time 1-spp spatiotemporal filter. The converged denoiser is **`oidn-final`**. SVGF stays in `shared-denoisers` for the realtime walkaround stack only. Do not "promote" — nothing to capture. |
| 10b | SVGF-real denoiser | **pt-webgl** | ❌ **`unsupported` — same regime mismatch** (converged tracer; only `oidn-final` is wired in `ptEngineWebGL2.ts`) | n/a | n/a | n/a | Same as #10: SVGF-real is real-time-only and intentionally unsupported on this converged backend. Use `oidn-final`. Not a code gap to fill. |
| 11 | BDPT (eye↔light) | pt-webgpu | ✅ GPU light-subpath shipped per roadmap §0.5 (`bdptExtendLightSubpath` @compute); CPU fill + kernel eval (mech: `bdptPlumbing.test.ts`) | ✅ converged A/B is a pixel claim | No (correctness); perf is a separate throughput win (roadmap §6.2) | Cornell-box BDPT-on scene at fixed seed (cf. `HARDWARE-VALIDATION-NEEDS.md V1`) | Capture adapter (§1.0); no dedicated BDPT gap-closure scenario in presets — author one |
| 11b | BDPT (eye↔light) | pt-webgl | ✅ fork path (mech: `forkUniformBridge.test.ts`) | ⚠️ pt-webgl is **WebGL2** — runs on lavapipe's GL surface poorly; native lavapipe is a **WebGPU** device. Use Windows Chrome (transport C). | No | same Cornell BDPT scene, `backend: pt-webgl` | pt-webgl needs a **GL** capture path; lavapipe-WebGPU does not help WebGL2 rows |

**Lavapipe-NOW rows (correctness, no perf gate):** #1, #2, #3, #4, #5, #6, #7,
#8, #11(pt-webgpu) — i.e. **every pt-webgpu hero-material/spectral/layered/
SSS/emitter/BDPT row**. These are the bulk of the matrix and the WSL env's whole
point: their claims are *"are the pixels physically right"*, which a CPU
rasteriser answers correctly.

**Additionally need real-GPU (transport C):**
- #9 caustic strategies — the acceptance criterion explicitly includes **perf**.
- Any pt-webgl-backed row (#8 is pt-webgpu; #11b, and pt-webgl spectral/caustics
  fidelity via `benchmark:pt-webgl-fidelity`) — pt-webgl is **WebGL2**, and the
  native lavapipe device is **WebGPU**; WebGL2 capture still needs a real browser.

**Not promotable at all (intentional regime mismatch):** #10 and #10b SVGF-real
are `unsupported` on **both** converged backends (pt-webgpu and pt-webgl) by
design — SVGF is a real-time 1-spp spatiotemporal filter, and a converged
progressive tracer's denoiser is **`oidn-final`**. This is not a code gap to fill;
the real SVGF impl stays in `shared-denoisers` for the realtime walkaround stack.

---

## 3. Sequencing (cheapest/highest-confidence first)

### Tier 1 — promote now on lavapipe, lowest risk

1. **#8 material-fields parity (pt-webgpu)** — baseline already committed
   (`baseline/ptwgpu-parity-material-fields.png`); just needs a full-tier strict
   re-capture (lavapipe clears full tier). Highest confidence, lowest effort.
2. **#5 layered front/back** and **#6 SSS/translucent** — baselines committed
   (`baseline/rfe03-*`, `baseline/rfe07-11-*`); WG-4 already landed for #5. Pure
   correctness A/B against the committed PNGs.
3. **#3 thin-film TMM** — verified real solver; baseline committed
   (`baseline/rfe14-*`); needs the angle-sweep variant captured. Visually
   striking + deterministic = high-confidence promotion.

### Tier 2 — lavapipe, depends on the spectral implementation landing

4. **#1 hero-λ, #2 spectral Beer–Lambert, #4 Cauchy dispersion** — all share the
   `rfe08-13-spectral-payload` preset + the `vitrum.ptWebgpu.spectralHeroWavelength`
   extension. The pt-webgpu side is verified-real. **Dependency:** the concurrent
   **pt-webgl real Jakob-Hanika** work (roadmap §0.5 item 2 — replace the
   `jakob-hanika-placeholder`). The pt-webgpu rows do **not** block on it (they use
   hero-λ + packed μ, not the fork's RGB→spectrum upsampling), but if the
   sibling agent is mid-flight in `shared-samplers/jakobHanika.ts` /
   `pt-webgl/forkUniformBridge.ts`, do **not** re-capture pt-webgl spectral until
   that lands — capture the **pt-webgpu** spectral rows first, which are
   independent.

### Tier 3 — lavapipe correctness, but author a scenario first

5. **#7 multi-emitter** — no emitter-count-only baseline committed; reuse
   `m5-multi-light-cornell` (seed 6121) or add a preset, then capture.
6. **#11 BDPT (pt-webgpu)** — GPU light-subpath is shipped; no dedicated
   gap-closure scenario exists. Author a Cornell-BDPT-on preset (cf.
   `HARDWARE-VALIDATION-NEEDS.md V1`, seeded), then lavapipe-capture the
   converged A/B.

(SVGF-real, formerly Tier-3 #10, is no longer in the promotion queue — it is
`unsupported` on both converged backends by design; see "Not in the promotion
queue" below.)

### Tier 4 — requires real GPU (transport C), schedule on a Windows-Chrome session

8. **#9 caustic strategies** — perf-gated (`none / manifold-nee / photon-map`
   ms/sample on the RTX 4090 via `VITRUM_USE_WIN_CHROME=1`). Cannot finish on
   lavapipe.
9. **All pt-webgl-backed rows** (#11b, pt-webgl spectral/caustic fidelity via
   `benchmark:pt-webgl-fidelity`) — WebGL2 path; needs real-browser GL capture.

### Not in the promotion queue — real code work (roadmap §0.5, north-star fidelity)

- **#10 / #10b SVGF-real (pt-webgpu AND pt-webgl)** — `unsupported` by **design**,
  not a missing-feature gap. Both are converged progressive tracers; SVGF is a
  real-time 1-spp spatiotemporal filter and the wrong regime. The converged
  denoiser is **`oidn-final`**. Do **not** implement or "promote" — the real SVGF
  impl stays in `shared-denoisers` for the realtime walkaround stack only.
- **pt-webgl Jakob-Hanika placeholder → real** (roadmap §0.5 item 2) — concurrent
  work; gates the *pt-webgl* spectral fidelity numbers (not the pt-webgpu rows).
- **BMFR denoiser** — in the `denoiser` union (`core/src/engine/factory.ts:70`)
  but not in `shared-denoisers`; roadmap §0.5 item 1 says implement, do not
  remove. Not a fidelity-matrix row; tracked separately.

### Dependency summary

```
lavapipe env (tools/gpu-env, DONE) ─┬─► Tier 1 (#8, #5, #6, #3)         [promote now]
                                    ├─► Tier 2 (#1, #2, #4)             [pt-webgpu independent of pt-webgl spectral]
                                    └─► Tier 3 (#7, #11-ptwgpu)         [author scenario, then capture]

native lavapipe PNG adapter (NOT WIRED, FINDINGS.md:138) ──► unblocks all of the above for hands-free WSL capture
real GPU / Win-Chrome (transport C) ──► Tier 4 (#9 perf, all pt-webgl rows)
SVGF-real (#10 / #10b) ──► NOT in queue: unsupported by design on both converged backends (oidn-final is the converged denoiser)
pt-webgl Jakob-Hanika real (CODE, concurrent) ──► pt-webgl spectral fidelity numbers
```

---

## 4. Quick-reference commands

```bash
# 0. Prove the WSL adapter clears full tier (do this once per session)
tools/gpu-env/run-gpu-validation.sh probe       # want hybridCanRun:true, ptWebgpuFullTier:true
tools/gpu-env/run-gpu-validation.sh smoke        # WGSL compile + compute readback PASS

# 1. Mechanical gate (per-row test named in the matrix's "Mechanical evidence" col)
npm run typecheck && npm test
npm run fork-shader-smoke                         # fork (pt-webgl) rows only

# 2. Correctness capture + diff for one row (transport A lavapipe, once adapter wired; else C)
VITRUM_GPU_CAPTURE=1 \
VITRUM_CAPTURE_CMD="<native-lavapipe-adapter | playwright under VITRUM_USE_WIN_CHROME=1>" \
VITRUM_GAP_SCENARIOS=rfe14-thinfilm-angle-shift \
  npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner

# 3. Perf number (PERF-GATED rows only — REAL GPU, Windows Chrome on the dGPU)
VITRUM_USE_WIN_CHROME=1 ... node tools/benchmark-runner/run-gpu-host-windows.mjs

# 4. pt-webgl fidelity PSNR artifact (WebGL2 rows; real browser)
npm run benchmark:pt-webgl-fidelity --workspace @vitrum/benchmark-runner

# 5. Strict gate ON once baselines stable
VITRUM_STRICT_GAP_CLOSURE=1 ...                   # gap-closure rows
VITRUM_PTWEBGL_FIDELITY_STRICT=1 ...              # pt-webgl fidelity rows
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
| `plan/archive/gap-closure-acceptance-matrix.md` | Per-RFE acceptance criteria (incl. caustics perf gate) |
