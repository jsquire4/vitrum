# @vitrum/benchmark-runner

This workspace contains executable benchmark and hardening runners (`run-*.mjs`)
for sweep gates, capture diffs, and backend reliability evidence.

## Current runner

- `npm run benchmark --workspace @vitrum/benchmark-runner`
- `npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner`
- `npm run benchmark:qualitymodes --workspace @vitrum/benchmark-runner`
- `npm run benchmark:lifecycle-soak --workspace @vitrum/benchmark-runner`
- `npm run benchmark:wave4-hardening --workspace @vitrum/benchmark-runner`
- `npm run benchmark:acceptance-metrics --workspace @vitrum/benchmark-runner`
- `npm run benchmark:pt-webgl-fidelity --workspace @vitrum/benchmark-runner`
- `npm run benchmark:wave0-baseline --workspace @vitrum/benchmark-runner`
- `npm run typecheck --workspace @vitrum/benchmark-runner`
- `npm test --workspace @vitrum/benchmark-runner`

Primary outputs include:

- `tools/benchmark-runner/results/gap-closure-verification-2026-05-10.json`
- `tools/benchmark-runner/results/quality-modes-<timestamp>.json`
- `tools/benchmark-runner/results/lifecycle-soak-<timestamp>.json`
- `tools/benchmark-runner/results/wave0/wave0-baseline-<timestamp>.json`
- `tools/benchmark-runner/results/wave4/wave4-hardening-<timestamp>.json`

## PR-6 hybrid release benchmarks (metadata)

Scenario ids for the primary-release program live in `scenario-presets.mjs` as
`PR_HYBRID_BENCHMARK_SCENARIOS`:

| Scenario ID | Intent |
|-------------|--------|
| `PR-hybrid-200k-static` | p95 frame time on ~200k tri static scene |
| `PR-hybrid-tlas-10-inst` | p95 frame time with 10 instanced meshes (TLAS) |
| `PR-hybrid-material-churn` | 100× `updatePrimitive` material-only, zero pipeline reinit |
| `PR-hybrid-emitter-churn` | 100× `updateEmitter` intensity patch |

Host harness wiring:

```bash
# Terminal A
npm run dev --workspace @vitrum-examples/two-engines-one-scene

# Terminal B — material / emitter churn (needs WebGPU + Playwright)
npm run benchmark:pr-hybrid --workspace @vitrum/benchmark-runner

# Single scenario
VITRUM_PR_SCENARIO=PR-hybrid-material-churn npm run benchmark:pr-hybrid --workspace @vitrum/benchmark-runner

# Auto-start dev server + 200k scene (5 min bench timeout default)
VITRUM_PR_START_SERVER=1 VITRUM_PR_SCENARIO=PR-hybrid-200k-static npm run benchmark:pr-hybrid --workspace @vitrum/benchmark-runner
```

`PR-hybrid-200k-static` uses `?scene=bench200k&targetTriangles=200000` on
`walkaround.html`. `PR-hybrid-tlas-10-inst` uses `?scene=tlas10inst&bvhMode=tlas`.

WG-0 pt-webgpu PNG capture:

```bash
VITRUM_OUTPUT_PNG=tools/reference-renders/baseline/ptwgpu-cornell.png \
  npm run benchmark:capture-pt-webgpu --workspace @vitrum/benchmark-runner
```

Set `VITRUM_GPU_CAPTURE=1` in CI only when Playwright + WebGPU are available (WG-0.4).

### WG-0.2 baseline seed + verify

`ptwgpu-parity-material-fields` uses the built-in `capturePtWebgpu.mjs` adapter when
`VITRUM_CAPTURE_CMD` is unset:

```bash
# One-shot seed (starts two-engines dev server, smoke resolution by default)
VITRUM_PR_START_SERVER=1 npm run benchmark:seed-wg0 --workspace @vitrum/benchmark-runner

# Verify against committed baseline PNG
VITRUM_GPU_CAPTURE=1 VITRUM_GAP_SCENARIOS=ptwgpu-parity-material-fields \
  npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

Baselines live under `tools/reference-renders/baseline/` (see README there).

Filter gap-closure to a subset: `VITRUM_GAP_SCENARIOS=id1,id2`.

## Wave 0 baseline orchestration

`run-wave0-baseline.mjs` captures the initial sweep baseline in one command:

```bash
npm run baseline:wave0
```

It executes a sequential baseline gate:

1. `npm run verify:mechanical`
2. gap-closure benchmark smoke pass
3. quality-modes benchmark smoke pass
4. optional quick reference capture (`VITRUM_WAVE0_CAPTURE_REFS=1`)

The quality-mode step is marked `warn` (non-blocking by default) when either:
- scenario rows fail (`summary.failures > 0`), or
- telemetry warmup never became ready (`warmupReady: false` rows).

It writes a timestamped report JSON to:

- `tools/benchmark-runner/results/wave0/wave0-baseline-<timestamp>.json`

Current Wave 0 schema id: `vitrum-wave0-baseline-2026-05-26`.

Useful env knobs:

- `VITRUM_WAVE0_MECHANICAL_TIMEOUT_MS`
- `VITRUM_WAVE0_GAP_TIMEOUT_MS`
- `VITRUM_WAVE0_QUALITY_TIMEOUT_MS`
- `VITRUM_WAVE0_CAPTURE_TIMEOUT_MS`
- `VITRUM_WAVE0_CAPTURE_REFS=1` (include `capture:refs:quick` step)

## Per-qualityMode benchmark

`run-quality-mode-bench.mjs` measures **frame time** and **SPP/sec** for each
`PTEngineWebGL2QualityMode` (`interactive` | `safe` | `final` | `capture`)
against one or more cornell-box scenarios.

The cornell-box example publishes live telemetry on `window.__vitrum.ptWebgl`
(`spp`, `lastFrameMs`, `frame`, `qualityMode`, `samplesTarget`, `isConverged`,
`sppPerSecond`, `renderWidth`, `renderHeight`). The benchmark drives the page
via Playwright, polls these every 100 ms for 30 s, and writes a JSON report
to `tools/benchmark-runner/results/quality-modes-<timestamp>.json`.

### Running

This script requires a live cornell-box dev server. In one terminal:

```bash
npm run dev --workspace @vitrum-examples/cornell-box
# default URL: http://127.0.0.1:5174/
```

In another terminal (typically your **main checkout**, not a worktree, to
avoid Playwright browser-version mismatches):

```bash
npm run benchmark:qualitymodes --workspace @vitrum/benchmark-runner
```

### Env knobs

- `VITRUM_CAPTURE_URL` — dev-server URL (default `http://127.0.0.1:5174/`)
- `VITRUM_BENCH_DURATION_MS` — poll window per (scenario, mode) (default `30000`)
- `VITRUM_BENCH_POLL_MS` — poll interval (default `100`)
- `VITRUM_BENCH_NAV_TIMEOUT_MS` — page navigation timeout (default `max(60000, 2×duration)`)
- `VITRUM_BENCH_WARMUP_TIMEOUT_MS` — max wait for first live telemetry before timed sampling starts (default `min(navTimeout, 30000)`)
- `VITRUM_BENCH_SAMPLES_TARGET` — `vitrumSpp` for the URL (default `128`)
- `VITRUM_BENCH_WIDTH` / `VITRUM_BENCH_HEIGHT` — render size (default `1280x720`)
- `VITRUM_BENCH_QUALITY_MODES` — comma-separated subset (default all 4)
- `VITRUM_BENCH_SCENARIOS` — comma-separated subset (default `cornell-box`)
- `VITRUM_BENCH_HEADLESS` — `0` for headed Chromium (default `1`)
- `VITRUM_BENCH_FAIL_FAST` — `1` to abort immediately on first scenario/mode failure
- `VITRUM_BENCH_STRICT` — `1` to exit non-zero if any scenario/mode row fails

### Worktree note

If your worktree has a Playwright browser version mismatch
(`Executable doesn't exist at .../headless_shell`), run the benchmark from
your main checkout where browsers are installed (`npx playwright install`).
The script writes its results into the worktree's
`tools/benchmark-runner/results/` directory regardless of which checkout it
runs in, because the path is resolved relative to the script file location.

### Output schema

```json
{
  "generatedAt": "2026-05-17T...",
  "schemaVersion": "quality-modes-bench-2026-05-17",
  "environment": { "platform": "linux", "node": "v20.x", ... },
  "qualityModes": ["interactive", "safe", "final", "capture"],
  "scenarios": ["cornell-box"],
  "results": [
    {
      "scenario": "cornell-box",
      "qualityMode": "interactive",
      "warmupMs": 1432,
      "warmupReady": true,
      "framesRendered": 142,
      "totalSpp": 128,
      "sppPerSec": 4.26,
      "meanFrameMs": 195.3,
      "p50FrameMs": 188.0,
      "p99FrameMs": 412.5,
      "converged": true,
      ...
    }
  ]
}
```

## Lifecycle soak benchmark

`run-lifecycle-soak.mjs` stress-tests repeated navigation/quality/size churn while
checking live telemetry progress (`window.__vitrum.ptWebgl`).

```bash
npm run benchmark:lifecycle-soak --workspace @vitrum/benchmark-runner
```

Output:

- `tools/benchmark-runner/results/lifecycle-soak-<timestamp>.json`

Current lifecycle soak schema id: `lifecycle-soak-2026-05-26`.

Useful knobs:

- `VITRUM_CAPTURE_URL` (default `http://127.0.0.1:5174/`)
- `VITRUM_LIFECYCLE_SOAK_ITERATIONS` (default `12`)
- `VITRUM_LIFECYCLE_SOAK_ITERATION_MS` (default `4000`)
- `VITRUM_LIFECYCLE_SOAK_POLL_MS` (default `100`)
- `VITRUM_LIFECYCLE_SOAK_NAV_TIMEOUT_MS` (default `90000`)
- `VITRUM_LIFECYCLE_SOAK_READY_TIMEOUT_MS` (default `60000`)
- `VITRUM_LIFECYCLE_SOAK_SPP` (default `256`)
- `VITRUM_LIFECYCLE_SOAK_SCENARIOS` (comma-separated, default `cornell-box`)
- `VITRUM_LIFECYCLE_SOAK_QUALITY_MODES` (comma-separated, default `interactive,safe,final,capture`)
- `VITRUM_LIFECYCLE_SOAK_STRICT=1` (exit non-zero when any iteration fails)
- `VITRUM_LIFECYCLE_SOAK_START_SERVER=1` (launch local dev server before soak)
- `VITRUM_LIFECYCLE_SOAK_DEV_CMD` (override dev-server launch command)
- `VITRUM_LIFECYCLE_SOAK_SERVER_READY_TIMEOUT_MS` (dev-server readiness timeout)
- `VITRUM_LIFECYCLE_SOAK_SERVER_POLL_MS` (dev-server readiness poll interval)

When server auto-start is enabled and Vite moves to a fallback port (for example
`5174` in use -> `5175`), the runner auto-detects the advertised local URL and
targets that URL for the soak iterations.

## Wave 4 hardening orchestration

`run-wave4-hardening.mjs` is the reliability gate bundle for this sweep:

1. `verify:mechanical` (unless skipped)
2. strict lifecycle soak (`benchmark:lifecycle-soak` with strict mode)
3. optional quality-mode smoke benchmark

```bash
npm run hardening:wave4
```

Output:

- `tools/benchmark-runner/results/wave4/wave4-hardening-<timestamp>.json`

Current Wave 4 schema id: `vitrum-wave4-hardening-2026-05-26`.

Useful knobs:

- `VITRUM_WAVE4_SKIP_MECHANICAL=1`
- `VITRUM_WAVE4_INCLUDE_QUALITY_SMOKE=1`
- `VITRUM_WAVE4_MECHANICAL_TIMEOUT_MS`
- `VITRUM_WAVE4_SOAK_TIMEOUT_MS`
- `VITRUM_WAVE4_QUALITY_TIMEOUT_MS`
- `VITRUM_WAVE4_STRICT=1` (treat warnings as failing gate)

## GPU capture mode

The runner is fail-closed by default. To execute real image/perf captures, set:

- `VITRUM_GPU_CAPTURE=1`
- `VITRUM_CAPTURE_CMD="<your capture command>"`

Optional:

- `VITRUM_BASELINE_DIR=tools/reference-renders/baseline`
- `VITRUM_ALLOW_BASELINE_GEN=1` (auto-generate missing baselines)
- `VITRUM_FAIL_ON_IDENTICAL_HASH=1` (strictly fail if before/after hashes match)
- `VITRUM_CAPTURE_SMOKE=1` (cap scenarios for local sanity checks; defaults to
  320×180, 8 SPP, 4 bounces)
- `VITRUM_CAPTURE_PROCESS_TIMEOUT_MS=120000` (hard-kill a stuck browser capture)

When a baseline capture emits `{"msPerSample": number}` on stdout, the runner stores
it beside the baseline PNG as `<scenario>.png.json` and reuses it as
`perfBaselineMsPerSample` on later runs.

- `VITRUM_STRICT_GAP_CLOSURE=1` — exit with code `1` if any scenario is not `PASS`
  (use after baselines exist and captures succeed).

The capture adapter command receives scenario parameters via env vars:

- `VITRUM_SCENARIO_ID`
- `VITRUM_SEED`
- `VITRUM_WIDTH`, `VITRUM_HEIGHT`
- `VITRUM_BOUNCES`
- `VITRUM_SPP`
- `VITRUM_CAUSTIC_STRATEGY` (`none`, `manifold-nee`, `photon-map`, etc.)
- `VITRUM_OUTPUT_PNG` (required output image path)

Adapter contract:

1. Write PNG to `VITRUM_OUTPUT_PNG`.
2. Exit with code `0` on success.
3. Optional: print one-line JSON including `msPerSample`, e.g.:
   `{"msPerSample":0.42}`.

### Acceptance metrics artifacts

Gated GPU acceptance tests consume harness-produced JSON metrics files:

- `packages/walkaround-hybrid/__tests__/rcAcceptance.gpu.test.ts`
  reads `VITRUM_RC_ACCEPTANCE_METRICS`.
- `packages/walkaround-hybrid/__tests__/neuralAcceptance.test.ts`
  reads `VITRUM_NEURAL_ACCEPTANCE_METRICS`.
- `packages/walkaround-rc/__tests__/rcBehavior.gpu.test.ts`
  reads `VITRUM_RC_BEHAVIOR_METRICS`.
- `packages/pt-webgpu/src/__tests__/tlasPromotionAcceptance.test.ts`
  reads `VITRUM_PTWGPU_TLAS_METRICS`.

Each file is expected to contain numeric fields documented in the test itself.
For PT-WebGPU TLAS promotion specifically, the metrics JSON includes
`tlasVsLegacyMeanAbs`, `tlasVsLegacyP95Abs`, and `tlasVsLegacyMaxAbs`.
Current schema id: `ptwgpu-tlas-metrics-2026-05-25`.
It also includes `orderedStats`, per-threshold `pass` booleans, and capture
metadata (`imageWidth`, `imageHeight`, `roi`) to make CI failures self-diagnosing.
The acceptance test thresholds are controlled via:

- `VITRUM_PTWGPU_TLAS_MAX_DELTA` (mean, default `0.02`)
- `VITRUM_PTWGPU_TLAS_MAX_P95_DELTA` (p95, default `0.06`)
- `VITRUM_PTWGPU_TLAS_MAX_PEAK_DELTA` (max, default `0.2`)
- `VITRUM_PTWGPU_TLAS_STRICT=1` (benchmark runner exits non-zero if TLAS metrics exceed those thresholds)

You can generate these JSON artifacts from captured PNG pairs:

```bash
VITRUM_RC_OFF_PNG=tools/reference-renders/W8-rc-off.png \
VITRUM_RC_ON_PNG=tools/reference-renders/W8-rc-on.png \
VITRUM_NEURAL_ATROUS_PNG=tools/reference-renders/neural-atrous.png \
VITRUM_NEURAL_PNG=tools/reference-renders/neural.png \
VITRUM_PTWGPU_LEGACY_PNG=tools/reference-renders/ptwgpu-legacy.png \
VITRUM_PTWGPU_TLAS_PNG=tools/reference-renders/ptwgpu-tlas.png \
npm run benchmark:acceptance-metrics --workspace @vitrum/benchmark-runner
```

The command prints ready-to-export paths:

- `VITRUM_RC_ACCEPTANCE_METRICS=...`
- `VITRUM_RC_BEHAVIOR_METRICS=...`
- `VITRUM_NEURAL_ACCEPTANCE_METRICS=...`
- `VITRUM_PTWGPU_TLAS_METRICS=...`

### PT-WebGL fidelity acceptance artifact

`benchmark:pt-webgl-fidelity` scans paired PNGs under
`tools/reference-renders/pt-webgl-fidelity/`:

- `<scenario>.baseline.png`
- `<scenario>.candidate.png`

It computes RGB PSNR + mean absolute delta and writes a JSON artifact consumed by
`packages/pt-webgl/src/__tests__/fidelityAcceptance.test.ts`.

```bash
npm run benchmark:pt-webgl-fidelity --workspace @vitrum/benchmark-runner
```

The command prints:

- `VITRUM_PTWEBGL_FIDELITY_METRICS=...`

Useful knobs:

- `VITRUM_PTWEBGL_FIDELITY_REQUIRED` (comma-separated scenario IDs; defaults derive from `scenario-presets.mjs` RFE scenarios)
- `VITRUM_PTWEBGL_FIDELITY_MIN_PSNR` (global threshold, default `28`)
- `VITRUM_PTWEBGL_FIDELITY_MIN_PSNR_BY_SCENARIO` (JSON object, e.g. `{"rfe05-caustic-strategy":26}`)
- `VITRUM_PTWEBGL_FIDELITY_STRICT=1` (exit non-zero when required scenarios are missing or any row fails threshold)

Example (Playwright adapter in this folder):

```bash
VITRUM_GPU_CAPTURE=1 \
VITRUM_CAPTURE_SMOKE=1 \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

For formal acceptance captures, omit `VITRUM_CAPTURE_SMOKE=1` and run on a
machine/browser profile with confirmed GPU acceleration. The full scenario matrix
uses 1280×720 / 512–1024 SPP and can exhaust software WebGL implementations.

## Scenario preset registry

`scenario-presets.mjs` exports `GAP_CLOSURE_SCENARIOS` (mirrors
`plan/gap-closure-acceptance-matrix.md`). Host capture pages can read
`vitrumScenario`, `vitrumSeed`, `vitrumWidth`, `vitrumHeight`, `vitrumBounces`,
`vitrumSpp`, and `vitrumCaustic` query parameters appended by
`capture-adapter-playwright.mjs`.

## Fork shader regression (no GPU)

From the vitrum repo root:

```bash
npm run fork-shader-smoke
```

## Sweep verification capture

This section documents the end-to-end procedure for generating, comparing, and adopting
GPU reference renders for the 11 sweep-2026-05-11 verification scenarios defined in
`scenario-presets.mjs` (scenarioIds: `m5-*`, `m7-*`, `m8-*`, `m9-*`, `m17-*`, `m18-*`).

The diff-report template lives at
`tools/reference-renders/sweep-2026-05-11-diff-report.md`. Fill it in as you go.

### 1 — Capture pre-sweep baselines (branch: `main`)

```bash
git checkout main

VITRUM_GPU_CAPTURE=1 \
VITRUM_ALLOW_BASELINE_GEN=1 \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

Move the generated PNGs to a side-car directory so they survive the post-sweep run:

```bash
for f in tools/reference-renders/baseline/m5-* \
          tools/reference-renders/baseline/m7-* \
          tools/reference-renders/baseline/m8-* \
          tools/reference-renders/baseline/m9-* \
          tools/reference-renders/baseline/m17-* \
          tools/reference-renders/baseline/m18-*; do
  cp "$f" "${f%.png}.pre-sweep.png"
done
```

Notes:
- Omit `VITRUM_CAPTURE_SMOKE=1` for formal captures. The new scenarios use
  1280×720, so a software-renderer fallback will be slow and likely incorrect.
- Run on a machine/browser profile with confirmed GPU acceleration
  (Chrome with `chrome://gpu` showing WebGL/WebGPU hardware-backed).
- The `wallAlbedoVariants`, `roughnessVariants`, and `causticVariants` arrays
  in a scenario cause the runner to iterate and set the corresponding env var
  (`VITRUM_WALL_ALBEDO`, `VITRUM_ROUGHNESS`, `VITRUM_CAUSTIC_STRATEGY`) for each
  sub-capture. Check that the host capture page reads these query params.
- Walkaround backend scenarios (`backend: 'walkaround'`) require the WebGPU
  path; Chrome 120+ on a discrete GPU recommended.

### 2 — Capture post-sweep candidates (branch: `feat/sweep-2026-05-11-fixes`)

```bash
git checkout feat/sweep-2026-05-11-fixes

VITRUM_GPU_CAPTURE=1 \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

The runner will compare each candidate against the baseline written in step 1.
Move the candidate PNGs to a side-car as well:

```bash
for f in tools/reference-renders/baseline/m5-* \
          tools/reference-renders/baseline/m7-* \
          tools/reference-renders/baseline/m8-* \
          tools/reference-renders/baseline/m9-* \
          tools/reference-renders/baseline/m17-* \
          tools/reference-renders/baseline/m18-*; do
  [[ "$f" == *.pre-sweep.png ]] && continue
  cp "$f" "${f%.png}.post-sweep.png"
done
```

### 3 — Populate the diff-report template

For each of the 11 scenarios in `sweep-2026-05-11-diff-report.md`:

1. Compute PSNR between the `.pre-sweep.png` and `.post-sweep.png` pair using
   any image-diff tool (e.g., `magick compare -metric PSNR pre.png post.png`).
2. Compute mean luminance of each image (e.g., `magick identify -verbose img.png
   | grep Mean`).
3. Record both values in the Pre-sweep / Post-sweep / Δ columns.
4. A/B the images side-by-side and confirm the directional change described
   in the **Expected change** field. Mark **Visual sign-off** as ☑ when satisfied.

Acceptance criteria (per plan/archive/sweep-2026-05-12-followup.md § Phase A4):
- Visual changes must be in the direction the math predicts (see each scenario's
  **Expected change** field).
- No new artifacts introduced that were not present in the pre-sweep image.
- PSNR regression from a previously-passing scenario is only acceptable if the
  diff is visually justified (e.g., physically correct darkening in a corner).

### 4 — Handling unexpected regressions

If a scenario shows a change in the wrong direction or introduces an artifact:

1. **First: verify it is not a capture artifact.** Re-run the capture with a
   higher SPP or more frames. MC variance can mimic systematic bias at low
   sample counts.
2. **If reproducible:** check the scenario parameters against the fix that item
   was meant to verify. The inline comment in `scenario-presets.mjs` cites the
   specific sweep item and algorithm. Read that code path end-to-end (do not
   rely on sub-agent summaries — see CLAUDE.md).
3. **If the code is wrong:** open a defect in the plan doc for the relevant
   milestone (e.g., `plan/sprint-M5-*.md`) and do not adopt the post-sweep PNG
   as a baseline until the defect is resolved.
4. **If the code is correct but the scenario is wrong:** revise the scenario
   (seed, framing, or parameters) so that it exercises the intended code path
   cleanly, re-capture, and document the revision in this README and in the
   diff-report.
5. **Do not revert the sweep fix** without understanding why the mathematical
   prediction did not hold. Regressions in rendering are almost always bugs in
   the measurement, not the algorithm.

### 5 — Adopt post-sweep PNGs as new baselines

After all 11 scenarios have passed visual sign-off:

```bash
for f in tools/reference-renders/baseline/m5-* \
          tools/reference-renders/baseline/m7-* \
          tools/reference-renders/baseline/m8-* \
          tools/reference-renders/baseline/m9-* \
          tools/reference-renders/baseline/m17-* \
          tools/reference-renders/baseline/m18-*; do
  [[ "$f" == *.pre-sweep.png || "$f" == *.post-sweep.png ]] && continue
  # The runner already wrote the candidate over the baseline path; nothing
  # to move. Delete the side-cars once satisfied.
  rm "${f%.png}.pre-sweep.png" "${f%.png}.post-sweep.png" 2>/dev/null || true
done
```

Future gap-closure runs will diff against the adopted post-sweep baselines.

**Strict-mode gate:** these scenarios are advisory-only until baselines are
stable (~2 capture cycles). Do not set `VITRUM_STRICT_GAP_CLOSURE=1` for the
sweep scenarios until both a pre- and post-sweep capture have completed and the
diff-report has been signed off.
