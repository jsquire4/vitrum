# @vitrum/benchmark-runner

Phase 6 sprint deliverables drop benchmark scripts here. Each benchmark file is
named `sprint-<N>-<name>.ts` and writes its before/after metrics to
`tools/benchmark-runner/results/sprint-<N>.json`.

See `plan/archive/phase-6-roadmap.md` Section 9 (Verification protocol) for the
benchmark file template.

## Current runner

- `npm run benchmark --workspace @vitrum/benchmark-runner`
- `npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner`

These commands execute `run-gap-closure-verification.mjs` and write:

- `tools/benchmark-runner/results/gap-closure-verification-2026-05-10.json`

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

From the vitrum repo root (with sibling `three-gpu-pathtracer` checkout):

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

Acceptance criteria (per plan/sweep-2026-05-12-followup.md § Phase A4):
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
