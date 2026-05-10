# @vitrum/benchmark-runner

Phase 6 sprint deliverables drop benchmark scripts here. Each benchmark file is
named `sprint-<N>-<name>.ts` and writes its before/after metrics to
`tools/benchmark-runner/results/sprint-<N>.json`.

See `plan/phase-6-roadmap.md` Section 9 (Verification protocol) for the
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

When a baseline capture emits `{"msPerSample": number}` on stdout, the runner stores
it beside the baseline PNG as `<scenario>.png.json` and reuses it as
`perfBaselineMsPerSample` on later runs.

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
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

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
