# Reference renders

Baseline PNGs for GPU verification and gap-closure runs live under `baseline/`.

## Layout

- **`baseline/<scenarioId>-<variantId>.png`** — canonical baseline image per
  scenario variant (`candidate` when no axis variants exist).
- **`baseline/<scenarioId>.png`** — compatibility alias for
  `<scenarioId>-candidate.png`.
  Expected scenarios come from
  [tools/benchmark-runner/scenario-presets.mjs](../benchmark-runner/scenario-presets.mjs)
  (mirrors [plan/gap-closure-acceptance-matrix.md](../../plan/gap-closure-acceptance-matrix.md)).
- **`<scenarioId>.png.json`** (optional) — perf sidecar written by the benchmark
  runner when the capture adapter prints `{"msPerSample": ...}` on stdout.

## Generating baselines

From the repo root, with a working GPU capture adapter:

```bash
VITRUM_GPU_CAPTURE=1 \
VITRUM_ALLOW_BASELINE_GEN=1 \
VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
VITRUM_CAPTURE_URL="http://127.0.0.1:5174/" \
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
```

Use `VITRUM_CAPTURE_SMOKE=1` for smaller resolutions during local smoke runs.

## Strict verification

Set `VITRUM_STRICT_GAP_CLOSURE=1` when running the gap-closure script so the
process exits non-zero if any scenario is not `PASS` (requires baselines and,
when capture is enabled, successful candidate captures).
