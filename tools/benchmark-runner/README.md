# @vitrum/benchmark-runner

This workspace contains executable benchmark and hardening runners (`run-*.mjs`)
for gap-closure verification, RC acceptance checks, and reference-render capture.

## Live scripts

The following scripts exist in `package.json` and can be invoked directly:

| Script | Command |
|--------|---------|
| `typecheck` | `node --check ./*.mjs` (syntax check all runners) |
| `benchmark` | `node ./run-gap-closure-verification.mjs` |
| `benchmark:gap-closure` | `node ./run-gap-closure-verification.mjs` |
| `benchmark:gap-closure-mechanical` | Gap-closure run with `VITRUM_GAP_MECHANICAL=1` |
| `benchmark:acceptance-metrics` | `node ./run-acceptance-metrics.mjs` |
| `benchmark:rc-acceptance` | `node ./run-rc-acceptance.mjs` |
| `benchmark:rc-acceptance-mechanical` | `node ./run-rc-acceptance-mechanical.mjs` |
| `benchmark:rc-behavior-mechanical` | `node ./run-rc-behavior-mechanical.mjs` |
| `write-rc-mechanical-fixtures` | `node ./write-rc-mechanical-fixtures.mjs` |

To run from the repo root:

```bash
npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
npm run benchmark:acceptance-metrics --workspace @vitrum/benchmark-runner
npm run benchmark:rc-acceptance-mechanical --workspace @vitrum/benchmark-runner
npm run typecheck --workspace @vitrum/benchmark-runner
npm test --workspace @vitrum/benchmark-runner
```

## GPU capture mode

The runner is fail-closed by default. To enable real image/perf captures:

1. Start a serving example in one terminal, e.g.:
   ```bash
   npm run dev --workspace @vitrum-examples/pt-webgl2-direct
   ```
2. Run the gap-closure verifier in another terminal:
   ```bash
   VITRUM_GPU_CAPTURE=1 \
   VITRUM_CAPTURE_CMD="node ./tools/benchmark-runner/capture-adapter-playwright.mjs" \
   VITRUM_CAPTURE_URL="http://127.0.0.1:5173/" \
   npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner
   ```

The capture adapter (`capture-adapter-playwright.mjs`) launches Playwright Chromium
with WebGPU flags (`--enable-unsafe-webgpu`, `--use-angle=vulkan` on Linux) and
screenshots the canvas after `globalThis.VITRUM_CAPTURE_READY === true` is set.

Optional env vars:

- `VITRUM_BASELINE_DIR` — path to baseline PNGs (default `tools/reference-renders/baseline`)
- `VITRUM_ALLOW_BASELINE_GEN=1` — auto-generate missing baselines
- `VITRUM_FAIL_ON_IDENTICAL_HASH=1` — fail if before/after hashes match
- `VITRUM_CAPTURE_SMOKE=1` — cap to 320×180 / 8 SPP / 4 bounces for local sanity
- `VITRUM_CAPTURE_PROCESS_TIMEOUT_MS` — hard-kill timeout (default 120 000 ms)
- `VITRUM_STRICT_GAP_CLOSURE=1` — exit non-zero if any scenario is not PASS
- `VITRUM_GAP_SCENARIOS=id1,id2` — filter to a comma-separated subset of scenario IDs

The adapter contract: write PNG to `VITRUM_OUTPUT_PNG`, exit 0 on success, optionally
print `{"msPerSample": <number>}` on stdout for perf telemetry.

## Gap-closure scenario registry

`scenario-presets.mjs` exports `GAP_CLOSURE_SCENARIOS` — the deterministic
metadata for every gap-closure scenario. Host capture pages read the
`vitrum*` query params that `capture-adapter-playwright.mjs` injects automatically:

| Param | Env source |
|-------|-----------|
| `vitrumScenario` | `VITRUM_SCENARIO_ID` |
| `vitrumSeed` | `VITRUM_SEED` |
| `vitrumWidth`, `vitrumHeight` | `VITRUM_WIDTH`, `VITRUM_HEIGHT` |
| `vitrumBounces` | `VITRUM_BOUNCES` |
| `vitrumSpp` | `VITRUM_SPP` |
| `vitrumCaustic` | `VITRUM_CAUSTIC_STRATEGY` |
| `vitrumBackend` | `VITRUM_BACKEND` |
| `vitrumAutoStart` | (always set to `1`) |

## Acceptance metrics artifacts

Gated GPU acceptance tests consume harness-produced JSON metrics files:

- `packages/walkaround-hybrid/__tests__/rcAcceptance.gpu.test.ts`
  reads `VITRUM_RC_ACCEPTANCE_METRICS`.
- `packages/walkaround-hybrid/__tests__/neuralAcceptance.test.ts`
  reads `VITRUM_NEURAL_ACCEPTANCE_METRICS`.
- `packages/walkaround-rc/__tests__/rcBehavior.gpu.test.ts`
  reads `VITRUM_RC_BEHAVIOR_METRICS`.
- `packages/pt-webgpu/src/__tests__/tlasPromotionAcceptance.test.ts`
  reads `VITRUM_PTWGPU_TLAS_METRICS`.

Generate them from captured PNG pairs:

```bash
VITRUM_RC_OFF_PNG=tools/reference-renders/W8-rc-off.png \
VITRUM_RC_ON_PNG=tools/reference-renders/W8-rc-on.png \
VITRUM_NEURAL_ATROUS_PNG=tools/reference-renders/neural-atrous.png \
VITRUM_NEURAL_PNG=tools/reference-renders/neural.png \
VITRUM_PTWGPU_LEGACY_PNG=tools/reference-renders/ptwgpu-legacy.png \
VITRUM_PTWGPU_TLAS_PNG=tools/reference-renders/ptwgpu-tlas.png \
npm run benchmark:acceptance-metrics --workspace @vitrum/benchmark-runner
```

## Mechanical RC acceptance gates (no GPU)

```bash
npm run benchmark:rc-acceptance-mechanical --workspace @vitrum/benchmark-runner
npm run benchmark:rc-behavior-mechanical --workspace @vitrum/benchmark-runner
```

These run without GPU and pin harness contracts using committed fixture PNGs under
`tools/reference-renders/W8-rc-{off,on}/` and `tools/reference-renders/bdpt-layered-mechanical/`.
Replace with real GPU captures to refresh baselines.

## Capture protocol (for example apps)

Any example app used as a capture target must set these globals after reaching
a deterministic converged state:

```ts
globalThis.VITRUM_CAPTURE_READY = true;          // sentinel Playwright awaits
globalThis.VITRUM_MS_PER_SAMPLE = 12.4;          // perf telemetry (optional)
globalThis.VITRUM_CAPTURE_TELEMETRY = {/*…*/};   // arbitrary JSON sidecar (optional)
globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR = '#c'; // which canvas to screenshot
```

And honour these URL params:
- `?vitrumScenario=<id>` — flip into capture mode
- `?vitrumAutoStart=1` — skip any "click to start" gate
- `?vitrumWidth`, `?vitrumHeight` — canvas size lock
- `?vitrumSpp`, `?vitrumBounces` — quality knobs

The H57 example apps under `examples/` implement this protocol. See
`tools/reference-renders/README.md` for the full A/B workflow.
