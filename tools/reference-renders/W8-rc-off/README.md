# W8 — Radiance Cascades reference renders (rcEnabled: false baseline)

Cornell-box captures from HybridEngine with `rcEnabled: false` (ReSTIR-GI indirect only).

Pairs with `../W8-rc-on/` (`rcEnabled: true`, `rcWeight: 1`).

## Capture

```bash
npm run benchmark:rc-acceptance
```

Writes:

- `cornell-walkaround-rc-off.png` — this directory
- `cornell-walkaround-rc-on.png` — `W8-rc-on/`

Metrics JSON under `tools/benchmark-runner/results/acceptance/`. Run the gated vitest with:

```bash
VITRUM_RC_ACCEPTANCE=1 \
  VITRUM_RC_ACCEPTANCE_METRICS=tools/benchmark-runner/results/acceptance/<rc-acceptance-metrics>.json \
  npm test --workspace @vitrum/walkaround-hybrid -- rcAcceptance.gpu
```

Env: `VITRUM_RC_CAPTURE_FRAMES` (default 48), `VITRUM_RC_SEED` (default 1701), `VITRUM_RC_REQUIRE_GPU=1` to fail when adapter is insufficient.

## Mechanical fixtures (CI)

Committed `cornell-walkaround-rc-off.png` (64×64) is a **metrics-oracle stub** until a GPU host replaces it with a real capture. Regenerate stubs:

```bash
npm run write-rc-mechanical-fixtures --workspace @vitrum/benchmark-runner
npm run benchmark:rc-acceptance-mechanical
```

## Status

Harness shipped 2026-05-27. Replace fixture PNGs with GPU captures when refreshing baselines (`npm run benchmark:rc-acceptance` on a hybrid-capable adapter).
