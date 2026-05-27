# W8 — Radiance Cascades reference renders (rcEnabled: true)

Cornell-box captures from HybridEngine with `rcEnabled: true` and `rcWeight: 1` (maximum RC contribution in the shade MIS path).

Pairs with `../W8-rc-off/`. See that README for `npm run benchmark:rc-acceptance` and `rcAcceptance.gpu.test.ts` wiring.

## Expected artifact

- `cornell-walkaround-rc-on.png`

PNG is written by the benchmark runner on a hybrid-capable GPU host; commit after visual review when refreshing baselines.
