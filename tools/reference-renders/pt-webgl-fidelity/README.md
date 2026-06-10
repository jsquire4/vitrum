# pt-webgl-fidelity — Archive Baselines

**Status: ARCHIVE — no active capture path.**

These baselines were captured from the deleted `@vitrum/pt-webgl` fork backend
(the `three-gpu-pathtracer` fork) prior to commit `e14000c` (THREE removal,
2026-06-09). They are preserved as the **sole THREE-cutover A/B reference**
for the fork-vs-native comparison: any future host can diff a new pt-webgl2
capture against these to verify numeric fidelity between the fork and the
native THREE-free port.

## Files

Each file is a `<scenarioId>.<variant>.{baseline,candidate}.png` pair captured
during the pre-cutover verification sweep:

| File | Scene | Notes |
|------|-------|-------|
| `rfe03-layered-front-back.{baseline,candidate}.png` | Layered BSDF | Fork vs fork-head |
| `rfe05-caustic-strategy.manifold-nee.{baseline,candidate}.png` | Caustic heuristic | manifold-nee variant |
| `rfe07-11-sss-mixed-panels.{baseline,candidate}.png` | SSS mixed | |
| `rfe08-13-spectral-payload.{baseline,candidate}.png` | Spectral/spectral | |
| `rfe09-bridge-global-cmf.{baseline,candidate}.png` | Global CMF bridge | |
| `rfe14-thinfilm-angle-shift.{baseline,candidate}.png` | Thin-film angle | |

## No Active Capture Path

There is no `examples/` app, no npm script, and no scenario preset that targets
this directory. The `@vitrum/pt-webgl` package has been removed and cannot be
reinstantiated. To use these as a reference:

1. Locate your pt-webgl2 native capture for the same scenario (in
   `tools/reference-renders/baseline/`).
2. Run `node tools/reference-renders/diff-baselines.mjs` to compare pixel-by-pixel.

## Decision record

Kept per decision D18 (h-remediation-plan-2026-06-09.md §1): "archive-relabel
(sole cutover A/B reference)". The alternative (delete) was rejected because
these are the only surviving pre-cutover fork renders and provide the radiometric
ground truth for the fork-vs-native fidelity claim.
