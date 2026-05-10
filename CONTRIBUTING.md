# Contributing to vitrum

## Commands

- `npm run typecheck` — TypeScript across workspaces.
- `npm test` — Vitest where configured.
- `npm run fork-shader-smoke` — Shader string regression on the sibling
  `three-gpu-pathtracer` checkout (`../three-gpu-pathtracer`, or `VITRUM_FORK_DIR`).

## GPU / reference renders

- Gap-closure harness: `tools/benchmark-runner` (`npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner`).
- Reference PNG baselines: `tools/reference-renders/baseline/` (not checked in
  until generated; use `VITRUM_ALLOW_BASELINE_GEN=1` when captures are enabled).

## Pull requests

- Do not claim **supported** in READMEs without tests and (where applicable)
  capture artifacts — use **approximate** / **experimental** per
  `plan/renderer-fidelity-matrix.md`.
