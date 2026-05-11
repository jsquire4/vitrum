# Contributing to vitrum

## Commands

- `npm run typecheck` — TypeScript across workspaces.
- `npm test` — Vitest where configured.
- `npm run fork-shader-smoke` — Shader string regression on the sibling
  `three-gpu-pathtracer` checkout (`../three-gpu-pathtracer`, or `VITRUM_FORK_DIR`).

## GPU / reference renders

- Gap-closure harness: `npm run benchmark:gap-closure` (runs
  `@vitrum/benchmark-runner`). See [tools/benchmark-runner/README.md](tools/benchmark-runner/README.md).
- Reference PNG baselines: `tools/reference-renders/baseline/` — see
  [tools/reference-renders/README.md](tools/reference-renders/README.md).
- Strict gate (exit non-zero unless every scenario is `PASS`):
  `VITRUM_STRICT_GAP_CLOSURE=1` (requires committed baselines and, when
  `VITRUM_GPU_CAPTURE=1`, a working `VITRUM_CAPTURE_CMD`).

## Pull requests

- Do not claim **supported** in READMEs without tests and (where applicable)
  capture artifacts — use **approximate** / **experimental** per
  `plan/renderer-fidelity-matrix.md`.
