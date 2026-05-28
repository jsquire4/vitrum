# Contributing to vitrum

## Commands

- `npm run typecheck` — TypeScript across workspaces.
- `npm test` — Vitest node tests across workspaces. **Does not require
  Playwright** — GPU-browser tests are opt-in (see below) so a fresh clone
  passes without the 200 MB headless-Chromium download.
- `npm run test:gpu` — Vitest GPU-browser tests only (currently
  `@vitrum/pt-webgpu` + `@vitrum/shared-denoisers`). Requires
  `npx playwright install chromium`; falls back to SwiftShader on machines
  without a real WebGPU adapter (see each package's
  `vitest.gpu.config.ts` for the Chromium flag set).
- `npm run test:all` — `test` then `test:gpu`. Use this for full local
  validation before pushing.
- `npm run fork-shader-smoke` — Shader-string regression on the absorbed fork at
  `packages/three-gpu-pathtracer` (no sibling checkout needed).

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
