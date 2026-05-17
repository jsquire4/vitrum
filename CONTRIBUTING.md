# Contributing to vitrum

## Commands

- `npm run typecheck` — TypeScript across workspaces.
- `npm test` — Vitest where configured (node tests only; fast, no browser).
- `npm run test:gpu` — Vitest in headless Chromium via Playwright for the
  packages with `vitest.gpu.config.ts` (`@vitrum/pt-webgpu`,
  `@vitrum/shared-denoisers`). **Requires Playwright browsers to be
  installed** — run `npx playwright install chromium` once. Skipped from
  the default `npm test` because (a) Playwright browsers are a 200MB
  install most contributors won't have, and (b) the GPU tests need a
  Vulkan/SwiftShader stack that's environment-dependent.
- `npm run test:all` — Both node + GPU passes per package.
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
