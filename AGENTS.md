# AGENTS.md

## Cursor Cloud specific instructions

### Overview

vitrum is a WebGPU + WebGL2 path tracing/GI engine. It's an npm workspaces monorepo with no build step — packages export raw TypeScript source. See `CLAUDE.md` for full architecture and `README.md` for the public API surface.

### Key commands

| Task | Command |
|------|---------|
| Install deps | `npm install` (from `/workspace`) |
| Typecheck | `npm run typecheck` |
| Test (node-only) | `npm run test --workspaces --if-present` |
| Lint | `npm run lint` |
| Format check | `npm run format:check` |
| Dev server (any example) | `npm run dev` from `examples/<name>/` |

### Sibling fork dependency

The `three-gpu-pathtracer` fork must exist at `/three-gpu-pathtracer` (one level above `/workspace`). The update script clones it if absent. Packages `pt-webgl` and all examples reference it via `"file:../../../three-gpu-pathtracer"`.

### GPU tests

`pt-webgpu` and `shared-denoisers` have test scripts that chain node tests + GPU browser tests (`vitest run && vitest run --config vitest.gpu.config.ts`). The GPU browser tests require Playwright Chromium and a GPU-capable browser. In headless cloud VMs without GPU hardware:

- Use `npm run test:node` per-package to run only the node-side tests (always pass).
- The default `npm test` will exit non-zero from these two packages due to the GPU test phase. All other packages pass cleanly.

### Lint pre-existing state

`npm run lint` reports ~170+ pre-existing errors (mostly `@typescript-eslint` type-safety issues in `pt-webgl`, `walkaround-hybrid`, and `shader-compile-ci`). These are not regressions — do not attempt to fix them unless specifically asked.

### WebGL2/WebGPU rendering

The example apps require real GPU hardware for rendering. In cloud VMs, the Vite dev server will start and serve the HTML/JS/CSS, but actual path tracing or GI rendering will fail with "Error creating WebGL2 context" or similar WebGPU adapter errors. This is expected.

### No build step

There is no `build` command needed for development. All packages export raw `.ts` source (`"main": "src/index.ts"`). Only `vite build` in examples produces bundles (for deployment, not required during dev).
