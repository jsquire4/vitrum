# Deep Remediation Pass Manifest

Date: 2026-05-21
Plan: `deep-remediation-pass_ffb87f35.plan.md`

## Wave A — Contracts + G-buffer correctness

- `packages/engine/src/createEngine.ts`
  - Added backend/denoiser compatibility validation before backend construction.
  - Enforced capability-honest proxy exposure for incremental scene methods.
- `packages/walkaround-hybrid/src/pipeline/denoisers/svgfReal.ts`
  - Added depth extraction pass (`gNormalDepth.w -> r32float`) for SVGF reprojection depth inputs.
- `packages/walkaround-hybrid/src/pipeline/resourceManager.ts`
  - Added `svgfDepthTextureA/B` persistent depth ping-pong textures.
- `packages/core/src/engine.ts`
  - Clarified denoiser support is backend-specific.
- Tests:
  - `packages/engine/__tests__/createEngine.test.ts`
  - `packages/engine/__tests__/telemetryProxy.test.ts`
  - `packages/walkaround-hybrid/__tests__/frameResourcesShape.test.ts`

Evidence:
- `npm run typecheck` (workspace): PASS
- `npm run test` (`packages/engine`): PASS
- `npm run typecheck` (`packages/walkaround-hybrid`): PASS
- `npm run test` (`packages/walkaround-hybrid`): PASS

## Wave B — Engine realtime parity + runtime alignment

- `packages/engine/src/createEngine.ts`
  - Requests required WebGPU limits/features for walkaround backend.
  - Added swap-chain auto-wiring (configure + resize + frame injection of `swapChainView` / `swapChainFormat`) when caller does not provide explicit swap-chain fields.
  - Uses canvas CSS size × DPR for initial physical resolution.
- `examples/two-engines-one-scene/src/main.ts`
  - Added `vitrumBackend` -> `mode` compatibility mapping.
- `tools/benchmark-runner/capture-adapter-playwright.mjs`
  - Mirrors `VITRUM_BACKEND` into `mode` query for two-engines host compatibility.

Evidence:
- `npm run test` (`packages/engine`): PASS
- `npm run typecheck` (`examples/two-engines-one-scene`): PASS

## Wave C — `_staging` deterministic rehabilitation

- Introduced deterministic `_staging` harness:
  - `_staging/tsconfig.typecheck.json`
  - root script: `typecheck:staging`
- Rehabilitated listed staging files to compile deterministically as compatibility shims:
  - `_staging/legacy-source/src/rendering/scene/lightingIntensityTable.ts`
  - `_staging/legacy-source/src/rendering/scene/ptEnvironment.ts`
  - `_staging/legacy-source/src/rendering/scene/lighting/renderers/sunPathTraced.tsx`
  - `_staging/legacy-source/src/rendering/scene/walkaround/HybridLayeredStage.tsx`
  - `_staging/legacy-source/src/rendering/scene/walkaround/engines/restir/RestirStage.tsx`
  - `_staging/legacy-source/src/rendering/scene/walkaround/lib/useSceneBVH.ts` (single canonical copy retained)

Evidence:
- `npm run typecheck:staging`: PASS

## Wave D — Benchmark capture + baseline contract

- Normalized host/capture port contract to `5174`:
  - `examples/cornell-box/vite.config.ts` (`strictPort: true`)
  - `tools/benchmark-runner/capture-adapter-playwright.mjs` default URL
  - `scripts/capture-cornell-suite.sh`
  - `tools/reference-renders/README.md`
- `tools/benchmark-runner/run-gap-closure-verification.mjs`
  - Added scenario filtering (`VITRUM_SCENARIO_FILTER`) for deterministic targeted runs.
  - Added per-scenario/per-variant progress logs.
  - Baseline generation now writes canonical variant file and compatibility alias (`<scenario>.png`) for candidate variants.
- `tools/reference-renders/README.md`
  - Documented canonical variant-aware baseline naming.

Evidence:
- `VITRUM_SCENARIO_FILTER="rfe03-layered-front-back" npm run benchmark:gap-closure --workspace @vitrum/benchmark-runner`: PASS (advisory mode)

## Wave E — CI/workflow hardening

- `.github/workflows/ci.yml`
  - Added explicit `_staging` typecheck gate.
  - Made benchmark gate honest:
    - strict gap-closure only on capture-enabled runners.
    - explicit "skipped (no capture)" message on default runners.
- `package.json`
  - `verify:mechanical` now includes `_staging` typecheck.
- `scripts/run-fork-shader-smoke.mjs`
  - Added deterministic expected-branch preflight (`phase4-normalmap-shadow-rays` by default) with clear failure diagnostics.

Evidence:
- `npm run fork-shader-smoke`: deterministic branch preflight failure (expected on local fork branch mismatch).

## Wave F — Final validation summary

Mechanical checks run:

- `npm run typecheck`: PASS
- `npm run typecheck:staging`: PASS
- `npm test`: PASS (all workspace test suites green)
- `npm run shader-compile-ci`: PASS (all 7 Cornell scenarios compiled cleanly)
- `npm run fork-shader-smoke`: FAIL with deterministic branch preflight (local fork checkout on `main`, expected `phase4-normalmap-shadow-rays`)

Gap-closure status:

- Benchmark harness logic and contract were hardened.
- Full strict capture validation and complete baseline population remain dependent on running capture-capable baseline generation sessions.
