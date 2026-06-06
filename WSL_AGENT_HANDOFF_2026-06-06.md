# WSL Agent Handoff — 2026-06-06 Audit Sweep

This repo needs a native WSL/Linux pass. The prior agent worked through a Windows-mapped path (`V:\`) and could typecheck, but Vitest/Vite path resolution failed before test collection (`V:/...` resolved to `/wsl.localhost/...`). Please continue from `/home/jsquire4/projects/vitrum` with native Linux Node/npm.

## Read First

- `AGENTS.md`
- `items_to_fix.md`
- `full-codebase-audit-2026-06-06.md`
- `plan/renderer-fidelity-matrix.md`

Do not push or publish. Do not rewrite `package-lock.json` unless you intentionally run a real dependency update and verify the lockfile diff is small and correct.

## Current Working-Tree State

The implementation sweep has code changes for:

- G-P0.1 ReSTIR-DI smooth-normal consistency.
- G-P0.2 DDGI irradiance convention.
- G-P0.3 pt-webgl fork RGB throughput restoration.
- G-P0.4 three-bindings data/space conversion gaps.
- G-P1.1 PPG update now reads real GI reservoir data; dead PPG guide pass removed.
- G-P1.2 WebGL BDPT compile/fill/math issues.
- G-P1.3 caustic support honesty in capabilities/docs.
- G-P1.4 shared-denoisers cleanup.
- Most P2 cleanup/test/docs items.
- Partial G-P2.6: memory accounting, static fork spectral-table uniform caching, and indirect atrous UBO ownership cleanup.

`items_to_fix.md` and `full-codebase-audit-2026-06-06.md` were updated to say implementation is in progress/mostly landed but validation is still pending.

## First Task: Native Validation

Run these from native WSL, not a Windows mapped drive:

```bash
cd /home/jsquire4/projects/vitrum
node --version
npm --version
npm run typecheck
npm test
node tools/generate-wgsl-layouts.mjs
(cd packages/three-gpu-pathtracer && node scripts/shader-smoke-check.js)
git diff --check
```

If `npm test` fails, prioritize real code/test failures over known GPU opt-in skips. The previous environment could not collect tests at all, so any collected failing test is new signal.

Also run these targeted tests if the full suite is too slow:

```bash
node node_modules/vitest/vitest.mjs run \
  packages/pt-webgl/src/__tests__/forkUniformBridge.test.ts \
  packages/walkaround-hybrid/__tests__/walkaroundPipelineCharacterization.test.ts \
  packages/walkaround-hybrid/src/shaders/__tests__/smoothNormals.test.ts \
  packages/walkaround-hybrid/__tests__/ddgiPipeline.test.ts \
  packages/walkaround-hybrid/__tests__/ppg.test.ts \
  packages/three-bindings/src/__tests__/sceneFromThreeJS.test.ts \
  packages/three-bindings/src/__tests__/meshAttributes.test.ts \
  packages/three-bindings/src/__tests__/vitrumSceneToThree.test.ts \
  packages/pt-webgl/src/__tests__/bdptSceneEmittersCpu.test.ts \
  packages/pt-webgl/src/__tests__/bdptLightPathPackWebGL.test.ts
```

## GPU / Reference Render Work Still Needed

The code changes touch rendering math, so mechanical tests are not enough. Capture before/after or current-vs-reference evidence for:

- G-P0.1: curved smooth-normal scene, ReSTIR reuse on vs RIS-only, to confirm no normal-mismatch bias.
- G-P0.2: uniform-room DDGI producer-to-receiver path, confirming Lambertian energy after the PI reconstruction.
- G-P0.3: Cornell/multi-bounce color bleed in pt-webgl vs pt-webgpu or independent baseline.
- G-P1.1: PPG enabled vs cosine-only at equal sample budget, looking for measurable variance reduction.
- G-P1.2: WebGL BDPT with `FEATURE_BDPT=1` compiles and renders on hardware.

Use the existing `tools/benchmark-runner/` and `tools/reference-renders/` conventions. If you capture new baselines, document what changed and why.

## Remaining Non-Validation Engineering Work

G-P2.6 still has larger optimizations that were deliberately not rushed:

- Bind-group and texture-view memoization in hot frame paths.
- SVGF lazy allocation so unused denoiser resources are not created eagerly.
- Broader allocator/resource lifetime cleanup after measuring the real hotspots.

Run GitNexus impact analysis before editing symbols, per `AGENTS.md`. Known high-risk areas:

- `WalkaroundGPUPipeline`
- `BvhBufferHost`
- `driveForkMaterialUniforms`
- pass registration / frame resource creation

If impact comes back HIGH or CRITICAL, say so before editing and keep the patch narrow.

## Known Environment Notes

- The prior agent restored `package-lock.json`; it should currently have no diff.
- The Windows-mapped run produced only CRLF warnings from `git diff --check`, no whitespace errors in the targeted files.
- `gitnexus detect-changes --repo vitrum` failed in the Windows-mapped environment because its internal `git diff` ran outside the worktree. Retry natively in WSL before any commit.

## Suggested Finish Criteria

- `npm run typecheck` passes.
- `npm test` passes in native WSL/CI.
- Fork shader smoke passes.
- `gitnexus detect-changes --repo vitrum --scope all` succeeds and affected scope matches Section G work.
- GPU/reference render evidence is captured or the remaining captures are explicitly documented with blockers.
- `items_to_fix.md` and `full-codebase-audit-2026-06-06.md` are updated from “validation pending” to precise closed/open status.
