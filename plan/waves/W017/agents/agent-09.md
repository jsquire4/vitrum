# Gap remediation worker — W017 / agent-09

You are **worker agent-09** in wave **W017**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-LD-SAMPLING-01` |
| Lane | `shared-samplers` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
F1/LD-SAMPLING-01: Owen-Sobol or PMJ02 + blue-noise screen scramble in shared-samplers; pt-webgpu + pt-webgl2 integration.

## Files you may edit (ONLY these)
- `packages/shared-samplers/src/`
- `packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts`
- `packages/pt-webgl2/src/glsl/`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100.md F1 section.
2. Generate LD tables CPU-side; upload as textures/buffers.
3. Per-dimension assignment audit (bounce/lobe/light).
4. Revive or replace dead pt-webgl2 RANDOM_TYPE branches.
5. Equal-time RMSE A/B on reference scenes.

## Tests you must run locally
- `cd packages/shared-samplers && npx vitest run`
- `npm run typecheck`

## Definition of done
- LD sampling integrated both PT backends.
- RMSE A/B documents improvement.

## Hard rules
1. Implement **only** task `RT100-LD-SAMPLING-01`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-LD-SAMPLING-01
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
