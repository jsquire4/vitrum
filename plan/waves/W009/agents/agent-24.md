# Gap remediation worker — W009 / agent-24

You are **worker agent-24** in wave **W009**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTWG-069` |
| Lane | `pt-webgpu` |
| Disposition | `BUG` |
| Priority | `P2` |

## Problem
pt-webgpu: implicit emitter desync

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/scene/meshAreaLights.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgpu/src/scene/meshAreaLights.ts.
2. Implement: Resync mesh-area on material patch
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTWG-069 complete.
- Tests green.

## Hard rules
1. Implement **only** task `PTWG-069`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTWG-069
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
