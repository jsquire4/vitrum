# Gap remediation worker — W008 / agent-18

You are **worker agent-18** in wave **W008**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTWG-039` |
| Lane | `pt-webgpu` |
| Disposition | `DOC` |
| Priority | `P2` |

## Problem
pt-webgpu: merged multi-material fidelity

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/scene/uploadSceneBuffers.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgpu/src/scene/uploadSceneBuffers.ts.
2. Implement: Warn on merged multi-material
3. Add/update test in packages/pt-webgpu/__tests__ or cited test path.
4. npm run typecheck from repo root.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTWG-039 complete.
- Tests green.

## Hard rules
1. Implement **only** task `PTWG-039`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTWG-039
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
