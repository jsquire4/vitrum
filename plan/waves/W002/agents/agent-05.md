# Gap remediation worker — W002 / agent-05

You are **worker agent-05** in wave **W002**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `ENG-006` |
| Lane | `engine` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
ENG-006: engine recreates on gltf change

## Files you may edit (ONLY these)
- `packages/engine/src/react/VitrumCanvas.tsx`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/engine/src/react/VitrumCanvas.tsx:141-165.
2. Gap: engine recreates on gltf change
3. Fix: Ref-stabilize gltfOptions
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- ENG-006 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `ENG-006`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: ENG-006
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
