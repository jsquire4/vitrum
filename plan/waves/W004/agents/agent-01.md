# Gap remediation worker — W004 / agent-01

You are **worker agent-01** in wave **W004**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `FP-06` |
| Lane | `walkaround-hybrid` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
Runtime UBO bits for material storage not wired.

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`
- `packages/walkaround-hybrid/src/HybridEngine.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Add HybridEngine UBO bit materialStorageQuantized from profile.
2. shade.wgsl reads bit to select atlas decode path.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run`

## Definition of done
- UBO bit wired end-to-end.

## Hard rules
1. Implement **only** task `FP-06`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: FP-06
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
