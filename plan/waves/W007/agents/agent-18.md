# Gap remediation worker — W007 / agent-18

You are **worker agent-18** in wave **W007**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `FP-05` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P2` |

## Problem
Pipeline-rebuild-required toggles undocumented.

## Files you may edit (ONLY these)
- `packages/core/src/engine/fidelityProfile.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Document which profile fields require engine recreation vs runtime toggle.

## Tests you must run locally
- `npm run typecheck`

## Definition of done
- JSDoc lists rebuild-required fields.

## Hard rules
1. Implement **only** task `FP-05`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: FP-05
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
