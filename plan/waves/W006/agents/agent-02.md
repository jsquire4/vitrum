# Gap remediation worker — W006 / agent-02

You are **worker agent-02** in wave **W006**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `MUT-08` |
| Lane | `walkaround-hybrid` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
Mutation matrix MUT-08: walkaround topology fallback

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/HybridEnginePrimitiveUpdates.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Complete linked task WH-037 first.
2. Verify MUT-08 acceptance: mutation ledger row matches runtime behavior.

## Tests you must run locally
- `npm run typecheck`
- `npm test`

## Definition of done
- MUT-08 closed when WH-037 done and ledger verified.

## Hard rules
1. Implement **only** task `MUT-08`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: MUT-08
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
