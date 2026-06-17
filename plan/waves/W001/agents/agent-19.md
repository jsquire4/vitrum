# Gap remediation worker — W001 / agent-19

You are **worker agent-19** in wave **W001**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `MUT-07` |
| Lane | `pt-webgpu` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
Mutation matrix MUT-07: pt-webgpu lite material GPU stale

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/sceneMutationRouter.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Complete linked task P0-001-PTWG-037 first.
2. Verify MUT-07 acceptance: mutation ledger row matches runtime behavior.

## Tests you must run locally
- `npm run typecheck`
- `npm test`

## Definition of done
- MUT-07 closed when P0-001-PTWG-037 done and ledger verified.

## Hard rules
1. Implement **only** task `MUT-07`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: MUT-07
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
