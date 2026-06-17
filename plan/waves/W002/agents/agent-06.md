# Gap remediation worker — W002 / agent-06

You are **worker agent-06** in wave **W002**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `ENG-008` |
| Lane | `engine` |
| Disposition | `VERIFY` |
| Priority | `P1` |

## Problem
ENG-008: setSize noop disposed

## Files you may edit (ONLY these)
- `packages/engine/src/idempotentDispose.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/engine/src/idempotentDispose.ts:108-109.
2. Gap: setSize noop disposed
3. Fix: Verify pt-webgpu never proxied setSize
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- ENG-008 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `ENG-008`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: ENG-008
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
