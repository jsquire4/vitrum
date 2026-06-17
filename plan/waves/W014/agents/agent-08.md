# Gap remediation worker — W014 / agent-08

You are **worker agent-08** in wave **W014**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `WH-112` |
| Lane | `walkaround-hybrid` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
walkaround: PPG flux readback fire-and-forget

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/walkaround-hybrid/src/ppg/ppgTrainPass.ts.
2. Implement: Await readback or double-buffer flux
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- WH-112 fix implemented per steps.
- Tests green.
- typecheck green.

## Hard rules
1. Implement **only** task `WH-112`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: WH-112
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
