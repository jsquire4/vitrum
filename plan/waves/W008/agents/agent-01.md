# Gap remediation worker — W008 / agent-01

You are **worker agent-01** in wave **W008**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `WH-007` |
| Lane | `walkaround-hybrid` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
walkaround: extensions partial consume

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts.
2. Implement: Line 50: warn on unknown extension keys via onWarning with code unknown-material-extension
3. Add or update test in packages/walkaround-hybrid __tests__.
4. Run: cd packages/walkaround-hybrid && npx vitest run
5. Run: npm run typecheck from repo root.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- WH-007 fix implemented per steps.
- Tests green.
- typecheck green.

## Hard rules
1. Implement **only** task `WH-007`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: WH-007
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
