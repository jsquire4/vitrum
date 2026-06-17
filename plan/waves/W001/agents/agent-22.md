# Gap remediation worker — W001 / agent-22

You are **worker agent-22** in wave **W001**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTWG-004` |
| Lane | `core` |
| Disposition | `TEST` |
| Priority | `P1` |

## Problem
pt-webgpu: lite vs ledger incrementalPatchSupport drift

## Files you may edit (ONLY these)
- `packages/core/src/__tests__/ledgerVsCapabilities.test.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/core/src/__tests__/ledgerVsCapabilities.test.ts.
2. Implement: Tier-aware ledger test for lite mutations
3. Add/update test in packages/core/__tests__ or cited test path.
4. npm run typecheck from repo root.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTWG-004 complete.
- Tests green.

## Hard rules
1. Implement **only** task `PTWG-004`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTWG-004
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
