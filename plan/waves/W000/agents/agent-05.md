# Gap remediation worker — W000 / agent-05

You are **worker agent-05** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-005-LEDGER-01` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P0` |

## Problem
promiseLedger.ts:907-911 stale comment claims point/spot DDGI-only approximate.

## Files you may edit (ONLY these)
- `packages/core/src/engine/promiseLedger.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open promiseLedger.ts lines 907-911.
2. Delete stale comment block about point/spot DDGI-only approximate.
3. WALKAROUND_EMITTERS already grades point/spot native — do not change ledger rows.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- Stale comment removed.
- ledgerVsCapabilities green.

## Hard rules
1. Implement **only** task `P0-005-LEDGER-01`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-005-LEDGER-01
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
