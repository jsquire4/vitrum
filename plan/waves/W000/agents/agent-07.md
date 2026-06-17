# Gap remediation worker — W000 / agent-07

You are **worker agent-07** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-007-LEDGER-03` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P0` |

## Problem
promiseLedger updateEnvironment note says HDRI intensity-only; Wave 4/5 env NEE is live.

## Files you may edit (ONLY these)
- `packages/core/src/engine/promiseLedger.ts`
- `packages/walkaround-hybrid/src/HybridEngine.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read HybridEngine.updateEnvironment implementation.
2. Rewrite promiseLedger.ts lines ~966-970 to describe actual behavior: scalar sky + HDRI CDF rebuild when resolver provides pixels.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- Comment matches updateEnvironment code.

## Hard rules
1. Implement **only** task `P0-007-LEDGER-03`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-007-LEDGER-03
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
