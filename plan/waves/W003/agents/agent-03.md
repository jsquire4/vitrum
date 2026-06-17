# Gap remediation worker — W003 / agent-03

You are **worker agent-03** in wave **W003**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `FP-03` |
| Lane | `core` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
capabilities.supportDetails not derived from active profile.

## Files you may edit (ONLY these)
- `packages/engine/src/createEngineInternals.ts`
- `packages/core/src/engine/capabilities.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. After mapFidelityProfile, patch supportDetails.materials rows affected by profile.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- Capabilities reflect active profile.

## Hard rules
1. Implement **only** task `FP-03`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: FP-03
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
