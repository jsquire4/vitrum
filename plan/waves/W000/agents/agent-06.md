# Gap remediation worker — W000 / agent-06

You are **worker agent-06** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-006-LEDGER-02` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P0` |

## Problem
core engine/index.ts JSDoc says walkaround captureFrame('output') rejects.

## Files you may edit (ONLY these)
- `packages/core/src/engine/index.ts`
- `packages/core/src/engine/promiseLedger.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/core/src/engine/index.ts captureFrame JSDoc (~402-411).
2. Replace walkaround 'output' rejects text with: supports colorSpace:'output' via captureOutputFrame (tonemapped present).
3. Fix matching promiseLedger method comment at ~977 if still wrong.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- JSDoc matches HybridEngine.captureFrame behavior.

## Hard rules
1. Implement **only** task `P0-006-LEDGER-02`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-006-LEDGER-02
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
