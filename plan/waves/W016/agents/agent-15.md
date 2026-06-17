# Gap remediation worker — W016 / agent-15

You are **worker agent-15** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-5D-DOC` |
| Lane | `repo-root` |
| Disposition | `DOC` |
| Priority | `P2` |

## Problem
Phase 5D documentation sync: fidelity matrix, items_to_fix §H, road-to-100 stale addenda (H1–H5), READMEs cite ledger (road-to-100 C5).

## Files you may edit (ONLY these)
- `plan/renderer-fidelity-matrix.md`
- `plan/road-to-100.md`
- `items_to_fix.md`
- `README.md`
- `packages/*/README.md`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Remove deleted pt-webgl column from fidelity matrix; ensure pt-webgl2 column accurate.
2. Strike/reconcile stale road-to-100 addendum bullets (e.g. H1–H5 inert — closed in items_to_fix).
3. Close or strike items_to_fix §H entries verified fixed.
4. README maturity claims cite BACKEND_PROMISE_LEDGER not prose.

## Tests you must run locally
- `npm run typecheck`

## Definition of done
- Doc sync checklist complete.
- No stale OPEN claims contradicting items_to_fix.

## Hard rules
1. Implement **only** task `RT100-5D-DOC`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-5D-DOC
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
