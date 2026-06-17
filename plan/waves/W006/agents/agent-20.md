# Gap remediation worker — W006 / agent-20

You are **worker agent-20** in wave **W006**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `TOOL-005` |
| Lane | `tools` |
| Disposition | `DOC` |
| Priority | `P2` |

## Problem
benchmark-runner no in-repo Playwright

## Files you may edit (ONLY these)
- `tools/benchmark-runner/README.md`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Document host Playwright requirement for browser capture modes

## Tests you must run locally


## Definition of done
- README updated.

## Hard rules
1. Implement **only** task `TOOL-005`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: TOOL-005
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
