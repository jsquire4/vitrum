# Gap remediation worker — W021 / agent-04

You are **worker agent-04** in wave **W021**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-V28-A7` |
| Lane | `repo-root` |
| Disposition | `VERIFY` |
| Priority | `P2` |

## Problem
A7 implementation landed — radiometric/variance GPU A/B pending (road-to-100 Bucket A).

## Files you may edit (ONLY these)
- `plan/road-to-100.md`
- `HARDWARE-VALIDATION-NEEDS.md`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Locate A7 Done= criteria in plan/road-to-100.md Bucket A row.
2. Run specified A/B scenario on wsl-gpu (equal-spp or equal-time per row).
3. Capture to tools/reference-renders/ and update validation matrix row.

## Tests you must run locally
- `wsl-gpu T1 smoke + scenario-specific capture script`

## Definition of done
- A7 V28 evidence captured or explicitly deferred with reason.

## Hard rules
1. Implement **only** task `RT100-V28-A7`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-V28-A7
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
