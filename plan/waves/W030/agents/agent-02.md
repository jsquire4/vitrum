# Gap remediation worker — W030 / agent-02

You are **worker agent-02** in wave **W030**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-F5-VOLUMES` |
| Lane | `core` |
| Disposition | `ACC` |
| Priority | `P3` |

## Problem
F5: heterogeneous volumes (null-collision delta tracking) — product-gated; stained-glass may never need (road-to-100 post-100).

## Files you may edit (ONLY these)
- `plan/road-to-100.md`
- `packages/core/src/scene/`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Confirm product scope with user before contract extension.
2. If in scope: add AnalyticShape/Material.extensions volume primitive first.
3. Else: document permanent unsupported + planner routing.

## Tests you must run locally
- `npm run typecheck`

## Definition of done
- F5 scope decision recorded.

## Hard rules
1. Implement **only** task `RT100-F5-VOLUMES`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-F5-VOLUMES
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
