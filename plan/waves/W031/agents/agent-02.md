# Gap remediation worker — W031 / agent-02

You are **worker agent-02** in wave **W031**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-F-BRIDGE` |
| Lane | `repo-root` |
| Disposition | `DOC` |
| Priority | `P3` |

## Problem
F-BRIDGE: experimental no-hardware-RT bridge levers — track as research backlog, not 100% blockers (road-to-100 §F-BRIDGE table).

## Files you may edit (ONLY these)
- `plan/road-to-100.md`
- `plan/roadmap.md`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Ensure F-BRIDGE table remains in road-to-100 with feasibility notes.
2. Cross-link active performance track (LD-SAMPLING, WBVH) vs bridge items.
3. No implementation unless promoted by user.

## Tests you must run locally
- `npm run typecheck`

## Definition of done
- F-BRIDGE backlog visible and separated from Phase 0–6 closure.

## Hard rules
1. Implement **only** task `RT100-F-BRIDGE`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-F-BRIDGE
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
