# Gap remediation worker — W016 / agent-21

You are **worker agent-21** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `TOOL-004` |
| Lane | `tools` |
| Disposition | `TEST` |
| Priority | `P2` |

## Problem
gltf-material-sweep limited fixtures

## Files you may edit (ONLY these)
- `tools/gltf-material-sweep/sweep.mjs`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Expand SWEEP_MAPS to cover all 65 material fields

## Tests you must run locally
- `node tools/gltf-material-sweep/sweep.mjs --dry-run`

## Definition of done
- Sweep map list complete.

## Hard rules
1. Implement **only** task `TOOL-004`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: TOOL-004
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
