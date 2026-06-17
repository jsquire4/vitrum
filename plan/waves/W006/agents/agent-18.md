# Gap remediation worker — W006 / agent-18

You are **worker agent-18** in wave **W006**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `TOOL-002` |
| Lane | `tools` |
| Disposition | `TEST` |
| Priority | `P2` |

## Problem
behavioral gate missing oidn/neural/svgf-real walkaround configs

## Files you may edit (ONLY these)
- `tools/behavioral-gate/gate.mjs`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Add opt-in wh/oidn, wh/neural, wh/svgf-real configs behind env flag VITRUM_HEAVY_DENOISER_GATE=1

## Tests you must run locally
- `node tools/behavioral-gate/gate.mjs --filter wh/oidn`

## Definition of done
- Configs documented and gated.

## Hard rules
1. Implement **only** task `TOOL-002`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: TOOL-002
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
