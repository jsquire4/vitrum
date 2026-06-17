# Gap remediation worker — W006 / agent-19

You are **worker agent-19** in wave **W006**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `TOOL-003` |
| Lane | `tools` |
| Disposition | `TEST` |
| Priority | `P2` |

## Problem
shader-gate compiles WGSL only not GLSL pt-webgl2

## Files you may edit (ONLY these)
- `tools/shader-gate/gate.mjs`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Add glslGate pass mirroring wgsl compose for pt-webgl2 composeTraceGlsl

## Tests you must run locally
- `npm run shader-gate`

## Definition of done
- GLSL paths in shader-gate.

## Hard rules
1. Implement **only** task `TOOL-003`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: TOOL-003
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
