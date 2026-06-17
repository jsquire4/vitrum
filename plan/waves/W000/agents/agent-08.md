# Gap remediation worker — W000 / agent-08

You are **worker agent-08** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-008-TOOL-001` |
| Lane | `tools` |
| Disposition | `TEST` |
| Priority | `P0` |

## Problem
behavioral gate has zero pt-webgl2 configs.

## Files you may edit (ONLY these)
- `tools/behavioral-gate/gate.mjs`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open tools/behavioral-gate/gate.mjs.
2. Add import for pt-webgl2 engine factory (mirror existing pt-webgpu import pattern).
3. Add PTGL_CONFIGS array with at least: ptgl/default, ptgl/spectral, ptgl/mesh-area.
4. Add run branch: label prefix ptgl/ uses WebGL2 factory.
5. Add EXPECTATION_TABLE entries for each ptgl/* label.
6. Document in gate header: pt-webgl2 needs WebGL2 context; skip via env if unavailable.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum && node tools/behavioral-gate/gate.mjs --filter ptgl 2>&1 | head -50`

## Definition of done
- ≥3 ptgl/* configs defined.
- Gate imports without error.

## Hard rules
1. Implement **only** task `P0-008-TOOL-001`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-008-TOOL-001
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
