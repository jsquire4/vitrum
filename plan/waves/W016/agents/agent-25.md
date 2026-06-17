# Gap remediation worker — W016 / agent-25

You are **worker agent-25** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-A9-BDPT` |
| Lane | `pt-webgpu` |
| Disposition | `VERIFY` |
| Priority | `P2` |

## Problem
A9/BDPT tail: independent radiometric/material-furnace oracle for light-subpath connections (road-to-100 §2C BDPT row Open).

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/wgsl/bdpt/bdptConnection.wgsl.ts`
- `packages/pt-webgpu/src/__tests__/bdptGlossyLightSubpath.test.ts`
- `tools/radiometric-ab/`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. BDPT estimator coherence landed; parallel dispatch still serial.
2. Add or run radiometric oracle vs forward-traced reference for glossy light vertices.
3. Document serial workgroup limitation; design parallel dispatch as follow-up IMP if needed.

## Tests you must run locally
- `cd packages/pt-webgpu && npx vitest run src/bdpt/`
- `tools/radiometric-ab/ if present`

## Definition of done
- BDPT radiometric oracle green or gap filed.
- A9 row updated in road-to-100.

## Hard rules
1. Implement **only** task `RT100-A9-BDPT`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-A9-BDPT
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
