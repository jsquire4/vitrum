# Gap remediation worker — W016 / agent-14

You are **worker agent-14** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-EMISSIVE-PDF` |
| Lane | `shared-samplers` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
B4 tail: full energy-weighted emissive texel alias/PDF with forward-hit MIS parity (road-to-100 — not just area-PDF).

## Files you may edit (ONLY these)
- `packages/shared-samplers/src/meshAreaLights.ts`
- `packages/pt-webgpu/src/scene/meshAreaLights.ts`
- `packages/pt-webgl2/src/scene/meshAreaLights.ts`
- `packages/walkaround-hybrid/src/restir/emitterList.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100 B4 Done= tail about texel alias/PDF.
2. Implement alias table for emissive texel importance on CPU packers.
3. Ensure forward NEE PDF matches forward-hit MIS weight.
4. Extend meshAreaMis.test.ts across backends.

## Tests you must run locally
- `npm test --workspaces --if-present`

## Definition of done
- Alias/PDF path live on all three backends.
- meshAreaMis parity tests green.

## Hard rules
1. Implement **only** task `RT100-EMISSIVE-PDF`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-EMISSIVE-PDF
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
