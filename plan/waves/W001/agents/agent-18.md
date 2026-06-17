# Gap remediation worker — W001 / agent-18

You are **worker agent-18** in wave **W001**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `FP-01` |
| Lane | `core` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
EngineFidelityProfile type does not exist — quality scattered across options.

## Files you may edit (ONLY these)
- `packages/core/src/engine/fidelityProfile.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Create packages/core/src/engine/fidelityProfile.ts with EngineFidelityProfile type.
2. Fields: materialStorage quantized|full, emissiveImportance, alphaInGi, grisEnabled, restirPtReuse, traceTier.
3. Export from packages/core/src/engine/index.ts.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run`

## Definition of done
- Type exported from @vitrum/core.

## Hard rules
1. Implement **only** task `FP-01`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: FP-01
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
