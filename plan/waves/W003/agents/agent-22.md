# Gap remediation worker — W003 / agent-22

You are **worker agent-22** in wave **W003**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-ADJ-001` |
| Lane | `pt-webgpu` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
pt-webgpu inverse adjoint path replay OPEN: alpha-map adjoint, normal/bump/transmission adjoints, env light terms, indirect paths (road-to-100 §2C Adjoint row).

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`
- `packages/pt-webgpu/src/inverse/pathTraceAdjoint.wgsl.ts`
- `packages/pt-webgpu/src/inverse/inverseSession.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100.md §2C integrator audit Adjoint row Still OPEN list.
2. Implement or permanently downgrade each OPEN adjoint domain with structured diagnostic.
3. Extend brdfAdjoint.test.ts + inverseSession.test.ts per domain.
4. Update promiseLedger inverse downgrade matrix comments.

## Tests you must run locally
- `cd packages/pt-webgpu && npx vitest run src/inverse/`
- `npm run typecheck`

## Definition of done
- Each OPEN adjoint row closed or downgraded with test.

## Hard rules
1. Implement **only** task `RT100-ADJ-001`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-ADJ-001
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
