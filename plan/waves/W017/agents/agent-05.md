# Gap remediation worker — W017 / agent-05

You are **worker agent-05** in wave **W017**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `INV-005` |
| Lane | `pt-webgpu` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
inverse: spectral path-replay downgrade

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Spectral adjoint
3. Add inverseSession regression test.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- INV-005 implemented or permanently documented as FD-only.
- inverse tests green.

## Hard rules
1. Implement **only** task `INV-005`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: INV-005
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
