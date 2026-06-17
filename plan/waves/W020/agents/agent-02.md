# Gap remediation worker — W020 / agent-02

You are **worker agent-02** in wave **W020**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `INV-008` |
| Lane | `pt-webgpu` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
inverse: normalScale/bumpScale FD only

## Files you may edit (ONLY these)
- `packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgpu/src/inverse/adjointPass.wgsl.ts.
2. Implement: Normal-map adjoint
3. Add inverseSession regression test.
4. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgpu && npx vitest run src/inverse/`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- INV-008 implemented or permanently documented as FD-only.
- inverse tests green.

## Hard rules
1. Implement **only** task `INV-008`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: INV-008
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
