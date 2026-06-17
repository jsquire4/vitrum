# Gap remediation worker — W007 / agent-12

You are **worker agent-12** in wave **W007**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTGL-023` |
| Lane | `pt-webgl2` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
PTGL-023: fold keying prim.id only

## Files you may edit (ONLY these)
- `packages/pt-webgl2/src/scene/foldEmissiveEmitters.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgl2/src/scene/foldEmissiveEmitters.ts:42.
2. Gap: fold keying prim.id only
3. Fix: meshId semantics across prims
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTGL-023 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `PTGL-023`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTGL-023
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
