# Gap remediation worker — W003 / agent-02

You are **worker agent-02** in wave **W003**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `PTGL-008` |
| Lane | `pt-webgl2` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
PTGL-008: displacement warns only

## Files you may edit (ONLY these)
- `packages/pt-webgl2/src/index.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/pt-webgl2/src/index.ts:87.
2. Gap: displacement warns only
3. Fix: Tessellate OR hard-drop
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/pt-webgl2 && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- PTGL-008 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `PTGL-008`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: PTGL-008
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
