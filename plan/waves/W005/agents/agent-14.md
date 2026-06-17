# Gap remediation worker — W005 / agent-14

You are **worker agent-14** in wave **W005**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `CORE-045` |
| Lane | `core` |
| Disposition | `DOC` |
| Priority | `P2` |

## Problem
CORE-045: material deep-merge vs emitter shallow

## Files you may edit (ONLY these)
- `packages/core/src/scene/patchScene.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Open packages/core/src/scene/patchScene.ts:103-112.
2. Gap: material deep-merge vs emitter shallow
3. Fix: Document OR deep-merge emitters
4. Add regression test if behavior changes.
5. npm run typecheck

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run`
- `cd /home/jsquire4/projects/vitrum && npm run typecheck`

## Definition of done
- CORE-045 fix applied.
- Tests green.

## Hard rules
1. Implement **only** task `CORE-045`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: CORE-045
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
