# Gap remediation worker — W002 / agent-11

You are **worker agent-11** in wave **W002**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `FP-02` |
| Lane | `engine` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
createEngine does not map fidelity profile.

## Files you may edit (ONLY these)
- `packages/engine/src/createEngineInternals.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Add mapFidelityProfile(backend, options) called during engine construction.
2. Merge result into effective engine options.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/engine && npx vitest run`

## Definition of done
- Profile mapping wired.

## Hard rules
1. Implement **only** task `FP-02`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: FP-02
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
