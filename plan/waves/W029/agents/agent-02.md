# Gap remediation worker — W029 / agent-02

You are **worker agent-02** in wave **W029**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-F4-WAVEFRONT` |
| Lane | `repo-root` |
| Disposition | `ACC` |
| Priority | `P3` |

## Problem
F4: wavefront PT rearchitecture — profile-gated research item, not arbitrary-glTF blocker (road-to-100 post-100).

## Files you may edit (ONLY these)
- `plan/road-to-100.md`
- `packages/pt-webgpu/src/wgsl/pathTrace/kernel.wgsl.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Run divergence profiling on reference scenes before scheduling.
2. Document gate criteria in plan/road-to-100.md or archive.
3. Do not implement unless profiling justifies; mark ACC in roadmap.

## Tests you must run locally
- `npm run typecheck`

## Definition of done
- F4 decision documented as deferred or scoped.

## Hard rules
1. Implement **only** task `RT100-F4-WAVEFRONT`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-F4-WAVEFRONT
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
