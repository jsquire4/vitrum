# Gap remediation worker — W017 / agent-10

You are **worker agent-10** in wave **W017**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-WBVH-01` |
| Lane | `shared-bvh` |
| Disposition | `IMP` |
| Priority | `P2` |

## Problem
F2/WBVH-01: compressed wide BVH (CWBVH) opt-in builder + traversal behind capability flag.

## Files you may edit (ONLY these)
- `packages/shared-bvh/src/`
- `packages/shared-bvh/wgsl/`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100.md F2 section.
2. Implement CWBVH build + WGSL traversal in shared-bvh.
3. CPU brute-force oracle vs binary BVH (T1 smoke pattern).
4. Per-backend opt-in until parity proven.

## Tests you must run locally
- `cd packages/shared-bvh && npx vitest run`
- `wsl-gpu T1 BVH oracles`

## Definition of done
- CWBVH behind capability flag.
- Oracle parity on test scenes.

## Hard rules
1. Implement **only** task `RT100-WBVH-01`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-WBVH-01
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
