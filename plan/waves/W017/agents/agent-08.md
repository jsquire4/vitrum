# Gap remediation worker — W017 / agent-08

You are **worker agent-08** in wave **W017**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-5C-GPU-MUT` |
| Lane | `walkaround-hybrid` |
| Disposition | `VERIFY` |
| Priority | `P2` |

## Problem
Phase 5C: adapter-backed GPU mutation matrix — real buffers, bind groups, denoiser history, GI propagation observed together.

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/__tests__/mutationMatrix.test.ts`
- `packages/pt-webgpu/src/__tests__/updatePrimitiveIncremental.test.ts`
- `tools/benchmark-runner/`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Extend wsl-gpu harness to observe GPU buffer writes on mutation scenarios.
2. Cover walkaround updatePrimitive/updateEnvironment/setSize + pt-webgpu incremental paths.
3. Document results in plan/archive or validation matrix.

## Tests you must run locally
- `wsl-gpu mutation observability harness`

## Definition of done
- GPU mutation matrix report exists.

## Hard rules
1. Implement **only** task `RT100-5C-GPU-MUT`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-5C-GPU-MUT
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
