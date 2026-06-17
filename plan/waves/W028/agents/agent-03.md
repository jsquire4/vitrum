# Gap remediation worker — W028 / agent-03

You are **worker agent-03** in wave **W028**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-GATE-FULL` |
| Lane | `tools` |
| Disposition | `TEST` |
| Priority | `P2` |

## Problem
Phase 5E tail: expand behavioral gate to full-tier pt-webgpu + walkaround glTF lanes when WSL adapter allows (road-to-100 §5E honesty boundary).

## Files you may edit (ONLY these)
- `tools/behavioral-gate/gate.mjs`
- `HARDWARE-VALIDATION-NEEDS.md`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Document current lite-tier limitation on lavapipe.
2. When full-tier adapter available: run glTF fixtures at full tier with rich material lobes.
3. Fix walkaround Deno/wgpu-hal panic blocker or route through alternate harness.

## Tests you must run locally
- `npm run behavioral-gate`
- `npm run behavioral-gate -- --filter gltf`

## Definition of done
- Full-tier gate path documented or implemented.
- Walkaround glTF lane unblocked or explicitly deferred.

## Hard rules
1. Implement **only** task `RT100-GATE-FULL`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-GATE-FULL
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
