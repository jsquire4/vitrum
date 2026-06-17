# Gap remediation worker — W016 / agent-17

You are **worker agent-17** in wave **W016**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-F3-DENO-AUTO` |
| Lane | `engine` |
| Disposition | `IMP` |
| Priority | `P3` |

## Problem
F3: denoiser:auto default when weights resolve; turnkey OIDN/neural without host wiring.

## Files you may edit (ONLY these)
- `packages/engine/src/createEngine.ts`
- `packages/walkaround-hybrid/src/HybridEngineOptions.ts`
- `packages/shared-denoisers/`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100 F3 section.
2. Add denoiser:auto union value.
3. Resolve bundled or downloadable weights at engine construction.
4. Clear error when assets missing.

## Tests you must run locally
- `cd packages/engine && npx vitest run`

## Definition of done
- denoiser:auto documented and functional when weights present.

## Hard rules
1. Implement **only** task `RT100-F3-DENO-AUTO`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-F3-DENO-AUTO
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
