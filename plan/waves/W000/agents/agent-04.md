# Gap remediation worker — W000 / agent-04

You are **worker agent-04** in wave **W000**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `P0-004-DENO-001` |
| Lane | `core` |
| Disposition | `IMP` |
| Priority | `P0` |

## Problem
walkaround NoneDenoiser exists but VALID_DENOISERS omits 'none' — construction throws.

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/HybridEngineOptions.ts`
- `packages/walkaround-hybrid/src/HybridEngineConfig.ts`
- `packages/walkaround-hybrid/__tests__/hybridEngineTuning.test.ts`
- `packages/core/src/engine/promiseLedger.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Add 'none' as first entry in VALID_DENOISERS in HybridEngineOptions.ts line 22.
2. Update HybridEngineConfig.ts validation error message to list none.
3. Confirm pipeline/denoisers/none.ts registered in denoiser index.
4. Update promiseLedger WALKAROUND_DENOISERS row for none if missing.
5. Fix hybridEngineTuning.test.ts: denoiser:'none' must construct without throw.
6. Add test: denoiser:'none' skips DenoiserAdapterPass denoise stage.

## Tests you must run locally
- `cd /home/jsquire4/projects/vitrum/packages/walkaround-hybrid && npx vitest run __tests__/hybridEngineTuning.test.ts __tests__/denoiserRegistry.test.ts`
- `cd /home/jsquire4/projects/vitrum/packages/core && npx vitest run src/__tests__/ledgerVsCapabilities.test.ts`

## Definition of done
- 'none' in VALID_DENOISERS.
- Construction succeeds.
- Ledger aligned.

## Hard rules
1. Implement **only** task `P0-004-DENO-001`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: P0-004-DENO-001
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
