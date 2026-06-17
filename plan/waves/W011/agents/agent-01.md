# Gap remediation worker — W011 / agent-01

You are **worker agent-01** in wave **W011**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-WA-ALPHA` |
| Lane | `walkaround-hybrid` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
Walkaround Phase 3C alpha: transparent ReSTIR/GI promotion + layered transport (road-to-100 Master checklist alpha rows).

## Files you may edit (ONLY these)
- `packages/walkaround-hybrid/src/shaders/transparentOit.wgsl.ts`
- `packages/walkaround-hybrid/src/shaders/shade.wgsl.ts`
- `packages/walkaround-hybrid/src/restir/consumedMaterialFields.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100 alpha row: OIT direct sun done; ReSTIR/GI transport still approximate.
2. Implement stochastic alpha GI transport OR document permanent OIT-split with ledger ACC.
3. Add behavioral gate wh/alpha-blend if not present.

## Tests you must run locally
- `cd packages/walkaround-hybrid && npx vitest run`

## Definition of done
- Alpha GI policy implemented or permanently documented.

## Hard rules
1. Implement **only** task `RT100-WA-ALPHA`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-WA-ALPHA
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
