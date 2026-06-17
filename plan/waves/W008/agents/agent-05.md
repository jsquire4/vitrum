# Gap remediation worker — W008 / agent-05

You are **worker agent-05** in wave **W008**. You own exactly **one** task.

## Task identity
| Field | Value |
|-------|-------|
| Task ID | `RT100-GLTF-PICK` |
| Lane | `engine` |
| Disposition | `IMP` |
| Priority | `P1` |

## Problem
Arbitrary glTF Phase 4: pickBackend must use feature report not triangle-count alone (road-to-100 §4 + createEngineScale footgun).

## Files you may edit (ONLY these)
- `packages/engine/src/createEngineScale.ts`
- `packages/gltf-adapter/src/featureReport.ts`
- `packages/engine/src/gltf.ts`

**Do not** edit any other file. Do not refactor unrelated code.

## Steps (execute in order)
1. Read road-to-100.md Phase 4 and trap table pickBackend row.
2. Wire rankGltfBackends / evaluateGltfBackendCompatibility into createEngine preference path.
3. Add test: textured hero asset must not auto-route to walkaround when PT supports features.

## Tests you must run locally
- `cd packages/engine && npx vitest run`
- `cd packages/gltf-adapter && npx vitest run`

## Definition of done
- pickBackend uses compatibility report for glTF assets.

## Hard rules
1. Implement **only** task `RT100-GLTF-PICK`.
2. Run every test command; fix until green.
3. **Do not** `git commit` — the orchestrator commits after verification.
4. **Do not** mark task done in progress files — the orchestrator verifies independently.
5. **Do not** claim completion if tests fail or files were not edited.
6. If blocked, report the blocker in your final message — do not invent a fake fix.

## Completion signal (informational only — orchestrator does not trust this)
When finished, end your message with:
```
TASK_COMPLETE: RT100-GLTF-PICK
FILES_TOUCHED: <comma-separated paths you actually modified>
TESTS_RUN: <comma-separated commands with exit 0>
```
