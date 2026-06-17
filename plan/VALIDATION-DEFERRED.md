# Deferred validation — code-first campaign

> **Policy (2026-06-16):** Implementation waves run **without** GPU or rendering
> validation. All radiometric A/B, golden PNG, reference-render capture, behavioral-gate
> render boots, and wsl-gpu harness work happens **after** the code remediation branch
> merges or in a follow-up validation sprint.

## What is deferred

| Category | Examples | Task IDs |
|----------|----------|----------|
| V28-B GPU recapture | All render-changing landings | `RT100-V28-*`, `RT100-V19-GRIS` |
| Golden / Khronos sweep | PNG hash, real-asset render | `RT100-5A-GOLDEN` |
| GPU mutation observability | wsl-gpu buffer watch | `RT100-5C-GPU-MUT` |
| Full-tier behavioral gate | Lite-only lavapipe today | `RT100-GATE-FULL` |
| Material promotion proof | Furnace, reference A/B | `RT100-WA-3E`, `RT100-PTWG-FURNACE`, `RT100-A9-BDPT` |
| Quality decisions | NRC default, neural weights | `RT100-A6-DECIDE`, `RT100-A10-WEIGHTS` |
| Code-gap render tails | Kulla furnace, gate sweep | `PTWG-017`, `PTWG-080`, `TOOL-004` |

## What still runs

- `npm run typecheck`
- `npm test` / package vitest (unit + oracle tests in CI)
- `npm run shader-gate` (WGSL/GLSL **compile** gate — not radiometric A/B)
- P0 `P0-008-TOOL-001` (adds pt-webgl2 **configs** to gate — code change, not full render proof)

## Scheduler behavior

`tools/gap-scan/validation-deferral.mjs` excludes deferred tasks from
`code-gap-parallel-schedule.json`. They remain in the task register for audit but
**no agent is dispatched** for them during the code-first run.

## After code merge

1. Re-enable tasks (remove from deferral set or run `plan/validation-sprint.md` TBD).
2. Recapture via `~/projects/wsl-gpu` per `HARDWARE-VALIDATION-NEEDS.md`.
3. Promote `approximate` ledger rows with evidence.
