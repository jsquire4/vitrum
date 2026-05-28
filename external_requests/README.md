# External feature requests (RFEs)

Numbered specs propose **API and algorithm** extensions to vitrum. They are **not** implemented until triaged and scheduled.

## Index

| # | Document | Status | Topic (short) |
|---|----------|--------|----------------|
| 01 | [01-spectral-rendering.md](./01-spectral-rendering.md) | Implemented (GPU A/B pending) | Spectral attenuation + dispersion: fork Beer–Lambert uses packed μ(λ); GPU acceptance still pending |
| 04 | [04-multilayer-thinfilm.md](./04-multilayer-thinfilm.md) | Implemented | Multi-layer thin film (TMM) — core contract + pt-webgl fork TMM applied |
| 08 | [08-sprint12-spectral-accumulator-fork-patch.md](./08-sprint12-spectral-accumulator-fork-patch.md) | Implemented | Sprint 12 spectral accumulator — fork is the absorbed monorepo package |
| 09 | [09-runtime-lighting-updates.md](./09-runtime-lighting-updates.md) | Partially Implemented (updateLighting batch method shipped; per-property methods + capability flag not added) | Runtime lighting updates without pipeline rebuild (`updatePrimaryLight`, `updateSkyDome`, `updateLights`) |
| 10 | [10-surface-texture-id-contract.md](./10-surface-texture-id-contract.md) | Informational / no-action | SURFACE_TEXTURE_ID wire-format contract |
| 11 | [11-dichroic-lut-consumption.md](./11-dichroic-lut-consumption.md) | Implemented | Dichroic LUT consumption in `@vitrum/three-bindings` |

**Status values:** `Proposed` → `Accepted` (in scope, target phase set) → `Implemented` | `Deferred` (reason + link) | `Rejected` (reason). RFEs that reach `Implemented` are removed from this directory and recorded in [`IMPLEMENTATION-STATUS.md`](./IMPLEMENTATION-STATUS.md).

## How these enter the product

Follow **[plan/archive/cursor-recommended-plan.md](../plan/archive/cursor-recommended-plan.md) §13** — summary:

1. Triage and set **Status** in the table above (and optional note in the RFE frontmatter).
2. Extend **`@vitrum/core`** (prefer `extensions` until two backends ship).
3. Update **`@vitrum/three-bindings`** where THREE maps exist.
4. Implement **per backend** (`pt-webgl` fork first for PT truth; walkaround only if the RFE allows realtime approximations).
5. Add **benchmark + reference renders**.
6. Update **[plan/archive/phase-6-roadmap.md](../plan/archive/phase-6-roadmap.md)** mode matrix when the feature lands.

## Quick link: plan mapping

[`plan/archive/cursor-recommended-plan.md`](../plan/archive/cursor-recommended-plan.md) **§13** maps each file to contract touchpoints, mode scope, **Phase 0–A (Tier 1 contract)** vs **Phase B–C (implementation)**, **Sprints 4–9** plus decision gates — not a late **Phase D** default. **Phase D** is only for explicit `pt-webgpu`-only or frontier-deferred work.

Key fork sprint specs:
- [`plan/archive/sprint-7-pt-fork-patch.md`](../plan/archive/sprint-7-pt-fork-patch.md)
- [`plan/archive/sprint-8-pt-fork-patch.md`](../plan/archive/sprint-8-pt-fork-patch.md)
- [`plan/archive/sprint-12-pt-fork-patch.md`](../plan/archive/sprint-12-pt-fork-patch.md)
