# External feature requests (RFEs)

Numbered specs propose **API and algorithm** extensions to vitrum. They are **not** implemented until triaged and scheduled.

## Index

| # | Document | Status | Topic (short) |
|---|----------|--------|----------------|
| 01 | [01-spectral-rendering.md](./01-spectral-rendering.md) | Accepted | Spectral attenuation/dispersion in progress (runtime closure pending) |
| 04 | [04-multilayer-thinfilm.md](./04-multilayer-thinfilm.md) | Accepted | Multi-layer thin film (TMM) in progress |
| 08 | [08-sprint12-spectral-accumulator-fork-patch.md](./08-sprint12-spectral-accumulator-fork-patch.md) | Accepted | Sprint 12 spectral accumulator integration in progress |

**Status values:** `Proposed` → `Accepted` (in scope, target phase set) → `Implemented` | `Deferred` (reason + link) | `Rejected` (reason). RFEs that reach `Implemented` are removed from this directory and recorded in [`IMPLEMENTATION-STATUS.md`](./IMPLEMENTATION-STATUS.md).

## How these enter the product

Follow **[plan/cursor-recommended-plan.md](../plan/cursor-recommended-plan.md) §13** — summary:

1. Triage and set **Status** in the table above (and optional note in the RFE frontmatter).
2. Extend **`@vitrum/core`** (prefer `extensions` until two backends ship).
3. Update **`@vitrum/three-bindings`** where THREE maps exist.
4. Implement **per backend** (`pt-webgl` fork first for PT truth; walkaround only if the RFE allows realtime approximations).
5. Add **benchmark + reference renders**.
6. Update **[plan/phase-6-roadmap.md](../plan/phase-6-roadmap.md)** mode matrix when the feature lands.

## Quick link: plan mapping

[`plan/cursor-recommended-plan.md`](../plan/cursor-recommended-plan.md) **§13** maps each file to contract touchpoints, mode scope, **Phase 0–A (Tier 1 contract)** vs **Phase B–C (implementation)**, **Sprints 4–9** plus decision gates — not a late **Phase D** default. **Phase D** is only for explicit `pt-webgpu`-only or frontier-deferred work.

Key fork sprint specs:
- [`plan/sprint-7-pt-fork-patch.md`](../plan/sprint-7-pt-fork-patch.md)
- [`plan/sprint-8-pt-fork-patch.md`](../plan/sprint-8-pt-fork-patch.md)
- [`plan/sprint-12-pt-fork-patch.md`](../plan/sprint-12-pt-fork-patch.md)
