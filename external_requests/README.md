# External feature requests (RFEs)

Numbered specs propose **API and algorithm** extensions to vitrum. They are **not** implemented until triaged and scheduled.

## Index

| # | Document | Status | Topic (short) |
|---|----------|--------|----------------|
| 01 | [01-spectral-rendering.md](./01-spectral-rendering.md) | Proposed | Spectral attenuation, dispersion / hero-λ |
| 02 | [02-volume-scattering.md](./02-volume-scattering.md) | Proposed | Henyey–Greenstein volume scattering |
| 03 | [03-layered-bsdf.md](./03-layered-bsdf.md) | Proposed | Front/back asymmetric layered BSDF |
| 04 | [04-multilayer-thinfilm.md](./04-multilayer-thinfilm.md) | Proposed | Multi-layer thin film (TMM) |
| 05 | [05-manifold-nee.md](./05-manifold-nee.md) | Proposed | Manifold NEE / photon map for caustics |

**Status values:** `Proposed` → `Accepted` (in scope, target phase set) → `Implemented` | `Deferred` (reason + link) | `Rejected` (reason).

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
