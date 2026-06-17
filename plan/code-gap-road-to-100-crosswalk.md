# Code-Gap ↔ Road-to-100 Crosswalk

> Maps `plan/road-to-100.md` open tails to executable `RT100-*` tasks in
> `tools/gap-scan/road-to-100-tasks.mjs`. Regenerate schedules after registry edits:
> `node tools/gap-scan/generate-implementation-plan.mjs`

## Scope split

| Track | Source | Phases | Purpose |
|-------|--------|--------|---------|
| **Code-gap** | Line-scan audit (`code-gap-exhaustive-audit.md`) | 0–7 | Bugs, ledger drift, missing tests, per-field MAT tasks |
| **Road-to-100** | `plan/road-to-100.md` | 8–12 | Validation (V28-B), promotion, decisions, SOTA perf |

Phases 0–6 of road-to-100 are **mostly landed**; remaining work is validation tails,
promotion evidence, and post-100 SOTA — not re-implementation of closed rows.

## Bucket A–D → RT100 tasks

| road-to-100 | Status in code | RT100 task(s) | Notes |
|-------------|----------------|---------------|-------|
| **A1** ReSTIR-PT | ✅ impl | `RT100-V28-A1` | Equal-spp variance A/B |
| **A2** PPG | ✅ impl | `RT100-V28-A2` | Multi-region guiding A/B |
| **A3** Spectral | ✅ impl | `RT100-V28-A3` | Radiometric A/B |
| **A4** SPPM progressive | ✅ impl | `RT100-V28-A4` | GPU validation |
| **A5** BDPT coherence | ✅ impl | `RT100-V28-A5`, `RT100-A9-BDPT` | V28 + radiometric oracle |
| **A6** NRC | ✅ impl, opt-in | `RT100-A6-DECIDE` | Quality A/B + default tier |
| **A7** RC | ✅ impl | `RT100-V28-A7` | Cascade A/B N=16/64 |
| **A8/F6** GRIS default | ✅ decided biased | `RT100-V19-GRIS` | Evidence before flip |
| **A9** BDPT parallel | ✅ serial build | `RT100-A9-BDPT` | Oracle + optional parallel IMP |
| **A10** Neural weights | ✅ pipeline | `RT100-A10-WEIGHTS`, `RT100-F3-DENO-AUTO` | Production checkpoint |
| **B1** Glass GI | ✅ impl | `RT100-V28-B1` | GPU A/B |
| **B2** DDGI glossy | ✅ impl | `RT100-V28-B2` | Metallic-sphere A/B |
| **B4** Mesh-area NEE | ✅ impl | `RT100-V28-B4`, `RT100-EMISSIVE-PDF` | V28 + texel alias tail |
| **B7** SBVH | planar fix only | `RT100-WBVH-01` | Optional perf track |
| **B8** Light-tree cones | ✅ impl | `RT100-V28-B8` | Directional A/B |
| **B13–B16** | ✅ impl | `RT100-V28-B15`, `RT100-V28-B16` | V28 recapture |
| **D bucket** | ✅ closed | — | D6 bind-group churn closed |

## Phase 2–3 → RT100 tasks

| road-to-100 section | Code-gap overlap | RT100 task(s) |
|---------------------|------------------|---------------|
| **§2C Adjoint OPEN** | `INV-*`, `P0-008` | `RT100-ADJ-001` |
| **§2C PTWG-MAT-01** | `PTWG-*`, `MAT-*` | `RT100-PTWG-MAT`, `RT100-PTWG-FURNACE` |
| **§2D pt-webgl2 mutations** | `PTGL-*`, `MUT-*` | `RT100-PTGL-MUT` |
| **§3C Alpha/ReSTIR GI** | `WH-*`, `MAT-WH-*` | `RT100-WA-ALPHA` |
| **§3D Atlas** | `MAT-WH-*` (generic) | `RT100-WA-3D` (concrete tail) |
| **§3E Extension lobes** | `MAT-WH-*` | `RT100-WA-3E` (promotion VERIFY) |
| **§3F Permanent unsupported** | `ACC` tasks in MAT | No RT100 — planner routes to PT |

**Important:** Generic `MAT-WH-*` tasks promote fields one-by-one. Road-to-100 Phase 3
commits to **full atlas scope**; `RT100-WA-3D/3E` are the authoritative epic tasks for
walkaround material 100%.

## Phase 4–5 → RT100 tasks

| road-to-100 | RT100 task |
|-------------|------------|
| **§4A pickBackend** (done for glTF) | `RT100-GLTF-PICK` — verify rankGltfBackends in scale path |
| **§5A Golden PNG + real assets** | `RT100-5A-GOLDEN` |
| **§5C GPU mutation observability** | `RT100-5C-GPU-MUT` |
| **§5D Doc sync** | `RT100-5D-DOC` |
| **§5E Full-tier behavioral gate** | `RT100-GATE-FULL` |
| **§0.3 V28-B baseline** | `RT100-V28-000` + per-bucket `RT100-V28-*` |

## Phase 0.2 gates → code-gap

| Gate | Status | Covered by |
|------|--------|------------|
| GATE-01 | ✅ | `CORE-*` ledgerVsCapabilities tests |
| GATE-02 | ✅ | `CORE-*` materialNativeEvidence |
| GATE-06 | ✅ | `TOOL-003` shader-gate |
| GATE-GLTF analyze | ✅ | glTF tests |
| GATE-GLTF **render** | ◻ | `RT100-5A-GOLDEN`, `RT100-GATE-FULL` |

## Post-100 SOTA (F1–F6, F-BRIDGE)

| Item | RT100 task | Blocker for glTF 100%? |
|------|------------|------------------------|
| F1 LD-SAMPLING | `RT100-LD-SAMPLING-01` | No — perf track |
| F2 WBVH | `RT100-WBVH-01` | No — perf track |
| F3 denoiser:auto | `RT100-F3-DENO-AUTO` | UX / provisioning |
| F4 wavefront | `RT100-F4-WAVEFRONT` | No — research ACC |
| F5 volumes | `RT100-F5-VOLUMES` | No — product gated |
| F6 GRIS flip | `RT100-V19-GRIS` | Decision after A8 evidence |
| F-BRIDGE table | `RT100-F-BRIDGE` | No — research backlog |

## P0 code-gap ↔ road-to-100

| P0 task | road-to-100 ref |
|---------|-----------------|
| `P0-001-PTWG-037` | §2E lite `updatePrimitive` |
| `P0-002-PTGL-003` | §2D mesh-area repack |
| `P0-003-WH-034` | §3D DDGI beer invalidation |
| `P0-004-DENO-001` | Bucket D / denoiser union |
| `P0-005–007 LEDGER` | §5D ledger truth |
| `P0-008-TOOL-001` | §5E behavioral gate pt-webgl2 |

## Orchestrator notes

- **VERIFY / DECIDE waves:** Many RT100 tasks are **SKIP** in code-first mode — see `plan/VALIDATION-DEFERRED.md`.
- **Phase 8+ waves** run after code-gap phases 0–7 complete.
- **Do not duplicate:** Skip `MAT-WH-*` when `RT100-WA-3D/3E` already owns the same file
  in the same wave (scheduler file mutex handles this).

## Stale doc hygiene

road-to-100.md addendum still lists **H1–H5 pt-webgl2 inert** — items_to_fix marks these
**CLOSED 2026-06-09**. `RT100-5D-DOC` must reconcile that prose.
