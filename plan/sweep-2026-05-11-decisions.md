# Sweep 2026-05-11 — Locked Decisions + Unified Execution Order

**Branch:** `main` | **Date:** 2026-05-11

This document locks the 12 user-deferred decisions from the engines + foundations
fix plans into definite choices, corrects a small set of inaccuracies in those
plans, and gives a unified execution order that supersedes the per-plan phase
breakdowns.

**Implementers: read this doc first**, then consult the per-item plan in
`sweep-2026-05-11-fixes-engines.md` or `sweep-2026-05-11-fixes-foundations.md`
for the technical detail on the locked option.

---

## Inaccuracies corrected in the per-item plans

These were fixed inline in the per-item plan files; this section just notes
the corrections so reviewers can grep for them.

| Item | Plan | Correction |
|---|---|---|
| Engines #1 | `sweep-…-fixes-engines.md` | Changed recommendation from brittle string-search update to a named marker `// @@PPG_GUIDE_EXTEND_COMBINED@@`, matching the existing `@@PPG_BOUNCE_INSERT@@` and `@@PPG_RECORD_INSERT@@` markers in the same file. Future shade.wgsl refactors will fail loud (marker-not-found) instead of silently. |
| Engines #4 | `sweep-…-fixes-engines.md` | Removed the speculative `nm.irridianceNode` recommendation (misspelled, and the hook does not exist in the TSL version vitrum uses). Locked in the pre-multiply approach: `nm.emissiveNode = mul(giNode, mul(materialColor, PI_INV))`. The math is correct because the GI input is already integrated irradiance E. |
| Engines #7 | `sweep-…-fixes-engines.md` | Locked Option A (one-line shader change to read `.x` instead of `.w`). Option B was contingent on a gbuffer-convention verification I could not complete with confidence; Option A is unconditionally safe. |
| Foundations #29 | `sweep-…-fixes-foundations.md` | Corrected the off-by-one in the prose: the current `stackPtr < 62u` guard wastes 1 stack slot (index 63), not 2. The recommended fix `stackPtr + 1u < 64u` is unchanged. |

---

## The 12 locked decisions

### D1. Engines #4 — RC GI BRDF injection

**Locked: pre-multiply by `albedo/π` before assigning to `emissiveNode`.**

Three.js TSL does not expose a clean `indirectDiffuseNode` / `irradianceNode`
hook in the version vitrum uses (verified by reading
`packages/walkaround-hybrid/src/lib/nodeMaterialUpgrade.ts`). The
pre-multiply approach is mathematically correct because the GI signal
arriving at the receiver is already integrated irradiance E (the cosθ weight
was applied during cascade-merge / DDGI-blend). The Lambertian receiver
equation is `L_o = (albedo/π) · E` — done.

### D2. Engines #7 — SVGF depth channel

**Locked: Option A (shader reads `.x`).**

One-line surgical fix in `svgf.wgsl.ts:216,248`. Aligns with the existing
host upload at `svgfWebGPU.ts:189,203` which writes depth into the `.r`
channel. If a later convention sweep normalizes all gbuffer depth to `.w`,
revisit then; for now do not chase a non-verified convention.

### D3. Engines #8 — SVGF temporal reuse

**Locked: rename, do NOT rewrite.**

Drop the Schied 2017 citation from `svgf.wgsl.ts` and rename:
- `SVGF_WGSL` → `ATROUS_VARIANCE_WGSL`
- `SVGF_DEFAULT_*` constants → `ATROUS_VARIANCE_DEFAULT_*`
- File `svgf.wgsl.ts` → `atrousVariance.wgsl.ts`
- File `svgfWebGPU.ts` → `atrousVarianceWebGPU.ts`
- File `svgfBindings.ts` → `atrousVarianceBindings.ts`
- File `svgfConstants.ts` → `atrousVarianceConstants.ts`
- Test files renamed to match.
- Update all imports across `shared-denoisers/`, `walkaround-hybrid/`,
  `pipeline/pipelineCompiler.ts`, etc. TypeScript compile catches all sites.
- The `denoiser: 'svgf'` mode in `HybridEngine` becomes `denoiser: 'atrous-variance'`.
  Keep `'svgf'` as a deprecated alias that warns once and routes to
  `'atrous-variance'`; remove the alias in a future sprint.

**Rationale:** what ships today is à-trous + a per-pixel variance scalar
lookup. The defining SVGF temporal-reuse pieces (bilinear reprojection,
disocclusion detection, per-pixel history length, variance-guided α-clamp,
paper Eq. 4 edge-stop form) are absent. Implementing real SVGF is weeks of
work for a marginal quality lift on top of the existing walkaround
indirect-temporal-accumulator (which already does Karis-style RGB-AABB
clipping). Per the project's "no SOTA-cargo-cult" principle, do the honest
rename now, schedule real SVGF as a future sprint *if* a render-quality gap
motivates it.

This decision subsumes Engines #9 (per-pixel Welford history): the global
`frameCount` Welford is acceptable for the renamed à-trous-variance pipeline.

Create `plan/sprint-svgf-real-future.md` capturing the design (motion
vectors, history-length texture, momentsHistory texture, disocclusion test
on depth+normal+objId) so the work can be revived if needed.

### D4. Engines #19 — Neural denoiser

**Locked: full delete.**

Delete:
- `packages/walkaround-hybrid/src/neural/` (entire directory)
- `packages/walkaround-hybrid/__tests__/sprint13-neural.test.ts`
- `tools/neural-denoiser-training/` (entire directory)

Remove the re-export of `InferenceGraph`, `unetArchitecture`, etc. from
`packages/walkaround-hybrid/src/index.ts`.

Create `plan/sprint-neural-denoiser-future.md` capturing the U-Net design
(channel widths, parameter budget, the 8 enumerated scaffold bugs to avoid
when re-implementing, the `train.py` exporter spec, OIDN-bridge alternative
worth comparing against) so the design work isn't lost.

**Rationale:** the current state is decorative scaffolding with shape
mismatches that prevent it running, no `'neural'` mode in `HybridEngine`,
no training pipeline (`train.py.md` is a `.md` file pretending to be code),
and pure string-grep tests. The architecture files are educational but
they're misleading-as-code. Half-implementations have been called out as
unwelcome by the user. Clean slate; revive when there's a real GPU
integration sprint scheduled.

### D5. Engines #20 — DDGI blend kernel

**Locked: Option A (paper-correct cosine kernel).**

Replace `pow(max(0, dot(dir, ray.direction)), 8.0)` with `max(0, dot(dir, ray.direction))`
in `probeUpdateBlend.wgsl.ts:130–134`. Same change to the visibility blend
weight (`pow(50)` → cosine) at line 212.

**Rationale:** physical correctness over visual polish. Atlas now holds true
irradiance, which is what the paper specifies and what the receiver equation
expects after Item 2 lands. The increased per-frame variance is mitigated by
Item 6 (random rotation) — both must land in the same DDGI milestone (see
unified order below).

### D6. Engines #21 — RC dimensional scaling

**Locked: Path A (fix merge integral, accept non-Sannikov dimensions).**

Keep the current cascade dimensions. Update the merge kernel
`cascadeMerge.wgsl.ts` to weight children by their actual solid-angle
coverage instead of assuming 4 children of equal weight summing / 4. Add a
header comment to `cascadePyramid.ts:31–37` explicitly documenting the
non-Sannikov scaling and the perf rationale.

**Rationale:** Path B (move to 2D Sannikov) is a multi-week architecture
change; Path A is a few-day merge-integral correction. The current dimensions
are tuned to the GPU budget; bringing them to Sannikov-faithful 3D would be
~10× slower at C0 and is incompatible with the WebGPU 30fps target.

### D7. Engines #25 — PPG

**Locked: full delete + future-sprint placeholder.**

Delete:
- `packages/walkaround-hybrid/src/ppg/` (entire directory)
- `packages/walkaround-hybrid/src/shaders/shadePpgGuide.wgsl.ts`
- `packages/walkaround-hybrid/src/shaders/shadePpgTrain.wgsl.ts`
- `packages/walkaround-hybrid/__tests__/ppgCellUpload.test.ts`
- `packages/walkaround-hybrid/__tests__/sprint11-ppg.test.ts`
- `packages/walkaround-hybrid/__tests__/sprint2-cellPower.test.ts` (cellPower
  exists for PPG)

Remove:
- `ppgEnabled` option from `HybridEngine` constructor / `setPPGEnabled` / etc.
- All PPG markers from `shade.wgsl.ts` (`@@PPG_TRAIN_BINDINGS_INSERT@@`,
  `@@PPG_GUIDE_DECLS_INSERT@@`, `@@PPG_BOUNCE_INSERT@@`,
  `@@PPG_RECORD_INSERT@@`).
- PPG bindings from `bindGroupLayouts.ts`, `pipelineCompiler.ts`,
  `resourceManager.ts`, `bindGroupBuilders.ts`, `uboUpdater.ts`.
- `cellPower` field from emitter packing in `restir/bvhCompute.ts:255–260`
  (it exists only for the deleted PPG light tree consumer per the comment
  block I updated earlier this session).

This deletion subsumes Engines #1 (PPG injector throw) entirely — there's no
injector to fix.

Create `plan/sprint-ppg-rebuild-future.md` capturing the 5 paper-faithful
requirements: adaptive sTree (Müller §3.1), adaptive dTree (Müller §3.2),
training on `L_i` not `L_o` (§3.3), MIS with BSDF (§3.4), per-bin
solid-angle weights from leaf area.

**Rationale:** the current PPG implementation deviates from Müller 2017 on
five independent axes. A partial implementation actively harms convergence
when enabled. Combined with #1 (injector crash) and #7 in the engines plan
(the test suite is structural, doesn't catch convergence regression), the
risk of "fix the crash and ship" is that PPG silently degrades quality on
any host that flips the flag. Clean slate is safer.

### D8. Foundations #11 — `bdptConnectionMIS` naming

**Locked: Option A (rename to `_partial` namespace).**

Rename:
- `bdptConnectionMIS` → `bdptConnectionMIS_partial`
- `buildBDPTStrategyPDFs` → `buildBDPTStrategyPDFs_partial`

Update the JSDoc to drop "Veach BDPT" framing in favor of "single-strategy
MIS aid for fork-side dispatch". Update test file names to match.

Create `plan/sprint-bdpt-veach-full-future.md` capturing the full §10.3
strategy enumeration (recursive `p_{s+1}/p_s` ratio per PBR4e Eq. 16.16,
specular-vertex zero-weight handling, area↔solid-angle G factor, camera/light
PDFs at endpoints) for when BDPT goes live.

**Rationale:** BDPT isn't wired anywhere; the export is honestly labeled as
a stub in JSDoc but the export name lies. The rename is a few-minute
mechanical change with TypeScript catching all import sites.

### D9. Foundations #15 — Multi-light MIS

**Locked: sum MIS over all area lights.**

Generalize `intersectRectAreaLightRay` and `intersectMeshAreaLightRay` to
accept a light index. In `bsdfAreaLightConnectionContribution`, iterate over
all `params.rectAreaLightCount` and `params.meshAreaLightCount` lights,
keeping the closest-distance hit (since BSDF samples a direction, only the
nearest light along that direction is "hit" by the BSDF sample). Multiply
the closest-hit's PDF by `lightCount` to cancel the uniform light-selection
probability, per the existing main-loop pattern.

**Rationale:** O(N_lights) extra intersection tests per bounce is acceptable
for the prototype (typical scenes have ≤ 8 area lights). Sum MIS is unbiased;
the balance-heuristic-per-randomly-chosen-light alternative is faster but
introduces selection variance.

### D10. Foundations #31 — pt-webgpu BVH builder

**Locked: implement binned SAH locally in `buildCpuBvh.ts`.**

Add `interface BinData`, `computeSAH(bins, splitIdx, totalSA)` helper, and a
binned-SAH loop with K=16 bins per axis. Carry `rightChildOrTriOffset` as a
relative offset (per Item 26 fix). Keep the file otherwise untouched.

**Rationale:** delegating to `shared-bvh` would add a `three` transitive
dependency to `pt-webgpu`, contradicting its stated goal of being a
host-agnostic backend. Local binned SAH is ~150 lines of TypeScript per
Wald 2007.

### D11. Foundations #35 — Spectral curve `Float32Array` deprecation

**Locked: remove the deprecated path now.**

Delete the `Float32Array` branch in `packages/three-bindings/src/material.ts:142–157`.
Tighten the type guard to require `SpectralCurve` shape (`wavelengthStart`,
`wavelengthEnd`, `values` all present). Remove the `console.warn` since the
path no longer exists.

Update the test file
`packages/three-bindings/src/__tests__/material-vitrum-roundtrip.test.ts:test('reads
vitrumSpectralAttenuation as raw Float32Array (back-compat)')` to either
delete the test or assert that the old shape is rejected.

Add a CHANGELOG entry under `### Removed`.

**Rationale:** vitrum is pre-alpha with no external consumers of the
deprecated path. Removing now is the right time. "Phase 7 / Sprint 1"
language was already stale; the project is in Phase 7 mid-sprint.

### D12. Foundations #36 — `TRI_INTERSECT_EPSILON` UBO plumbing

**Locked: UBO-plumb.**

Add `triIntersectEpsilon: f32` to `WalkaroundUBO` (and `FrameParams` for
pt-webgpu). Default = `1e-5` (current value, metre-scale). Replace
`const TRI_INTERSECT_EPSILON: f32 = 1e-5;` in
`walkaround-hybrid/src/shaders/common.wgsl.ts:30` and
`pt-webgpu/src/wgsl/common.wgsl.ts:16` with reads from the UBO.

Leave `safe_normalize` floor `1e-8` as a constant — it is a vector-length
floor, not geometry-scale dependent. (The agent's concern about `1e-8`
being below f32 sig-figs is unfounded for length comparisons; `1e-8` is
~7 orders below 1.0, comfortably in f32 representable range.)

**Rationale:** the project's library-generality remediation pattern has
been to UBO-plumb tunables aggressively (commits `0088a78`, `bd7a7d1`).
This is consistent with that pattern. The UBO layout bump is a
single-field addition with no alignment risk.

---

## Three small items the per-item plans missed

| Item | Status | Action |
|---|---|---|
| Two locked worktree branches (`worktree-agent-a06812aa6c5b09b98`, `worktree-agent-a193d8b569f806c50`) | Sub-agent execution state, low priority | Leave alone. If they remain after the 38-item work completes, hand-clean then. |
| 3 tools/*.md docs referencing `plan/phase-6-roadmap.md` (non-archive path) | Updated this session | DONE — all 3 now point to `plan/archive/phase-6-roadmap.md`. |
| Items 2 + 4 + 6 + 20 should land coherently as one DDGI milestone | Captured in execution order below | See "DDGI Coherent Milestone" phase. |

---

## Unified execution order

This supersedes the per-plan phase breakdowns. Items are grouped into
milestones; within each milestone, sub-items can land in any order or
parallel. Milestones land sequentially; do not skip ahead until the previous
milestone's tests are green.

### Milestone 1 — Quick wins + cleanups (no behavior change)

Land as a single PR. Pure rename / docstring / comment / structural changes.

| Item | Action |
|---|---|
| Engines #11 (D8) | Rename `bdptConnectionMIS` → `bdptConnectionMIS_partial` |
| Engines #12 | Drop Estévez-Kulla 2018 from `lightTree.ts` references |
| Engines #13 | Rename `nodePowerPrefixSum` → `_powerPrefixSumDebug`, mark `@internal` |
| Foundations #32 | Cross-link the two RFE tracker files |
| Foundations #34 | Replace "future sprint" language in `HybridEngine.ts:592` |
| Foundations #38 | Update `_staging/README.md` table to be accurate |

Concurrent: create the three future-sprint placeholders:
- `plan/sprint-svgf-real-future.md` (D3)
- `plan/sprint-neural-denoiser-future.md` (D4)
- `plan/sprint-ppg-rebuild-future.md` (D7)
- `plan/sprint-bdpt-veach-full-future.md` (D8)

### Milestone 2 — Decisive deletes (D3 rename, D4 + D7 deletes)

Land each as its own PR for clean revert ability. After each, `npm test` and
`npm run typecheck` must remain green.

1. **D3 — Rename SVGF → atrous-variance** across `shared-denoisers`,
   `walkaround-hybrid`, `pipeline/`, tests.
2. **D4 — Delete neural denoiser scaffold** (`neural/` dir, sprint13 test,
   `tools/neural-denoiser-training/`, `index.ts` re-exports).
3. **D7 — Delete PPG** (`ppg/`, `shadePpgGuide.wgsl.ts`,
   `shadePpgTrain.wgsl.ts`, related tests, `HybridEngine` `ppgEnabled`
   API, all PPG markers in `shade.wgsl.ts`, PPG bindings throughout
   `pipeline/`, `cellPower` from emitter packing).

Subsumes Engines #1 (PPG injector crash — no longer reachable).

### Milestone 3 — Numerical test infrastructure (Foundations #33)

Land **before** any algorithmic-correctness items (Milestone 4+) so the new
tests can validate the fixes.

Implementation order per Foundations #33:
1. **33-D Energy conservation (white furnace)** — gates BSDF correctness
2. **33-A PDF normalizations** — HG, equiAngular, mixturePdf, octahedral, environment
3. **33-E Octahedral encode/decode round-trip**
4. **33-F Light-tree leaf PDF sums to 1**
5. **33-H Numerics (half-float overflow, safeInvDir, frDielectric bounds)**
6. **33-G BVH cross-package compatibility round-trip**
7. **33-I Dielectric Fresnel bounds** (re-prioritized to land alongside Item 16)
8. **33-B VNDF PDF normalization** (re-prioritized to land alongside Item 14)
9. **33-C MC convergence** (Lambertian sphere, single-bounce diffuse) — last; needs
   the CPU mini-ray-tracer scaffolding

### Milestone 4 — Surgical correctness fixes

Order within milestone is flexible; land each as its own small PR.

| Item | Action |
|---|---|
| Engines #5 | ReSTIR-DI p̂ consistency: add `emitterGeometry(...)` to `temporal.wgsl.ts:96` and `spatial.wgsl.ts:77` |
| Engines #7 (D2) | Change SVGF (now atrous-variance) shader depth read from `.w` to `.x` (lines 216, 248) |
| Engines #10 | Fix `equiAngular.ts:135` PDF/sample mismatch (use clamped `t` in PDF computation) |
| Foundations #17 | Add `transformNormal((M⁻¹)ᵀ)` to `mat4.ts`; replace `transformDirection` calls for normals |
| Foundations #18 | Remove `min(hit.dist, 32.0)` Beer-Lambert clamp |
| Foundations #28 | Add `safeInvDir` helper; replace 5 `1/dir` sites |
| Foundations #29 | Fix BVH stack guard arithmetic (`stackPtr + 1u < 64u`) |
| Foundations #36 (D12) | UBO-plumb `triIntersectEpsilon` |
| Foundations #35 (D11) | Delete deprecated `Float32Array` spectral path |

### Milestone 5 — pt-webgpu BSDF correctness

Touches `pathTraceBruteforce.wgsl.ts` heavily. Land as one PR with reference
render before/after.

1. **Foundations #14** — Replace `glossyReflectionSample` with Heitz 2018
   VNDF (full WGSL outline in foundations plan Item 14). Add 33-B test.
2. **Foundations #16** — Add `frDielectric`, replace heuristic dielectric
   branch with Fresnel-weighted split. Add 33-I test.
3. **Foundations #15 (D9)** — Generalize area-light intersect helpers to
   loop all lights with sum MIS.

### Milestone 6 — pt-webgpu BVH consolidation

Order matters: encoding before SAH before traversal optimizations.

1. **Foundations #26** — Canonicalize pt-webgpu BVH to relative right-child
   offsets (atomic update of builder + traversal).
2. **Foundations #27** — Formalize index buffer stride contracts; add
   upload-time assertions.
3. **Foundations #31 (D10)** — Replace median-split BVH with binned SAH in
   `buildCpuBvh.ts`.
4. **Foundations #30** — Add ordered (split-axis) BVH traversal to
   walkaround-hybrid `common.wgsl`.

### Milestone 7 — DDGI Coherent Physical Model (Items 2 + 6 + 20 + 4)

These four items jointly redefine the DDGI energy model. They MUST land in
a single PR with a reference render before/after — landing them piecemeal
will produce intermediate visual states that are demonstrably wrong and
will trigger band-aid attempts. Order within the PR:

1. **Engines #6** — Unfreeze `randomRotation` (Halton-based SO(3) per
   frame).
2. **Engines #20 (D5)** — Replace `pow(w, 8)` with paper Lambertian cosine.
3. **Engines #2** — Remove `* albedo / PI` from producer
   (`probeUpdateRays.wgsl.ts:547`); atlas now holds true irradiance.
4. **Engines #4 (D1)** — Fix RC GI BRDF injection in `giReceiver.ts:103`
   (pre-multiply `albedo/π` before `nm.emissiveNode`).
5. Remove the `disable DDGI gain` band-aid from commit `3fb63e3`.

Reference scene: white Cornell box + grey wall variant. Verification:
indirect illumination on the grey wall should be ~half the white wall
(albedo proportionality). Total energy should be conserved (no greater than
direct illumination).

### Milestone 8 — DDGI border padding (Item 3)

Standalone after Milestone 7. Adds new compute pass.

1. **Engines #3** — Add `probeUpdateBorder` compute pass that runs after
   `probeUpdateBlend`. Iterates each probe's border ring, copies from the
   octahedral-mirrored interior texel. Use scratch buffer pattern (collect
   border values during blend, write in border pass) to avoid WebGPU's
   read+write-on-same-texture restriction.

### Milestone 9 — RC + GTAO physical fixes

| Item | Action |
|---|---|
| Engines #22 | Replace RC `4π/N` normalization with per-bin solid-angle weights from octahedral grid |
| Engines #21 (D6) | Fix RC merge integral to weight by actual child solid-angle coverage (Path A) |
| Engines #23 | Replace GTAO `(h1+h2)/π` with full Jiménez 2016 slice integral |
| Engines #24 | Add albedo demodulation (divide by albedo before atrous, multiply back after) — depends on Engines #7 |

### Milestone 10 — CHANGELOG + final docs

| Item | Action |
|---|---|
| Foundations #37 | Catch up CHANGELOG.md with Sprint 12–18 entries (use git log to enumerate) |

---

## Cross-cutting concerns

### Reference rendering protocol

Per `CLAUDE.md` Testing protocol: every milestone that changes algorithmic
behavior must capture before/after reference renders of:
- Cornell box (the canonical regression scene at `examples/cornell-box/`)
- Multi-material scene if available

Renders go to `tools/reference-renders/<milestone-name>/{before,after}.png`.
A/B them. Numerical regression is acceptable only if visually justified.

### Rollback contract

Each milestone PR must be revertible cleanly. If a milestone introduces a
visual regression that can't be diagnosed within a day, revert the milestone
and re-plan rather than chasing band-aids on top of a broken physical
model. The `3fb63e3 disable DDGI gain` commit is the cautionary example —
do not repeat that pattern.

### Testing gates

After each milestone:
1. `npm run typecheck` clean across the workspace.
2. `npm test` green (test count is allowed to *increase*; never to decrease).
3. Reference render diff if applicable.
4. Update `memory/in-flight-sweep.md` to mark the milestone's items as
   resolved.

### Future-sprint placeholders to create (during Milestone 1)

- `plan/sprint-svgf-real-future.md` — D3 design notes
- `plan/sprint-neural-denoiser-future.md` — D4 architecture notes (the U-Net
  spec from the deleted `unetArchitecture.ts` is worth preserving)
- `plan/sprint-ppg-rebuild-future.md` — D7 paper-faithful requirements
- `plan/sprint-bdpt-veach-full-future.md` — D8 Veach §10.3 enumeration

These docs prevent the design work from being lost; they live in `plan/`
(not `plan/archive/`) because they describe future, not completed, work.
