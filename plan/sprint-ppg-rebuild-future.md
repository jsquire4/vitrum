# PPG Rebuild — Future Sprint Placeholder

**Status:** Future / unscheduled
**Created:** 2026-05-11
**Context:** This functionality was deleted during sweep-2026-05-11 because the existing
implementation was not paper-faithful. This doc preserves the design intent for when a
real implementation sprint is scheduled.

---

## What was deleted and why

The following were removed (D7, decisions doc):

- `packages/walkaround-hybrid/src/ppg/` (entire directory: `buildPpgKdTree.ts`,
  `ppgCellUpload.ts`, `types.ts`, and WGSL: `ppgUpdate.wgsl.ts`, `ppgCommon.wgsl.ts`)
- `packages/walkaround-hybrid/src/shaders/shadePpgGuide.wgsl.ts`
- `packages/walkaround-hybrid/src/shaders/shadePpgTrain.wgsl.ts`
- PPG-related tests: `ppgCellUpload.test.ts`, `sprint11-ppg.test.ts`, `sprint2-cellPower.test.ts`
- `ppgEnabled` constructor option, `setPPGEnabled()`, and all PPG markers
  (`@@PPG_TRAIN_BINDINGS_INSERT@@`, `@@PPG_GUIDE_DECLS_INSERT@@`, `@@PPG_BOUNCE_INSERT@@`,
  `@@PPG_RECORD_INSERT@@`) from `shade.wgsl.ts`
- PPG bindings from `bindGroupLayouts.ts`, `pipelineCompiler.ts`, `resourceManager.ts`,
  `bindGroupBuilders.ts`, `uboUpdater.ts`
- `cellPower` field from emitter packing in `restir/bvhCompute.ts:255–260`

The deletion was necessary because the implementation deviated from Müller 2017 on five
independent axes (see below), actively crashing before it ran (Item 1 injector throw),
and the test suite was purely structural. A partial implementation that guides on the
wrong signal harms convergence when enabled — the clean-slate decision is safer.

---

## Paper reference

Müller, T., Gross, M., Novák, J.
"Practical Path Guiding for Efficient Light-Transport Simulation."
_Eurographics Symposium on Rendering_, 2017.
https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf

Key sections: §3.1 (spatial tree), §3.2 (directional tree), §3.3 (training signal),
§3.4 (MIS with BSDF), and the supplemental for the GPU-side update pseudocode.

---

## The 5 paper-faithful requirements

The deleted implementation got all 5 wrong. A re-implementation must satisfy all 5
before the feature is wired into `HybridEngine`.

### Requirement 1 — Adaptive spatial tree (sTree), Müller §3.1

**What the paper requires:**
The spatial domain is partitioned by a binary kd-tree. Cells are split adaptively when
accumulated sample variance exceeds a threshold — specifically, when the sum of squared
radiance estimates at samples within the cell exceeds `sTreeThreshold`. After splitting,
both children inherit the parent's directional distribution (dTree). The tree is rebuilt
periodically (typically once per iteration, or once per N frames in real-time mode) and
is NOT static after initialization.

**What the deleted implementation did:**
Built a static kd-tree once from a 4×4×4 uniform grid. No variance tracking, no splits,
no rebuilds. This is not an adaptive spatial tree — it is a fixed uniform grid.

**Implementation requirements:**

- Per-cell sample counter and accumulated-variance scalar (atomic GPU buffer).
- Split decision: if `sum_variance(cell) > sTreeThreshold`, split along the longest axis
  of the cell's bounding box and initialize two child dTrees from the parent dTree.
- Tree serialization: the sTree must be serialized to a flat GPU buffer each rebuild
  cycle (fixed-size per cell; maximum cell count capped at a host-configurable limit).
- `sTreeThreshold` should be a UBO-plumbed tunable; Müller uses 4000 samples/cell.

### Requirement 2 — Adaptive directional tree (dTree), Müller §3.2

**What the paper requires:**
Each sTree cell has its own dTree — a 5D binary tree over (position × direction) that
is refined based on directional radiance flux. The dTree is represented as a full binary
tree over the unit sphere, starting with a small fixed number of leaves and splitting
high-flux leaves (those whose accumulated flux exceeds `dTreeThreshold × totalFlux`).
The split axis alternates between the two spherical coordinate axes (θ and φ) or uses
the axis that produces the most balanced subdivision.

The defining data structure is the **quadtree / binary tree over octahedral UV** in 2D:
each leaf covers a solid-angle patch on the sphere, and accumulates flux from incident
radiance estimates. High-flux leaves are split; low-flux leaves may be merged.

**What the deleted implementation did:**
Used a fixed 4×4 octahedral grid (16 bins) for all sTree cells. No directional splits,
no flux-fraction-driven refinement. 16 bins cannot represent sharp indirect caustics —
the angular resolution is ~90° per bin.

**Implementation requirements:**

- Per-cell dTree stored as a binary tree over [0,1]² (octahedral map). Starting depth:
  2 (4 leaves for a 2-level tree). Maximum depth: configurable (8 → 256 leaves).
- Flux accumulation: per-leaf atomic float buffer, incremented during the training pass.
- Split rule: `leaf.flux > dTreeThreshold × totalCellFlux`. `dTreeThreshold` is
  typically 0.01 (split any leaf carrying > 1% of total cell flux).
- Merge rule: `leaf.flux < mergeThreshold × totalCellFlux` and leaf is at depth > 1.
- Rebuild cycle: flush leaf accumulators → compute split/merge decisions → emit new tree
  → reset accumulators.

### Requirement 3 — Train on `L_i`, not `L_o`, Müller §3.3

**What the paper requires:**
The guiding PDF is trained on **incoming radiance** at the sampled point — the radiance
arriving at a surface point from a given direction, weighted by the BSDF. The training
signal is the per-sample product estimate:

```
w_sample = f(ωo, ωi) · L_i(x, ωi) · cos(θi) / p(ωi)
```

where `f(ωo, ωi)` is the BSDF, `L_i` is the incoming radiance from direction `ωi`
(estimated by continuing the path), `θi` is the angle to the normal, and `p(ωi)` is the
current sampling PDF. This signal is deposited into the dTree cell corresponding to
direction `ωi`.

**What the deleted implementation did:**
Trained on `L_o` — outgoing radiance from the shade pass. `L_o` is post-BRDF and
includes the cosine weight and albedo; training on it produces a PDF that is biased
toward surface reflectance rather than the light field. A white wall and a black wall
with identical illumination would have different guide PDFs under the deleted scheme,
even though the incoming light field is the same.

**Implementation requirements:**

- The training signal must be deposited at the _sample point_ (where the ray was born),
  in the direction of the _incoming_ radiance (the direction toward the next-bounce hit
  or light source).
- The deposit value is the path throughput estimate at that bounce, not the shade output.
- In the walkaround context, this requires tapping the indirect illumination estimate
  before the BRDF multiply, which means the training pass must run during the shade
  loop, not after it.

### Requirement 4 — MIS with BSDF, Müller §3.4

**What the paper requires:**
Importance sampling at each bounce mixes the guide PDF (learned dTree) with the BSDF
sampling PDF using multiple importance sampling:

```
p_mixed(ωi) = α · p_guide(ωi) + (1 − α) · p_bsdf(ωi)
```

where α is a mixing weight (typically 0.5 or adapted per-cell based on guide quality).
The MIS weight for a guide-sampled direction is:

```
MIS_guide = p_mixed(ωi) / p_guide(ωi)
```

and the throughput is divided by `p_mixed(ωi)` (not just `p_guide(ωi)`).

**What the deleted implementation did:**
Used the guide PDF alone to weight samples, without MIS. This produces a biased estimator
when the guide PDF is not yet converged (which is always the case in the early frames).
MIS with the BSDF provides a safety net: even if the guide PDF is completely wrong, the
BSDF term ensures the estimator remains unbiased.

**Implementation requirements:**

- Compute `p_guide(ωi)` from the dTree at the sample point and direction.
- Compute `p_bsdf(ωi)` from the BSDF PDF at the same direction.
- Mix: `p_mixed = α · p_guide + (1 − α) · p_bsdf`.
- Divide throughput by `p_mixed`, not by `p_guide` alone.
- `α` should start at 0 (pure BSDF) and ramp toward 0.5 as the guide converges. Track
  guide quality per sTree cell (e.g., variance ratio or iteration count).

### Requirement 5 — Per-bin solid-angle weights from leaf area, Müller §3.2

**What the paper requires:**
Each dTree leaf covers a specific solid-angle patch on the sphere. The patch solid angle
is determined by the leaf's position in the octahedral tree — leaves near the octahedral
fold have smaller solid angle than equatorial leaves. The normalization factor when
drawing a sample from the dTree must use the exact leaf solid angle `Ω_leaf`, not a
uniform approximation `4π/N`.

The sampling PDF for a direction `ωi` drawn from the dTree is:

```
p_guide(ωi) = leaf.flux / (totalCellFlux · Ω_leaf)
```

where `Ω_leaf` is the solid angle of the leaf's octahedral patch computed from its
bounding box in [0,1]² space.

**What the deleted implementation did:**
Used `ppgGuideSolidAngle = 4π/16` for all 16 fixed bins — a uniform-sphere approximation.
At 16 bins the per-bin solid-angle variation is ~15% from mean (Cigolle et al. 2014
§2); using the uniform approximation introduces systematic bias in the guide PDF.

**Implementation requirements:**

- Precompute per-leaf solid angle at tree build time. For a leaf at octahedral UV patch
  `[u0, u1] × [v0, v1]`, the solid angle is the spherical surface area of the
  corresponding spherical polygon — computed analytically or by numerical integration
  during build.
- Store per-leaf `solidAngle` in the dTree node buffer.
- Read `solidAngle` per leaf during sampling PDF evaluation.

---

## GPU implementation notes

All five requirements above interact with the GPU render loop. The paper describes
a full-frame training pass (not alternating-frame), which in the walkaround context
translates to:

1. **Guide pass** (each frame): for each path bounce, look up the dTree for the current
   sTree cell and sample a direction using `p_mixed`. This replaces the BSDF-only sample
   for indirect bounces.

2. **Train pass** (each frame, after shading): for each path bounce where a valid
   `L_i` estimate was computed, deposit the weighted flux into the appropriate dTree leaf
   via atomic add. The deposit key is `(x, ωi)` — the position maps to an sTree cell,
   the direction `ωi` maps to a dTree leaf.

3. **Rebuild pass** (every N frames, or once per iteration): read the accumulated flux,
   compute split/merge decisions, emit the new tree topology, reset accumulators.
   This is a CPU-side operation (tree topology changes are complex to implement on GPU);
   the resulting tree is serialized to a flat GPU buffer.

Atomic GPU buffer updates (Requirement 1 and 2) require care in WebGPU:

- Use `atomicAdd` on `array<atomic<u32>>` with integer-encoded flux (fixed-point f32).
- The maximum number of dTree nodes must be bounded at initialization (pre-allocated
  flat buffer, no dynamic allocation on GPU).

---

## Suggested implementation order

1. CPU-side `buildSTree` + `buildDTree` with static test geometry — get the data
   structures right before touching the GPU.
2. GPU flux accumulation (atomic buffer, one dispatch per bounce, CPU readback to
   verify correctness on a known scene).
3. GPU-side guide sampling from a frozen (static) dTree — verify MIS weight calculation.
4. Adaptive split/merge loop (CPU-side, reading the GPU flux buffer each N frames).
5. End-to-end convergence test: white Cornell box, guide should converge to cosine
   distribution within 200 training iterations.

---

## Pre-requisites before scheduling

1. Walkaround shade loop must expose a per-bounce hook where the guide direction can be
   injected. The deleted `@@PPG_BOUNCE_INSERT@@` marker pattern was the right approach —
   revive it when scheduling this sprint.
2. The SVGF/atrous-variance depth fix (D2) must be landed so the G-buffer at the guide
   sample point is reliable.
3. ReSTIR-DI p̂ fix (Item 5) should land first: PPG and ReSTIR both touch the indirect
   sampling PDF and must be consistent.
