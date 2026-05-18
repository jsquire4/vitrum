# Sprint 3 PT fork patch — Sampling theory upgrade

> **Status**: Documented, not yet applied.
> Apply when resuming active fork work after Sprint 3 vitrum-side work is complete.
> Fork branch: `phase4-normalmap-shadow-rays`
> Fork path: `~/projects/three-gpu-pathtracer/`

---

## Overview

Sprint 3 has three deliverables. The vitrum-side scope (light tree CDF CPU build +
mixture PDF MIS heuristics) is complete in `@vitrum/shared-samplers`. This document
covers the three fork-side shader changes.

**Dependencies**: Sprint 2 `light.power` field (see `plan/sprint-2-pt-fork-patch.md`)
must be applied first — the light tree binary search in GLSL reads `light.power`
for leaf-node power values.

**Variance benchmark target**: render the reference interior scene at 192 samples,
measure floor-pixel standard deviation. Target: ≥3× reduction vs. baseline.
Captured post-sprint in `plan/sprint-3-benchmark.md`.

---

## Deliverable 1 — Mixture PDF (BSDF + env + light)

### File: `src/shader/direct_lighting.glsl.js`

**Current behavior**: Direct lighting draws either from the environment map or
from the light list via a binary 50/50 coin flip (or weighted by rough
environment-vs-light heuristic). Each sample uses a single strategy's PDF
as the MIS denominator.

**Change description**:

Replace the binary env-vs-light branch with a three-way mixture PDF:
`p_mixture = pBSDF * pdf_BSDF + pEnv * pdf_env + pLight * pdf_light`

1. Define three selection probabilities that sum to 1.0:
   - `pBSDF` — derived from material roughness and specular lobe weight
     (e.g., `clamp(specularWeight, 0.1, 0.8)`). Smooth/mirror surfaces get
     higher `pBSDF`; rough/diffuse get lower.
   - `pEnv` — proportional to env map intensity (existing `envMapIntensity` uniform).
     `pEnv = (1 - pBSDF) * envWeight` where `envWeight = envMapIntensity /
(envMapIntensity + totalLightPower)`.
   - `pLight = 1 - pBSDF - pEnv`.

2. Sample one strategy uniformly (draw a random in [0,1), select by CDF of
   [pBSDF, pBSDF+pEnv, 1.0]).

3. Evaluate all three PDFs at the chosen direction regardless of which strategy
   generated it.

4. Apply the power heuristic (β=2) as the MIS weight:
   `weight = pdf_chosen^2 / (pBSDF*pdf_BSDF^2 + pEnv*pdf_env^2 + pLight*pdf_light^2)`
   This matches the standard Veach power heuristic formulation for one-sample MIS
   (Pharr, Jakob & Humphreys PBR4 §13.10).

5. Contribution = `Le * bsdfEval * weight / pdf_chosen`.

**New uniform needed**: none beyond what Sprint 2 adds. `totalLightPower` is read
from a uniform already populated by the existing lights texture fill.

**DoD verification** (GPU required):

- Shader inspection: `direct_lighting.glsl.js` no longer contains a single
  `if (rand < 0.5)` branch for env-vs-light selection.
- Render the reference scene at 32 samples: verify no fireflies introduced
  by incorrect PDF division (stddev of bright pixels should not increase).
- With `pBSDF=0` and `pEnv=1` (env-only mode), output must match baseline
  env-only reference render pixel-for-pixel (verifies env path is untouched).

---

## Deliverable 2 — Light tree CDF + GPU binary search

### File 1: `src/shader/utils/light_tree.glsl.js` (new file)

**Change description** (new shader chunk):

Implement a GPU binary-search traversal over the CPU-built light tree uploaded
by `@vitrum/shared-samplers`'s `packLightTreeForGPU`. The node array is uploaded
as a `sampler2D` (RGBA32F, width=nodeCount, height=3) per the layout in
`lightTree.ts` (see `packLightTreeForGPU` doc comment for exact texel layout):

```
texel row 0: [emitterIndex, totalPower, leftChild, rightChild]
texel row 1: [aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x]
texel row 2: [aabbMax.y, aabbMax.z, pad, pad]
```

GLSL traversal function signature:

```glsl
// Descend the light tree from the root, returning a leaf emitter index.
// xi        — uniform random in [0, 1)
// shadePos  — world-space position of the shading point (for proximity)
// Returns the emitter index into the lights array.
int sampleLightTree(float xi, vec3 shadePos);
```

Algorithm:

1. Start at node 0 (root).
2. Read `[emitterIndex, totalPower, leftChild, rightChild]` from row 0.
3. If `leftChild == -1`, this is a leaf → return `int(emitterIndex)`.
4. Read left child's `totalPower`. Let `pLeft = leftPower / currentNode.totalPower`.
5. Apply optional spatial proximity correction: scale `pLeft` by
   `exp(-distToLeftAabb / distToRightAabb)` where dist is the minimum
   L2 distance from `shadePos` to each child's AABB. This biases selection
   toward closer emitters (Estevez & Kulla 2018 §3). Clamp to [0.05, 0.95]
   to prevent degenerate selection.
6. If `xi < pLeft`, descend left (rescale `xi /= pLeft`); else descend right
   (rescale `xi = (xi - pLeft) / (1 - pLeft)`).
7. Repeat from step 2 with the chosen child.
8. The resulting leaf PDF is the product of selection probabilities along
   the path from root to leaf.

**Texture uniform name**: `lightTreeSampler` (sampler2D, RGBA32F).
**Node count uniform**: `lightTreeNodeCount` (int).

### File 2: `src/PathTracingSceneGenerator.js` (JS uploader)

**Change description**:

After scene upload (where the lights texture is currently built), additionally:

1. Extract `cellPower` per light from the light list (already available as
   `light.power` from Sprint 2).
2. Compute centroids and AABBs from each light's `position`, `u`, `v`, `area`.
3. Call `buildLightTree({ powers, centroids, aabbs })` from `@vitrum/shared-samplers`.
4. Call `packLightTreeForGPU(nodes)` to get the `Float32Array`.
5. Upload as a `DataTexture` with format `RGBAFormat`, type `FloatType`,
   width = nodeCount, height = 3.
6. Assign to `PhysicalPathTracingMaterial`'s `lightTreeSampler` uniform.

**Rebuild trigger**: wire to the existing scene-dirty signal that rebuilds the
lights texture. The `buildLightTree` call is <1ms for typical light counts
(<100 emitters), well within the 50ms debounce in `PathtracerSceneSync`.

### File 3: `src/shader/direct_lighting.glsl.js` (continued from Deliverable 1)

After selecting `strategy == light`, replace the current uniform light selection
(`light = lights[int(xi * lightCount)]`) with a call to `sampleLightTree(xi, worldPos)`.
The returned index feeds the existing light evaluation path unchanged.

The PDF for the light sample is now the product of tree-descent selection
probabilities × (1 / light.area) for the surface-area PDF at the leaf.

**DoD verification** (GPU required):

- Open `PhysicalPathTracingMaterial` uniforms inspector: `lightTreeSampler` exists
  and has non-zero node count.
- Render at 192 samples: measure floor-pixel stddev. Target ≥3× reduction vs.
  baseline uniform sampling (baseline measured before Sprint 3 patch applied).
- Equal-power scene (all lights identical power): tree sampling degenerates to
  uniform sampling; output should be visually indistinguishable from pre-sprint
  baseline (regression test).
- High-contrast power scene (one 1000× brighter light): bright light gets
  selected far more often; floor directly below bright source should converge
  faster. Verify via per-pixel variance map.

---

## Deliverable 3 — Back-face NEE resample

### File: `src/shader/direct_lighting.glsl.js`

**Current behavior**: when a sampled light direction hits the back face of a
transmissive panel (dot(lightDir, panelNormal) < 0), the sample is discarded
with zero contribution. This wastes the sample budget.

**Change description**:

Wrap the back-face check in a resample loop (up to 4 attempts):

```glsl
for (int attempt = 0; attempt < 4; attempt++) {
    // Generate new light sample (or resample env/BSDF depending on strategy)
    vec3 lightDir = sampleChosenStrategy(rand4());

    float cosTheta = dot(lightDir, shadingNormal);
    if (cosTheta > 0.0) {
        // Front-face hit — proceed with MIS evaluation
        contribution = evaluateDirect(lightDir, ...);
        break;
    }
    // Back-face — try again (next loop iteration)
    // After 4 attempts, contribution stays zero
}
```

The PDF used for MIS weighting must account for the resampling: since we draw
until front-face hit (or exhaust attempts), the effective PDF is the original
strategy PDF × (1 / frontFaceProbability). For typical scenes, `frontFaceProbability`
≈ 0.5 for hemispherical distributions; use the exact cosTheta distribution if
the strategy is cosine-weighted.

Simpler conservative approach (acceptable for Sprint 3): after 4 attempts, if
no front-face sample found, contribute zero. Use the original un-resampled PDF
as the MIS denominator — this slightly biases the estimate but avoids the costly
PDF correction and remains unbiased in expectation over many paths.

**DoD verification** (GPU required):

- Render a transmissive panel scene with a light source behind the panel.
  Before: pixels on the transmissive side show back-face fireflies or dark
  splotches. After: back-face artifacts are absent or greatly reduced.
- Shader inspection: the NEE shadow ray section contains a loop with
  `attempt < 4` guard rather than a single unconditional sample.
- No energy loss on opaque scenes (no transmissive panels): comparison render
  must differ by < 0.1% mean pixel error from pre-sprint baseline.

---

## Variance benchmark spec (Sprint 3 DoD)

Per `plan/phase-6-roadmap.md` §4, Sprint 3 DoD requires a variance benchmark
captured in `plan/sprint-3-benchmark.md`. The benchmark procedure:

**Reference scene**: interior room with 4 stained-glass panels, directional sun,
HDR sky. Camera looking toward floor receiving colored light patches.

**Measurement procedure**:

1. Render at exactly 192 samples with PT_FINAL settings (1.0× DPR, full bounces).
2. Capture the raw Float32 accumulation buffer before tone mapping.
3. For a 16×16 region of floor pixels directly under the brightest panel:
   - Compute per-pixel luminance.
   - Compute standard deviation of luminance across the 256 pixels.
4. Record: `stddev_before` (pre-sprint baseline) and `stddev_after` (post-sprint).
5. Target: `stddev_before / stddev_after ≥ 3`.

**Baseline capture**: must be taken with the same scene and settings BEFORE applying
any Sprint 3 shader patch. Recommended: take baseline on `phase4-normalmap-shadow-rays`
branch tip before creating a sprint-3 branch.

**Format**: see `plan/phase-6-roadmap.md` §9 for the benchmark file template.
File path: `plan/sprint-3-benchmark.md`.
