# Phase 6 Sprints 1–11 Audit — 2026-05-09

> **REMEDIATED** — All findings addressed in commit `6a0da62` (fix: remediate remaining
> audit findings — M-1, M-2, M-4, M-5, M-6, L-2) and commit `2cf887a` (HIGH findings H-1,
> H-2, M-3, L-1, L-3 fixed in the same session). Tests grew from 346 → 542 (22 test files).
> This document is preserved as a historical record of pre-remediation findings.

## Summary
- Findings: HIGH=2, MEDIUM=6, LOW=3
- Tests at audit time: 346 passing (193 walkaround + 69 shared-denoisers + 54 shared-samplers + 18 pt-webgl + 11 shared-bvh + 1 three-bindings)
- tsc: clean (exit 0, no diagnostics)

---

## HIGH severity

### H-1 — PPG leaf stride mismatch between ppgUpdate.wgsl and ppgSample.wgsl
**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts:163-165`
**Also:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgSample.wgsl.ts:55-61`

`ppgUpdate.wgsl` reads the leaf buffer as `array<atomic<u32>>` and computes leaf offsets via:
```wgsl
fn ppgLeafSlot(leafIdx: u32, binIdx: u32, field: u32) -> u32 {
  return leafIdx * 32u + binIdx * 2u + field;
}
```
`leafIdx * 32u` means 32 u32 slots = 128 bytes per leaf.

`ppgSample.wgsl` binds the same buffer as `array<PPGDirectionalLeaf>`, where:
```wgsl
struct PPGDirectionalLeaf {
  bins:      array<vec2f, 16>,  // 128 bytes
  _reserved: array<vec2f, 16>,  // 128 bytes reserved
};
```
Each struct element is 256 bytes (`PPG_LEAF_BYTE_STRIDE` in `types.ts:129`).

For `leafIdx >= 1`, the two shaders address different memory locations in the same buffer: ppgSample expects leaf 1 to start at byte 256, ppgUpdate writes to byte 128. When these shaders are integrated, any scene with more than one occupied PPG spatial cell will silently corrupt the directional bin data for all cells beyond cell 0.

**Suggested fix:** Change `ppgUpdate.wgsl` to use `leafIdx * 64u` (256 bytes / 4 bytes per u32 = 64 u32 slots per leaf). Update the `ppgLeafSlot` formula accordingly and add a comment linking to `PPG_LEAF_BYTE_STRIDE`.

**Remediation:** Fixable now. Both shaders are string-only (not dispatched), so this is a pre-integration correction with no runtime risk today. Must be caught before Sprint 11 integration wires the dispatch.

---

### H-2 — PPG fixed-point radiance clamp saturates at 256 nits, not 65536
**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgUpdate.wgsl.ts:195`

```wgsl
let lumFixed = u32(clamp(lum * PPG_RADIANCE_SCALE, 0.0, f32(0xFFFFFFu)));
```

`0xFFFFFFu` = 16,777,215. With `PPG_RADIANCE_SCALE = 65536`, the maximum representable luminance before saturation is `16777215 / 65536 ≈ 256 nits`. The accompanying comment says: _"Max representable radiance: 2^32 / 65536 = 65536"_ — which would require clamping to `0xFFFFFFFFu` (4,294,967,295).

For HDR scenes with emitters above 256 nits (common: sun-through-glass can reach 1000+ nits), all samples exceeding the threshold contribute the same saturated fixed-point value, destroying directional discrimination in bright cells.

**Suggested fix:** Change `f32(0xFFFFFFu)` to `f32(0xFFFFFFFFu)`. Update the comment to reflect the correct maximum.

**Remediation:** Fixable now. Shader is string-only (not dispatched) — zero integration risk today.

---

## MEDIUM severity

### M-1 — Light tree CDF is not a real CDF: exceeds 1.0 and cannot drive sampling
**File:** `packages/shared-samplers/src/lightTree.ts:238-248`

The `buildLightTree` function builds a CDF over all nodes in pre-order traversal, using each node's `totalPower`. Because internal nodes aggregate subtree power, their power is counted once in the running sum before each leaf's power is counted again. For any tree with >1 leaf, `cdf[0]` = `root.totalPower / root.totalPower` = 1.0, then subsequent entries exceed 1.0.

The docstring at line 199 claims _"CDF for stochastic root-to-leaf descent"_ — this is misleading. A value that starts at 1.0 and increases cannot drive uniform sampling. The test at `__tests__/lightTree.test.ts:74` acknowledges the issue in a comment (_"values can exceed 1.0 before normalization"_) but does not assert the final value is exactly 1.0, making the "monotonic" claim in the test description a partial truth.

The GPU never consumes this CDF (it does its own binary traversal), so there is no correctness impact on rendering today. The impact is on any future CPU-side caller who reads the docstring and treats this as a standard sampling CDF.

**Suggested fix:** Either (a) rename to `subtreePowerPrefix` and document it accurately, or (b) build the CDF only over leaf nodes (filter to `emitterIndex >= 0` before summing) so it is a real CDF from 0 to 1.

**Remediation:** Requires judgment — the GPU traversal is unaffected, but the public API name/doc is misleading. Fixable now if desired.

---

### M-2 — SVGF sigmaColor=10 claimed from Schied 2017 Table 1, but Table 1 shows σ_l = 4
**File:** `packages/shared-denoisers/src/svgfBindings.ts:90-91`

`SVGF_DEFAULT_UNIFORMS.sigmaColor = 10.0` with the comment "Default σ values follow Schied 2017 Table 1." Schied 2017 Table 1 specifies σ_l (luminance color σ) = 4.0, not 10.0. The formulation differs slightly (this code uses `sigmaColor²` as a divisor inside `variance * sigmaColor²`; the paper uses `σ_l²`), so 10 may be an intentional re-tuning for a different normalization convention. If so, the claim of provenance from Table 1 is incorrect and should be clarified.

**Suggested fix:** Change the comment to note the value is tuned for this engine's edge-stop formulation and is not directly from the paper, or document the conversion factor.

**Remediation:** Documentation fix — requires judgment on whether the value is intentionally different or an unchecked carry-over.

---

### M-3 — HWC↔NCHW round-trip helpers are not directly tested
**File:** `packages/shared-denoisers/src/oidnBridge.ts:287-330`

`_hwcToNchw` and `_nchwToHwc` are internal helpers with no direct unit tests. The test file claims in its header that they are "correct (tested via the public denoiseFinal return type guarantee)" — but the test only confirms `denoiseFinal` returns a Promise and throws when ORT is absent; it never exercises the layout transform at all.

A layout transpose bug in either helper would silently produce scrambled output that looks plausible (same pixel count, different channel-to-position assignment).

**Suggested fix:** Export `_hwcToNchw` and `_nchwToHwc` with an underscore-prefixed name (or create a `@internal` export path), then add a direct round-trip test: build a small 2×2 HWC buffer, convert to NCHW, convert back, and assert identity.

**Remediation:** Fixable now. Low-effort.

---

### M-4 — PPG kd-tree brute-force O(N) scan at 10K cells is a per-sample cost in the shade pass
**File:** `packages/walkaround-hybrid/src/ppg/wgsl/ppgSample.wgsl.ts:112-131`

The `ppgFindCellIndex` function runs a linear scan over all `cellCount` cells per shading invocation. At `PPG_MAX_SPATIAL_CELLS = 10_000`, this is 10K distance comparisons per indirect-bounce path vertex per pixel per frame. The commit message acknowledges this ("acceptable during the structural-prep phase") and tags a follow-up, but no tracking issue exists in the plan docs.

At typical walkaround resolution (1920×1080, checkerboard = 50% pixels, 2 bounces), this is approximately 10K × 1080 × 960 = 10.4 billion comparisons per frame — not acceptable even in deferred state if a future sprint wires dispatch without first addressing the scan.

**Suggested fix:** Add a tracking entry to `plan/sprint-11-ppg-integration.md` explicitly blocking dispatch wiring on kd-tree index replacement. The current code is fine for structural prep; the risk is that it gets wired without the optimization.

**Remediation:** Requires user judgment on sprint ordering and blocking conditions.

---

### M-5 — mixturePdf divide-by-zero not protected when all probabilities are 0
**File:** `packages/shared-samplers/src/mixturePdf.ts:95-99`

`mixturePdf` returns `result = Σ probabilities[i] × pdfs[i]`. If all probabilities are 0, result = 0 — this is not a throw and is not protected. However, the function does not divide. The issue is a silent 0 return where a 0-probability mix is logically undefined (no strategy active). A caller using this as a PDF denominator in MIS weighting will then divide by 0.

The docstring does not state the behavior for all-zero probabilities. `balanceHeuristic` and `powerHeuristic` protect against their zero cases (returning 0.5); `mixturePdf` does not have an equivalent note.

**Suggested fix:** Add a `@throws` or `@returns` note documenting the all-zero case, or optionally throw if `result === 0` when all probabilities are non-negative. No callers today so this is pre-emptive.

**Remediation:** Documentation/minor guard — fixable now.

---

### M-6 — SVGF atrous iteration count is fixed at 5 in shader; not overridable via UBO
**File:** `packages/shared-denoisers/src/wgsl/svgf.wgsl.ts:19-20` (docstring)

The `svgfAtrousMain` shader performs exactly one iteration per dispatch call. The host is supposed to dispatch it 5 times. The `iteration` field in `SVGFAtrousUBO` controls the step width for that one call (0–4). There is no way to run fewer than 5 iterations without changing the host dispatch logic.

The iteration count is not exposed as a uniform, meaning a host that wants 3 iterations (for performance on mobile) must rebuild pipeline bindings. This is a MEDIUM concern because iteration-count tuning is a legitimate quality-vs-speed lever in the Schied paper.

**Suggested fix:** Add `maxIterations: u32` to `SVGFAtrousUBO` and early-exit the shader if `atrousUBO.iteration >= atrousUBO.maxIterations`. Alternatively, document explicitly that iteration count is a host-side dispatch parameter (not a shader parameter), which is also accurate.

**Remediation:** Requires user judgment — current design is valid if the host controls dispatch count.

---

## LOW severity

### L-1 — jakobHanika.ts TODO not tracked in any plan document
**File:** `packages/shared-samplers/src/jakobHanika.ts:49`

The `TODO (Sprint 12 or earlier)` comment references integrating the full precomputed table but no corresponding entry exists in `plan/phase-6-roadmap.md` or the sprint docs. Risk: it gets forgotten.

**Suggested fix:** Add a Sprint 12 entry to `plan/phase-6-roadmap.md` noting the license investigation and table integration path.

---

### L-2 — Light tree median split degrades for clustered centroids without warning
**File:** `packages/shared-samplers/src/lightTree.ts:148-152`

When all emitter centroids are co-located (e.g., all panel cells in a single stained-glass window are modeled at the window centroid), the split axis span is 0 and the sort produces an arbitrary but valid partition. The resulting tree has degenerate depth (no spatial separation) and the GPU proximity heuristic will not function. No warning or detection is emitted.

This is low severity because (a) real scenes have spread centroids and (b) the GPU traversal degrades gracefully to power-weighted sampling without proximity bias.

**Suggested fix:** Add a check: if `Math.max(spanX, spanY, spanZ) < 1e-4`, emit a `console.warn` noting that the centroid cloud is degenerate and the spatial heuristic will be ineffective.

---

### L-3 — BDPT exports appear in shared-samplers index.ts but were not part of audited sprints
**File:** `packages/shared-samplers/src/index.ts:7,18-31`

A lint/auto-save pass (noted in system context) added Sprint 10c BDPT exports to `index.ts`. The BDPT files (`bdptVertex.ts`, `bdptMIS.ts`) exist on disk and tsc compiles them, but they are not part of the Sprint 1–11 audit scope and were not covered by any of the 346 tests in this audit. Per `plan/phase-6-roadmap.md`, Sprint 10c is explicitly deferred ("trigger criterion: Sprint 7 caustic gap identified").

No correctness risk to existing code since exports don't break callers. Risk is that deferred code appears in the public API before integration testing.

**Suggested fix:** Move BDPT exports behind a comment block marking them as deferred/unreleased, or remove from index.ts until Sprint 10c is officially opened.

---

## Sprint-by-sprint review notes

**Sprint 1+2** — Solid. Rec. 709 luminance weights verified correct (0.2126/0.7152/0.0722) in both `bvhCompute.ts` and `sceneToThree.ts`. cellPower per-triangle calculation is correct (not per-mesh average). Formula agreement between walkaround and PT sides is confirmed. rect-area formula `4 × |uAxis × vAxis|` is correct for half-extent inputs.

**Sprint 3+4** — Light tree build logic and GPU pack layout are correct. The 12-floats/node RGBA32F layout is verified (3 texels × 4 components). The CDF is structurally present but misleadingly named (M-1). MIS heuristics are numerically stable at large PDF ratios (verified). `mixturePdf` zero-probability case undocumented (M-5).

**Sprint 5+6** — `packCameUBO` std140 layout is correct: vec3+float pairs share the vec4 boundary, webThickness at offset 8, explicit padding to 64 bytes. Truncation-with-warn behavior confirmed. `supportedAnalyticShapes` correctly returns empty Set when `MAX_FRAGMENT_UNIFORM_VECTORS < 256`. Sprint 6 hexagonal spatial filter is well-formed.

**Sprint 7** — HG phase function is correct. g=0 isotropic (1/4π), g→±1 numerical stability guarded by clamp to ±0.9999. PDF normalization integral verified by tests. `sampleHG(u1, u2, 0)` returns cosTheta = 1−2u2, which is uniform on sphere (correct — not cosine hemisphere). Equi-angular sampling present and tested.

**Sprint 8** — Placeholder is explicitly labelled with a prominent doc block. Negative RGB inputs clamp to 0 (graceful). White → flat spectrum → near-white reconstruction: the sigmoid of a flat polynomial returns a value near 0.5 not 1.0 for pure white — this is a known limitation of the Gaussian-peak approximation and is documented in the source.

**Sprint 9** — Welford n=1 returns 0 (verified in code and tests). `SampleBudgetUniforms` thresholds are in the UBO struct (host-overridable, not hardcoded). Motion-vector 1×1 sentinel detection confirmed: `mvTexW <= 1u && mvTexH <= 1u` early-returns zero vector. Ping-pong parity convention in resolve.wgsl uses `(px+py) & 1u == frameParity` — distinct from PPG's frame-level parity — both are consistent within their respective passes.

**Sprint 10a** — SVGF WelfordVariance struct matches common.wgsl byte-for-byte (both: `mean: f32, m2: f32`). Atrous iteration count is dispatch-controlled by host (MEDIUM if iteration pruning is desired, L if not — current design is defensible). sigmaColor=10 provenance claim is imprecise (M-2).

**Sprint 10b** — `new Function('id', 'return import(id)')` lazy-import compiles cleanly (tsc exit 0 with no ORT installed). Missing-ORT error message is clear and actionable. HWC↔NCHW helpers are untested directly (M-3).

**Sprint 11** — PPG types, buffer allocation, and shader strings are structurally sound. Two correctness bugs exist in the WGSL (H-1 stride mismatch, H-2 saturation bound), both pre-integration so no runtime impact today. Brute-force scan is a performance concern to track (M-4).

---

## Concerns: remediate now vs. needs user judgment

**Remediate now (safe, non-disruptive):**
- H-1: `ppgUpdate.wgsl` leaf stride formula — change `leafIdx * 32u` to `leafIdx * 64u`
- H-2: `ppgUpdate.wgsl` clamp bound — change `0xFFFFFFu` to `0xFFFFFFFFu`
- M-3: Add direct HWC↔NCHW round-trip test in `oidnBridge.test.ts`
- L-1: Add Sprint 12 tracking entry for jakobHanika table integration
- L-3: Remove or mark-deferred the BDPT exports that appeared in `index.ts`

**Requires user judgment:**
- M-1: Light tree CDF rename/rebuild — affects public API surface; GPU consumer is unaffected
- M-2: SVGF sigmaColor claim vs. paper — may be intentionally re-tuned; need authorial confirmation
- M-4: PPG brute-force O(N) scan — determine whether Sprint 11 integration is blocked on kd-tree
- M-5: mixturePdf all-zero documentation — decide whether to throw or document
- M-6: SVGF iteration count as dispatch vs. shader parameter — current design is valid if host controls dispatch count
- L-2: Clustered centroid warning in light tree — low priority quality-of-life
