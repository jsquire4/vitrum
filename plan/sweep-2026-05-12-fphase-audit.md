# T1.F — Sub-agent claim re-verification

**Date:** 2026-05-11 | **Branch:** feat/sweep-2026-05-12-followup | **Auditor:** independent read of source files

---

## F1: `bsdfAreaLightConnectionContribution` rewrite

🟡 status: MOSTLY CLEAN — PDF cancellation claim was imprecise; code is correct but different from what was claimed

**File:lines verified:** `packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts:454–513`

**Evidence:**

- **Both loops present:** `for (var li = 0u; li < params.rectAreaLightCount; ...)` (line 483) AND `for (var mi = 0u; mi < params.meshAreaLightCount; ...)` (line 495). ✅
- **Closest-hit selection:** `if (... && rectDist < bestDist)` (line 488), `if (... && meshDist < bestDist)` (line 500). Correct minimum-t selection. ✅
- **PDF handling:** The plan claimed "PDF multiplied by `(rectAreaLightCount + meshAreaLightCount)` to cancel uniform light-selection." The actual code uses `powerHeuristic(bsdfPdf, bestLightPdf)` at line 511 — no count multiplication. This is **correct** for the "closest hit along BSDF direction" design: a BSDF-sampled direction hits at most one light (the nearest one), so there is no uniform-selection probability to cancel. The comment at line 476 documents this design explicitly. The prior agent's description of the mechanism was wrong, but the code itself is correct.
- **TIR/refraction unchanged:** `frDielectric` (line 303) and refraction branches (lines 1139, 1312) are entirely outside `bsdfAreaLightConnectionContribution`. The function only calls `brdfDirectionalPdf` and `evaluateBrdf`, neither of which were changed.

**Defects:** None in the code. The prior agent claim was imprecise (wrong explanation of PDF cancellation mechanism), but the implementation is correct.

---

## F2: Binned SAH builder

🟢 status: CLEAN — all 4 claims verified

**File:lines verified:** `packages/pt-webgpu/src/scene/buildCpuBvh.ts:1–423`

**Evidence:**

- **SAH cost formula with both SA terms:** `cost = (leftSA * leftCount + rightSA * rightCount) / parentSA` (line 309–311). Both `leftSA` and `rightSA` are computed from the prefix/suffix sweep arrays (lines 298–305). ✅
- **Leaf-cost early-exit:** `if (bestCost >= leafCost || bestCost === Infinity)` (line 324) makes a leaf instead of splitting. `leafCost = subset.length` (line 218). ✅
- **All 3 axes:** `for (let axis = 0; axis < 3; axis++)` (line 224) iterates X, Y, Z. Degenerate axis (`span <= 1e-9`) is skipped via `continue` (line 226). ✅
- **Stride-4 index packing:** `reorderedIndices[newTri * 4 + 3] = 0` (line 398) — `.w` is always zero-filled. ✅
- **Relative-offset encoding:** `node.rightChildOrTriOffset = rightChild - nodeIndex` (line 363). A dev-mode invariant check at lines 373–388 verifies the offset is in `[1, totalNodes)`. ✅

**Defects:** None.

---

## F3: Halton axis-angle conversion

🟢 status: CLEAN — Shoemake form, unit-quaternion conversion, Rodrigues consumer all verified

**File:lines verified:** `packages/walkaround-hybrid/src/ddgi/probeUpdatePass.ts:670–719`; `packages/shared-samplers/src/wgsl/hammersley.wgsl.ts:31–47`

**Evidence:**

- **Shoemake quaternion form:** `sigma1 = sqrt(1-u1)`, `sigma2 = sqrt(u1)`, `theta1 = 2π·u2`, `theta2 = 2π·u3`; `qw = sigma2·cos(theta2)`, `qx = sigma1·sin(theta1)`, `qy = sigma1·cos(theta1)`, `qz = sigma2·sin(theta2)` (lines 686–693). Matches Shoemake 1992 exactly. ✅
- **`sin(θ/2) = sqrt(1 - qw²)`:** `sinHalf = Math.sqrt(Math.max(0, 1 - qw * qw))` (line 697). Since `qw²+qx²+qy²+qz² = 1` by construction, this equals `sqrt(qx²+qy²+qz²) = ||(qx,qy,qz)||`. ✅
- **Rodrigues packed vec3:** `data[0..2] = [ax*angle, ay*angle, az*angle]` (lines 708–710). ✅
- **WGSL consumer convention:** `rotateAngleAxis(v, angleAxis)` in `hammersley.wgsl.ts:32–38` extracts `angle = length(angleAxis)`, `axis = angleAxis/angle`, then applies full Rodrigues formula. Convention matches. ✅

**Defects:** None.

**New test file:** `packages/walkaround-hybrid/__tests__/ddgiHaltonRotation.test.ts` (added — see below).

---

## F4: RC merge solid-angle weighting

🟡 status: MOSTLY CLEAN — formula verified correct; minor precision discrepancy between WGSL and TS implementations documented (intentional approximation)

**File:lines verified:** `packages/walkaround-hybrid/src/rc/wgsl/cascadeMerge.wgsl.ts:65–76, 190–208`; `packages/walkaround-hybrid/src/rc/octahedralSolidAngles.ts:113–148`

**Evidence:**

- **Merge formula is weighted average:** `merged = Σ child·Ω / Σ Ω` via `merged += childRad * childOmega; omegaTotal += childOmega;` then `merged /= max(omegaTotal, 1e-6)` (lines 202–206). ✅
- **WGSL `octCellSolidAngle(cx, cy, N)` vs TS `computeOctahedralSolidAngles(N)[cy*N+cx]`:** Both decode the same 4 corners (u0, v0), (u1, v0), (u0, v1), (u1, v1) and compute `sphericalQuadArea`. Index convention matches: WGSL `cx`=column, `cy`=row → TS `out[row*N+col]`. ✅
- **Precision difference (intentional, quantified):** The TS function sub-divides each cell into SUB=16 sub-cells (< 0.05% per-bin error); the WGSL computes only 1 quad (4 corners, no subdivision). Measured divergence: N=4 sum error ~13%, N=8 ~4%, N=16 ~2%. Per-cell max error up to ~6% (N=16). The comment in `cascadeMerge.wgsl.ts` lines 33–41 documents this is an intentional "two-triangle approximation" citing Cigolle 2014. The test verifies both implementations are in the same ballpark and documents the approximation gap, but does NOT require exact equality.

**Defects:** None — the approximation is documented and intentional.

**New test file:** `packages/walkaround-hybrid/__tests__/cascadeMergeWeights.test.ts` (added — see below).

---

## F5: Albedo demodulation pipeline plumbing

🟢 status: CLEAN — all 10 file claims verified

**Files verified:**

- **`shade.wgsl.ts`:** `Lo_indirect` at line 397 explicitly omits albedo with comment "Item 24: omit albedo here". `hdrAlbedoOut` written at line 481 with `albedo`. `hdrTotalOut` at line 483 re-applies `albedo` (`clampedIndirect * albedo`). ✅
- **`indirectCombine.wgsl.ts`:** `indirect = filteredLighting * albedo; output = direct + indirect` (lines 51–52). ✅
- **`resourceManager.ts`:** `albedoTexture` allocated at lines 561–565, declared in interface at line 143, destroyed at line 652. ✅
- **`bindGroupLayouts.ts`:** Binding 14 in shade layout (line 77), binding 4 in indirectCombine layout (lines 387–388). ✅
- **`bindGroupBuilders.ts`:** Shade bind group includes `{ binding: 14, resource: r.albedoTexture.createView() }` (line 86); indirectCombine bind group includes `{ binding: 4, resource: albedoView }` (line 519). ✅
- **`WalkaroundGPUPipeline.ts`:** Shade dispatched at line 750 before atrous chain starting at line 803, before indirectCombine. Dispatch order unchanged. ✅
- **`atrousVarianceWebGPU.ts`:** File does not exist as a standalone; the atrous-variance pipeline is integrated directly into `WalkaroundGPUPipeline.ts` methods `_dispatchAtrousVariance`. The `albedoView` is passed to `buildIndirectCombineBG` at line 519. No separate `atrousVarianceWebGPU.ts` file — the plan's F5 description referenced a file that was folded inline. The functionality is present and correct; only the file name in the plan doc was wrong.

**Defects:** None in the code. Plan doc F5 referenced `atrousVarianceWebGPU.ts` as a standalone file — it doesn't exist; the atrous-variance logic lives in `WalkaroundGPUPipeline.ts`. This is a stale plan-doc reference, not a code defect.

---

## Summary

| Check | Verdict | Notes |
|-------|---------|-------|
| F1 — bsdfAreaLightConnectionContribution | 🟡 | Code correct; prior agent's PDF-cancellation explanation was wrong (MIS-based, not count-multiplied). Not a defect. |
| F2 — Binned SAH builder | 🟢 | All 4 structural claims verified exactly. |
| F3 — Halton axis-angle conversion | 🟢 | Shoemake form, unit quaternion, Rodrigues consumer all verified. |
| F4 — RC merge solid-angle weighting | 🟡 | Merge formula correct; WGSL uses 1-quad approx vs TS's SUB=16 (intentional, documented). |
| F5 — Albedo demodulation plumbing | 🟢 | All bindings allocated, bound, and dispatched correctly. `atrousVarianceWebGPU.ts` name in plan was stale. |

**No code defects found.** Two plan-doc imprecisions: F1 PDF mechanism description wrong (but code correct), F5 filename reference stale (but code correct).
