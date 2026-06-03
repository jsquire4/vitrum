# V24 engine-side path-replay adjoint pass — build spec

Recon + executable plan for the LAST piece of the GPU differentiable-RT adjoint:
implementing `InverseEngineHooks.computeAdjointGradient` in the pt-webgpu engine.
Captured 2026-06-03 after the two adjoint-math stages were GPU-validated and the
session consumer was wired. This is an **architecture/plan** artifact (no code
landed yet) so the next build iteration starts sharp instead of re-deriving it.

## What's already done (the building blocks — all proven)

| Piece | Where | Validation |
|---|---|---|
| BRDF partials `dBrdf/dθ` | `wgsl/pathTrace/pathTraceAdjoint.wgsl.ts` (`dBrdf_dBaseColor`, `dBrdf_dRoughness`) | GPU == FD oracle, f32 precision (`wsl-gpu/scripts/adjoint-validate.ts`) |
| Chain rule + grad accumulation | `inverse/adjointHarness.wgsl.ts` (`ADJOINT_SHADING_FD_WGSL`, `adjointScatter`) | analytic == on-device FD, rel 7.9e-4 (`adjoint-fd-validate.ts`) |
| Session consumer | `inverse/inverseSession.ts` (`computeAdjointGradient` hook + `method` resolution) | 5 unit tests (resolution + consumption + length guard) |

## The remaining pass — flow

Per pixel, single-bounce direct-lighting adjoint (mirrors `kernel.wgsl.ts:main`
first bounce):

1. `generatePrimaryRay(px, py, jitter=0)` — FROZEN seed (reuse `kernelCore`).
2. `traceClosest` → first hit (pos, normal, triId) — reuse `intersection.wgsl.ts`
   (`PT_WEBGPU_INTERSECTION_*`). For the Phase-1 scene assume BLAS or the TLAS
   traversal already wired; `hitMaterialId` → matId.
3. `decodeMaterial(matId)` → baseColor/roughness/metallic (reuse `material.wgsl.ts`).
4. NEE: `sampleLightTree(hitPos, …, &rng)` with the SAME frozen rng → lightDir +
   Li + lightPdf (reuse the `connect.wgsl.ts` path the forward NEE uses). Hold
   the sampled direction CONSTANT (path-replay freezes the light choice).
5. `evaluateBrdf(baseColor, roughness, metallic, normal, wo, lightDir)` → brdf;
   `rendered_contrib = brdf · NdotL · Li / lightPdf`.
6. Partials: `dBrdf_dBaseColor` / `dBrdf_dRoughness` (GPU-validated).
7. `∂loss/∂θ = dLoss_dRendered[pixel] · (dBrdf/dθ · NdotL · Li / lightPdf)`,
   diagonal for baseColor, dot-reduced over channels for roughness (exactly the
   `ADJOINT_SHADING_FD_WGSL` math, now over a re-traced hit).
8. `adjointScatter(gradSlot, g)` — but ONLY when the hit's matId matches a param's
   matId; gradSlot = that param's flat-gradient offset (+0/1/2 for baseColor rgb).

## The bind-group challenge (the bulk of the work)

The forward pipeline uses ALL 4 bind groups (`gpuResources.ts`): group0 (~12-14:
scene buffers + output textures + variance), group1 (10: analytics/env/area
lights), group2 (TLAS 5 + BDPT 2), group3 (lightTree + P2 textures). `maxBindGroups`
default = 4, so the adjoint I/O can't go in a 5th group.

**Chosen approach — a FOCUSED adjoint-only pipeline** (not a forward variant): the
adjoint needs only the single-bounce read subset, freeing slots the forward spends
on output textures / variance / BDPT eye-stack:
- group0: `params` UBO (camera VP, counts), positions, normals, indices,
  triMaterialIds, materials, bvhNodes (or TLAS set), point/spot/rect/mesh lights,
  lightTree.
- group1 (adjoint I/O): `dLossDRendered` (read-only storage, per-pixel
  `channels`-wide), `gradAccum` (atomic<i32> array, read_write), `adjointParams`
  (read-only: per-param `{matId, fieldCode, gradOffset}`), `adjointParamCount` (in
  the UBO).
Compose the kernel WGSL by reusing the forward modules + `pathTraceAdjoint` +
`composeWgsl` topo-sort; declare the bindings to match.

## Validation plan (autonomous-safe — TIGHT, not just within-5×)

1. **EXACT single-bounce A/B (the key gate):** run the engine's inverse render at
   `maxBounces = 1` (single-bounce forward), then GPU-adjoint gradient MUST equal
   the GPU single-bounce FD gradient the `step()` loop computes, to ~f32 + FD
   truncation (rel ≲ 1e-2). Because both are single-bounce, this is EXACT — a
   wrong kernel fails it. Extend `wsl-gpu/tests/v24-inverse-fit.mjs` with a
   `--method=path-replay` arm + a single-bounce config.
2. **within-5× full-render** (the documented Phase-1 acceptance): on the normal
   multi-bounce render, GPU-adjoint within ~5× of GPU-FD (single-bounce
   approximates multi-bounce — expected gap, not a bug).
3. **Dispose clean:** the adjoint pipeline/buffers are engine-owned; free them in
   the engine dispose (V24 item #2).

## Wire-up

`index.ts:createInverseSession` adds `computeAdjointGradient` to the `hooks`
object → builds/caches the adjoint pipeline lazily, uploads `dLoss_dRendered` +
the param descriptor, dispatches `ceil(w·h/64)`, reads back `gradAccum`, divides
by `ADJOINT_GRAD_FP` (2^20), returns the flat gradient. The session already
resolves `method:'path-replay'` once the hook exists + params are eligible.

## DO NOT

- Commit an uncalled adjoint kernel (dead code — the repo already had that with
  `pathTraceAdjoint.wgsl` and it cost a sweep to notice). Land kernel + dispatch +
  hook + the EXACT A/B together.
- Differentiate through the light/BSDF SAMPLING — path-replay freezes it; only the
  continuous shading is differentiated (sidesteps visibility discontinuities).
