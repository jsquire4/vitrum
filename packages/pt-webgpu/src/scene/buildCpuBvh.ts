/**
 * Thin re-export of the canonical CPU BVH builder.
 *
 * The Wald-2007 binned-SAH implementation (~423 LOC) was hoisted into
 * `@vitrum/shared-bvh` (W2-C2 — see `plan/premium-grade-refactor-20260517.md`)
 * so that pt-webgpu's hand-rolled builder and shared-bvh's THREE-coupled
 * `buildSceneBVH` no longer carry two structurally-identical copies of the
 * same algorithm. The canonical `buildArrayBvh` lives in
 * `packages/shared-bvh/src/buildArrayBvh.ts`.
 *
 * pt-webgpu calls the builder with the historical (stride 4 positions,
 * stride 4 indices, maxLeafTriangles=4, binCount=16) configuration —
 * exactly the defaults of `buildArrayBvh`, so the re-export below is
 * byte-identical to the pre-hoist behaviour. Existing call sites that
 * import `buildCpuBvh` from this path continue to work unchanged.
 */

export { buildArrayBvh as buildCpuBvh } from '@vitrum/shared-bvh';
