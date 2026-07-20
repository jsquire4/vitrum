/**
 * giSceneBindings.wgsl.ts — the shared @group(1) scene (BVH + TLAS) binding
 * block used verbatim by all four GI-reuse bodies.
 *
 * D8-7 (complexity-sweep 2026-07-20, T4-4): the OFF/default and GRIS opt-in
 * bodies of BOTH temporalGi and spatialGi declare the SAME eight @group(1)
 * BVH/TLAS bindings byte-for-byte. The deliberate OFF-vs-GRIS two-body split
 * STAYS (accepted-intentional); only the byte-identical binding block is shared
 * here so it lives in one place.
 *
 * These are CONSUMER bindings (@group(1) resources), so per the composeWgsl
 * ordering rule they are shared as a RAW-STRING fragment interpolated into each
 * body — NOT a WgslModule. The fragment is emitted byte-for-byte identically at
 * each site so the composed temporalGi/spatialGi WGSL (both variants) stays
 * byte-identical. Companion to temporalGiCommon.wgsl.ts / spatialGiCommon.wgsl.ts.
 */

/**
 * The @group(1) scene BVH + TLAS binding declarations shared by temporalGi and
 * spatialGi (OFF + GRIS). Emitted with NO trailing newline so each call site
 * controls the surrounding whitespace exactly.
 */
export const GI_SCENE_GROUP_BINDINGS_WGSL = /* wgsl */ `@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;`;
