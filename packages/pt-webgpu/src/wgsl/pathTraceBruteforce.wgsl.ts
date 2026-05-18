import { HAMMERSLEY_WGSL, OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';
import { PT_WEBGPU_COMMON_WGSL } from './common.wgsl.js';
import { PT_WEBGPU_FRAME_PARAMS_WGSL } from './frameParams.wgsl.js';
import { PT_WEBGPU_BSDF_MIS_WGSL } from './bsdf/mis.wgsl.js';
import { PT_WEBGPU_BSDF_DIFFUSE_WGSL } from './bsdf/diffuse.wgsl.js';
import { PT_WEBGPU_BSDF_GLOSSY_WGSL } from './bsdf/glossy.wgsl.js';
import { PT_WEBGPU_BSDF_LAYERED_WGSL } from './bsdf/layered.wgsl.js';
import { PT_WEBGPU_BVH_WGSL } from './bvh.wgsl.js';
import { PT_WEBGPU_MATERIAL_DECODE_WGSL } from './materialDecode.wgsl.js';
import { PT_WEBGPU_LIGHT_ENVIRONMENT_WGSL } from './lights/environment.wgsl.js';
import { PT_WEBGPU_LIGHT_RECT_AREA_WGSL } from './lights/rectArea.wgsl.js';
import { PT_WEBGPU_LIGHT_MESH_AREA_WGSL } from './lights/meshArea.wgsl.js';
import { PT_WEBGPU_LIGHT_EMISSIVE_WGSL } from './lights/emissive.wgsl.js';
import { PT_WEBGPU_MNEE_WGSL } from './mnee.wgsl.js';
import { PT_WEBGPU_PHOTON_MAP_WGSL } from './photonMap.wgsl.js';
import { PT_WEBGPU_MAIN_WGSL } from './main.wgsl.js';

/**
 * Brute-force path tracing kernel — composed from per-concern modules
 * (W4-A4 refactor of the previous 1908-LOC monolith).
 *
 * Composition order (top-down):
 *   1. Common WGSL  — PCG RNG, BVHNode/Ray structs, triangle intersection.
 *   2. Hammersley + octahedral encoders (from @vitrum/shared-samplers).
 *   3. FrameParams UBO struct + 24 group(0) bindings + material constants.
 *   4. BSDF primitives — Fresnel/microfacet/MIS helpers (`bsdf/mis`).
 *   5. BSDF eval+PDF — Cook-Torrance unified BRDF (`bsdf/diffuse`).
 *   6. BSDF glossy — Heitz 2018 VNDF + ONB helpers (`bsdf/glossy`).
 *   7. BVH traversal — `traceClosest` / `traceAny` + analytic shapes.
 *   8. Material decode — packed-UBO accessors + DecodedMaterial.
 *   9. Environment lighting — sky / HDRI / importance sample.
 *  10. Rect-area + mesh-area light intersectors (sum-MIS direction mode).
 *  11. Emissive connections — BSDF-direction-mode MIS contributions.
 *  12. MNEE — caustic strategy mode 1.
 *  13. Photon map — caustic strategy mode 2.
 *  14. Layered BSDF direction sampler — `sampleNextBounceDirection`.
 *  15. Main kernel + per-pixel orchestration.
 *
 * Constraints maintained by this composition:
 *  - The compiled WGSL string is asserted byte-equivalent against a golden
 *    snapshot in `__tests__/pathTraceBruteforceComposition.test.ts`.
 *  - WGSL function declarations are order-independent for fn-fn calls; the
 *    chosen order satisfies struct-before-use for `BVHNode`, `Ray`,
 *    `SceneHit`, `DecodedMaterial`, `BounceSample`, and `RRResult`.
 *  - The exported `PT_WEBGPU_TRACE_WGSL` name and the consumer in
 *    `packages/pt-webgpu/src/index.ts` are unchanged.
 *
 * Dead code removed during this refactor:
 *  - `sampleMeshAreaLight` — uncalled per sweep finding F9; the main kernel's
 *    per-light loop handles mesh-area sampling inline.
 *  - The accompanying "sampleRectAreaLight was removed" comment block.
 *
 * Future work flagged in module headers:
 *  - W2-C1: source `BVH_TRAVERSE_WGSL` from `@vitrum/shared-bvh`.
 *  - Shared `BsdfSample {wi, pdf, value}` return struct for per-BSDF
 *    sample/PDF/eval split (deferred from W4-A4 — requires signature change).
 */
export const PT_WEBGPU_TRACE_WGSL = /* wgsl */ `
${PT_WEBGPU_COMMON_WGSL}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${PT_WEBGPU_FRAME_PARAMS_WGSL}
${PT_WEBGPU_BSDF_MIS_WGSL}
${PT_WEBGPU_BSDF_DIFFUSE_WGSL}
${PT_WEBGPU_BSDF_GLOSSY_WGSL}
${PT_WEBGPU_BVH_WGSL}
${PT_WEBGPU_MATERIAL_DECODE_WGSL}
${PT_WEBGPU_LIGHT_ENVIRONMENT_WGSL}
${PT_WEBGPU_LIGHT_RECT_AREA_WGSL}
${PT_WEBGPU_LIGHT_MESH_AREA_WGSL}
${PT_WEBGPU_LIGHT_EMISSIVE_WGSL}
${PT_WEBGPU_MNEE_WGSL}
${PT_WEBGPU_PHOTON_MAP_WGSL}
${PT_WEBGPU_BSDF_LAYERED_WGSL}
${PT_WEBGPU_MAIN_WGSL}
`;
