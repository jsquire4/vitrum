// @vitrum/shared-samplers — sampling utilities for path tracers and walkaround engines.
//
// Phase 1 deliverable: Hammersley QMC sequence + sphere sampling WGSL fragment.
// Sprint 3 (Phase 6): light tree CDF (CPU-side build + GPU pack), mixture PDF / MIS heuristics.
// Sprint 7 (Phase 6): HG phase function + equi-angular volume distance sampling.
// Sprint 8b (Phase 6): Jakob+Hanika spectral upsampling.
// Sprint 10c (Phase 6): BDPT vertex layout + connection-PMF MIS weights.
// Sprint 12 (Phase 6): CIE CMF tables, hero-wavelength sampling, Cauchy IOR.

export * from './wgsl/hammersley.wgsl.js';
export { OCTAHEDRAL_CORE_WGSL } from './wgsl/octahedralCore.wgsl.js';
export { LUMINANCE_WGSL, LUMINANCE_MODULE_NAME } from './wgsl/luminance.wgsl.js';
export { HERO_WAVELENGTH_WGSL, HERO_WAVELENGTH_MODULE_NAME } from './wgsl/heroWavelength.wgsl.js';
export {
  acesFilmic,
  reinhard,
  agx,
  applyTonemap,
  linearToSrgb,
  srgbToLinear,
  TONEMAP_MODE_INDEX,
  type TonemapMode,
} from './tonemap.js';
export { tonemapWgsl } from './wgsl/tonemap.wgsl.js';
export { PCG_WGSL, PCG_MODULE_NAME, PCG_HASH_TO_F32_WGSL } from './wgsl/pcg.wgsl.js';
export { BSDF_PRIMITIVES_WGSL, BSDF_PRIMITIVES_MODULE_NAME } from './wgsl/bsdfPrimitives.wgsl.js';
export { luminance, luminanceAt } from './luminance.js';
export { haltonSO3AxisAngleFromFrameIndex } from './haltonSo3.js';
export { bakePreethamSkyEquirect } from './preethamSky.js';
export type { PreethamSkyBake, PreethamSkyBakeOptions } from './preethamSky.js';
export {
  buildLightTree,
  packLightTreeForGPU,
  LIGHT_TREE_FLOATS_PER_NODE,
  /** Canonical full-sphere orientation cone constant (axis=[0,0,0], thetaO=π,
   *  thetaE=π). Use for unoriented emitters; cone importance term ≡ 1. */
  FULL_SPHERE_CONE,
  // CPU reference traversal — the byte-for-byte oracle the WGSL `sampleLightTree`
  // mirrors. Used by pt-webgpu's WS2 unbiasedness / variance-reduction tests and
  // any host that needs a light's selection pdf independently of the GPU draw.
  sampleLightTreeCPU,
  lightTreePdfCPU,
} from './lightTree.js';
export type {
  LightTreeNode,
  LightTreeBuildInput,
  OrientationCone,
  LightTreeDebugOutput,
} from './lightTree.js';
// Canonical GPU light-tree traversal WGSL (binding-agnostic). Both
// walkaround-hybrid (ReSTIR-DI candidate selection) and pt-webgpu (NEE
// importance sampling) build their light-tree shader from this one source.
export {
  LIGHT_TREE_TRAVERSAL_WGSL,
  LIGHT_TREE_MODULE_NAME,
  lightTreeBindingWgsl,
  lightTreeWgsl,
} from './wgsl/lightTree.wgsl.js';
// ReGIR (Boksansky 2021 grid-based reservoirs) CPU reference core + the packed
// per-cell survivor stride shared with the WGSL grid-build kernel.
export {
  REGIR_FLOATS_PER_SURVIVOR,
  regirBuildSurvivorCPU,
  regirCellTargetFromTree,
  regirCellPmfExact,
} from './regir.js';
export type { ReGIRSurvivor } from './regir.js';
export { balanceHeuristic, powerHeuristic, mixturePdf } from './mixturePdf.js';
export { evaluateHG, sampleHG, pdfHG } from './hgPhase.js';
export { sampleEquiAngular } from './equiAngular.js';
export type { EquiAngularSample, EquiAngularOptions } from './equiAngular.js';
export {
  // `rgbToSpectralCoefficients` is the stable public alias; the underlying
  // `rgbToJakobHanikaCoefficients` runs the genuine Jakob & Hanika 2019
  // Gauss–Newton sigmoid-coefficient solve (see jakobHanika.ts). Tests +
  // production code use the stable name. `spectralCoefficientsToRGB` is the
  // exact inverse (integrate S(λ) under D65 + CMFs → linear sRGB), exported
  // for round-trip verification.
  rgbToSpectralCoefficients,
  evaluateSpectrum,
  spectralCoefficientsToRGB,
} from './jakobHanika.js';
// ── Sprint 10c (BDPT) ────────────────────────────────────────────────────────
// Sprint 10c (BDPT) — applied 2026-05-12. See external_requests/IMPLEMENTATION-STATUS.md §Sprint 10c.
// The _full MIS helpers (T2.H4) are the canonical consumer-facing API.
export {
  BDPT_KIND_LIGHT,
  BDPT_KIND_EYE,
  BDPT_KIND_CONNECTION,
  BDPT_KIND_INVALID,
  BDPT_VERTEX_FLOATS,
  BDPT_VERTEX_BYTES,
  BDPT_MAX_LIGHT_BOUNCES,
  BDPT_MAX_EYE_BOUNCES,
  packBDPTVertex,
  unpackBDPTVertex,
} from './bdptVertex.js';
export type { BDPTVertex } from './bdptVertex.js';
// T2.H4 — Full Veach §10.3 BDPT MIS strategy enumeration.
// Sprint-10c `_partial` variants removed 2026-05-18 (no production consumers).
export { geometricTermG, buildBDPTStrategyPDFs_full, bdptConnectionMIS_full } from './bdptMIS.js';
export type { BDPTFullVertex } from './bdptMIS.js';

// GRIS / ReSTIR-PT reconnection-shift CPU oracle (Lin et al. 2022).
// Phase-0 foundation for evolving ReSTIR-GI toward path reuse: the reconnection
// shift mapping T, its inverse, the reconnection-edge geometry term, and the
// change-of-variables Jacobian |∂T/∂·|. Verified against the Phase-1/2 WGSL,
// the same way bdptConnectionMisFull mirrors bdptMIS.
export {
  reconnectionGeometryTerm,
  reconnectionShift,
  reconnectionShiftInverse,
  reconnectionJacobian,
} from './reconnectionShift.js';
export type { ReconnectionPath } from './reconnectionShift.js';

// Sprint 12 — hero-wavelength spectral path tracing utilities
export {
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  CIE_D65_TABLE,
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_MAX,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
  sampleCMF,
  xyzToLinearSRGB,
} from './cieCmf.js';

export {
  sampleHeroWavelength,
  sampleHeroWavelengthMIS,
  wavelengthToRGB,
  X_CMF_INTEGRAL,
  Y_CMF_INTEGRAL,
  Z_CMF_INTEGRAL,
  X_CMF_CDF,
  Y_CMF_CDF,
  Z_CMF_CDF,
  HERO_LAMBDA_MIN,
  HERO_LAMBDA_MAX,
} from './wavelengthSampling.js';

export {
  cauchyIOR,
  abbeNumber,
  CAUCHY_CROWN_GLASS,
  CAUCHY_FLINT_GLASS,
  CAUCHY_LEAD_CRYSTAL,
  FRAUNHOFER_D_NM,
  FRAUNHOFER_F_NM,
  FRAUNHOFER_C_NM,
} from './cauchyIor.js';

// W2-C13 — declarative UBO codegen (defineUbo)
export { defineUbo } from './uboCodegen.js';
export type {
  UboFieldType,
  UboFieldSpec,
  UboValue,
  UboDefinition,
  UboWgslOptions,
} from './uboCodegen.js';
