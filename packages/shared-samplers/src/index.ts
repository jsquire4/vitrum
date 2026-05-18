// @vitrum/shared-samplers — sampling utilities for path tracers and walkaround engines.
//
// Phase 1 deliverable: Hammersley QMC sequence + sphere sampling WGSL fragment.
// Sprint 3 (Phase 6): light tree CDF (CPU-side build + GPU pack), mixture PDF / MIS heuristics.
// Sprint 7 (Phase 6): HG phase function + equi-angular volume distance sampling.
// Sprint 8b (Phase 6): Jakob+Hanika spectral upsampling.
// Sprint 10c (Phase 6): BDPT vertex layout + connection-PMF MIS weights.
// Sprint 12 (Phase 6): CIE CMF tables, hero-wavelength sampling, Cauchy IOR.
// Future: Sobol QMC (Sprint 3 fork-side), Welford variance struct (Sprint 9 rider).

export * from './wgsl/hammersley.wgsl.js';
export { OCTAHEDRAL_CORE_WGSL } from './wgsl/octahedralCore.wgsl.js';
export { buildLightTree, packLightTreeForGPU } from './lightTree.js';
export type { LightTreeNode, LightTreeBuildInput } from './lightTree.js';
export { balanceHeuristic, powerHeuristic, mixturePdf } from './mixturePdf.js';
export { evaluateHG, sampleHG, pdfHG } from './hgPhase.js';
export { sampleEquiAngular } from './equiAngular.js';
export type { EquiAngularSample, EquiAngularOptions } from './equiAngular.js';
export {
  rgbToApproxSpectralCoefficients,
  rgbToSpectralCoefficients,
  evaluateSpectrum,
} from './jakobHanika.js';
// ── Sprint 10c (BDPT) ────────────────────────────────────────────────────────
// Sprint 10c is open (un-deferred by user 2026-05-12). The _full MIS helpers
// (T2.H4) are the canonical consumer-facing API. Fork-side GLSL dispatch is
// BLOCKED on Sprints 4–6 fork patches (no spec or fork commits exist for those
// sprints yet). See plan/sprint-10c-pt-fork-patch.md §Status for full blocker
// detail and external_requests/IMPLEMENTATION-STATUS.md §Sprint 10c.
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
export { bdptConnectionMIS_partial, buildBDPTStrategyPDFs_partial } from './bdptMIS.js';
// T2.H4 — Full Veach §10.3 BDPT MIS strategy enumeration
export {
  geometricTermG,
  buildBDPTStrategyPDFs_full,
  bdptConnectionMIS_full,
} from './bdptMIS.js';
export type { BDPTFullVertex } from './bdptMIS.js';

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
