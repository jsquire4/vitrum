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
// ── Sprint 10c (BDPT) — DEFERRED ──────────────────────────────────────────────
// Trigger criterion: Sprint 7 hero-render floor-caustic noise exceeds threshold.
// Until Sprint 10c is officially opened, these exports are present but not
// integrated. See plan/sprint-10c-pt-fork-patch.md for the full spec and
// plan/archive/phase-6-roadmap.md §Sprint 10c for the trigger gate.
// AUDIT NOTE L-3 (2026-05-09): Exports appear in the public API before
// integration testing is complete. They compile and are tested structurally
// in __tests__/bdpt.test.ts, but the fork-side dispatch is deferred.
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
  wavelengthToRGB,
  Y_CMF_INTEGRAL,
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
