// @vitrum/shared-samplers — sampling utilities for path tracers and walkaround engines.
//
// Phase 1 deliverable: Hammersley QMC sequence + sphere sampling WGSL fragment.
// Sprint 3 (Phase 6): light tree CDF (CPU-side build + GPU pack), mixture PDF / MIS heuristics.
// Sprint 7 (Phase 6): HG phase function + equi-angular volume distance sampling.
// Sprint 8b (Phase 6): Jakob+Hanika spectral upsampling.
// Future: Sobol QMC (Sprint 3 fork-side), Welford variance struct (Sprint 9 rider).

export * from './wgsl/hammersley.wgsl.js';
export { buildLightTree, packLightTreeForGPU } from './lightTree.js';
export type { LightTreeNode, LightTreeBuildInput } from './lightTree.js';
export { balanceHeuristic, powerHeuristic, mixturePdf } from './mixturePdf.js';
export { evaluateHG, sampleHG, pdfHG } from './hgPhase.js';
export { sampleEquiAngular } from './equiAngular.js';
export type { EquiAngularSample } from './equiAngular.js';
export { rgbToSpectralCoefficients, evaluateSpectrum } from './jakobHanika.js';
