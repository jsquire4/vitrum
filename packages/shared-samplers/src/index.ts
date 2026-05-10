// @vitrum/shared-samplers — sampling utilities for path tracers and walkaround engines.
//
// Phase 1 deliverable: Hammersley QMC sequence + sphere sampling WGSL fragment.
// Sprint 3 (Phase 6): light tree CDF (CPU-side build + GPU pack), mixture PDF / MIS heuristics.
// Future Phase 6 sprints add: Sobol QMC (Sprint 3 fork-side), HG phase function (Sprint 7),
// Jakob+Hanika spectral upsampling (Sprint 8b), Welford variance struct (Sprint 9 rider).

export * from './wgsl/hammersley.wgsl.js';
export { buildLightTree, packLightTreeForGPU } from './lightTree.js';
export type { LightTreeNode, LightTreeBuildInput } from './lightTree.js';
export { balanceHeuristic, powerHeuristic, mixturePdf } from './mixturePdf.js';
