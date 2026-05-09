// @vitrum/shared-samplers — sampling utilities for path tracers and walkaround engines.
//
// Phase 1 deliverable: Hammersley QMC sequence + sphere sampling WGSL fragment.
// Future Phase 6 sprints add: Sobol QMC (Sprint 3), light tree CDF (Sprint 3),
// HG phase function (Sprint 7), Jakob+Hanika spectral upsampling (Sprint 8b),
// Welford variance struct (Sprint 9 rider).

export * from './wgsl/hammersley.wgsl.js';
