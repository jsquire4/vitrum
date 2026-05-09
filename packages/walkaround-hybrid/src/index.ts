// @vitrum/walkaround-hybrid — WebGPU layered DDGI + RC + ReSTIR DI engine.

// DDGI subsystem
export { ProbeUpdatePass } from './ddgi/probeUpdatePass.js';
export type { ProbeUpdatePassOptions } from './ddgi/probeUpdatePass.js';
export { ProbeGrid } from './ddgi/probeGrid.js';
export type { ProbeGridDims, ProbeGridParams } from './ddgi/probeGrid.js';
export * from './ddgi/ddgiAtlasLayout.js';
export { DDGI_SAMPLE_WGSL } from './ddgi/ddgiSampleWgsl.js';
export type { DDGILight } from './ddgi/types.js';

// ReSTIR pipeline
export {
  WalkaroundGPUPipeline,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
} from './pipeline/WalkaroundGPUPipeline.js';
export type { PipelineFrameInputs } from './pipeline/WalkaroundGPUPipeline.js';
export {
  buildSceneBVH,
  disposeSceneBVH,
} from './restir/bvhCompute.js';
export type { SceneBVHBuffers } from './restir/bvhCompute.js';

// WGSL shader strings (consumed by pipelineCompiler internally; re-exported
// so host apps can inspect or extend them).
export { COMMON_WGSL } from './shaders/common.wgsl.js';
export { RIS_WGSL } from './shaders/ris.wgsl.js';
export { TEMPORAL_WGSL } from './shaders/temporal.wgsl.js';
export { SPATIAL_WGSL } from './shaders/spatial.wgsl.js';
export { SHADE_WGSL } from './shaders/shade.wgsl.js';
export { ATROUS_WGSL } from './shaders/atrous.wgsl.js';
export { TEMPORAL_ACCUM_WGSL } from './shaders/temporalAccum.wgsl.js';
export { COMPOSITE_VERT_WGSL, COMPOSITE_FRAG_WGSL } from './shaders/composite.wgsl.js';

// Shared lib utilities
export { upgradeToNodeMaterial } from './lib/nodeMaterialUpgrade.js';
