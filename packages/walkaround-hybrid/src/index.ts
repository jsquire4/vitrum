// @vitrum/walkaround-hybrid — WebGPU layered DDGI + RC + ReSTIR DI engine.

// Engine class (the public Engine implementation)
export { HybridEngine, createWalkaroundEngine_Hybrid } from './HybridEngine.js';
export type { HybridEngineOptions } from './HybridEngine.js';

// DDGI subsystem (class-based, de-React-ified)
export { DDGI } from './ddgi/DDGI.js';
export type { DDGIOptions, DDGIFrameInputs } from './ddgi/DDGI.js';

// DDGI subsystem (lower-level)
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
export { ATROUS_WGSL, TEMPORAL_ACCUM_WGSL } from '@vitrum/shared-denoisers';
export { COMPOSITE_VERT_WGSL, COMPOSITE_FRAG_WGSL } from './shaders/composite.wgsl.js';

// Shared lib utilities
export { upgradeToNodeMaterial } from './lib/nodeMaterialUpgrade.js';

// ─── RC subsystem ─────────────────────────────────────────────────────────────
// Cascade pyramid storage layout.
export {
  CASCADE_DIMS,
  CASCADE_COUNT,
  allocateCascades,
  disposeCascades,
  fillCascadeDebug,
} from './rc/cascadePyramid.js';
export type { CascadeDim, CascadeBuffers } from './rc/cascadePyramid.js';

// RC BVH builder (StorageBufferAttribute-typed adapter over @vitrum/shared-bvh).
export { buildSceneBVH as buildRCSceneBVH } from './rc/bvhCompute.js';
export type { SceneBVH as RCSceneBVH, BvhBuildOpts as RCBvhBuildOpts } from './rc/bvhCompute.js';

// Cascade dispatch — raw WebGPU compute (converted from TSL per RD-12).
export { RCDispatcher, dispatchCascadePasses } from './rc/cascadeDispatch.js';
export type { RCDispatchOpts } from './rc/cascadeDispatch.js';

// Cascade buffer manager (de-React-ified from useCascadeBuffers).
export { CascadeBufferManager } from './rc/cascadeBuffers.js';

// GI receiver material wrapper (TSL-preserved; requires three/webgpu + three/tsl).
export { GIReceiver } from './rc/giReceiver.js';
export type { GIReceiverExclusionPredicate, GIReceiverOptions } from './rc/giReceiver.js';

// Walkaround diffuse lighting node (TSL-preserved; requires three/tsl).
export { buildWalkaroundLightingNode } from './rc/walkaroundDiffuseLighting.js';
export type { WalkaroundLightingNodes } from './rc/walkaroundDiffuseLighting.js';

// DDGI shading injection (TSL-preserved; requires three/webgpu + three/tsl).
export { applyDDGIShading } from './rc/applyDDGIShading.js';

// Raw WGSL shader strings (for host inspection or headless testing).
export { PROBE_RAY_CAST_WGSL } from './rc/wgsl/probeRayCast.wgsl.js';
export { CASCADE_MERGE_WGSL } from './rc/wgsl/cascadeMerge.wgsl.js';
