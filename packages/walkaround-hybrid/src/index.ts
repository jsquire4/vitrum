// @vitrum/walkaround-hybrid — WebGPU ReSTIR DI + DDGI engine (RC subsystem exported separately; see README).

// Host / binding seams (THREE coupling documented for future non-THREE graphs)
export type {
  WalkaroundBVHSceneRoot,
  WalkaroundDDGIScene,
  WalkaroundThreeHostScene,
} from './hostScene/types.js';

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
  HYBRID_WEBGPU_REQUIRED_FEATURES,
} from './pipeline/WalkaroundGPUPipeline.js';
export type { PipelineFrameInputs } from './pipeline/WalkaroundGPUPipeline.js';
export {
  buildReSTIRSceneBVH,
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
export { COMPOSITE_VERT_WGSL, COMPOSITE_FRAG_WGSL } from './shaders/composite.wgsl.js';

// Shared lib utilities
export { upgradeToNodeMaterial } from './lib/nodeMaterialUpgrade.js';

// Wire contract: surface-texture id enum (consumed by host scene
// bindings that stamp `userData.surfaceTextureId`, by `packBVHIndexW`,
// and by the WGSL `surfaceTextureMod` switch in
// shaders/surfaceTextures.wgsl.ts).
export {
  SURFACE_TEXTURE_ID,
  type SurfaceTextureName,
  type SurfaceTextureId,
} from './surfaceTextureIds.js';

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
export { buildRCSceneBVH } from './rc/bvhCompute.js';
export type { SceneBVH as RCSceneBVH, BvhBuildOpts as RCBvhBuildOpts } from './rc/bvhCompute.js';

// Cascade dispatch — raw WebGPU compute (converted from TSL per RD-12).
export { RCDispatcher, dispatchCascadePasses, disposeSharedDispatcher } from './rc/cascadeDispatch.js';
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

export type { FrameResourceOptions } from './pipeline/resourceManager.js';

// ─── PPG (T2.H3 — Practical Path Guiding, Müller et al. 2017) ────────────────
// CPU-side adaptive sTree + dTree + MIS combination.
// Enable via HybridEngineOptions.ppgEnabled: true (default false).
export {
  buildSTree,
  findSTreeLeaf,
  sTreeAccumulate,
  splitOverflowLeaves,
  resetAccumulators,
  aabbContains,
} from './ppg/sTree.js';
export {
  buildEmptyDTree,
  findDTreeLeaf,
  dTreeSample,
  dTreePdf,
  refineDTree,
  sumLeafSolidAngles,
  sumLeafPdfIntegrals,
} from './ppg/dTree.js';
export { computeMISWeights } from './ppg/ppgGuide.wgsl.js';
export {
  PPG_CELL_SPLIT_THRESHOLD,
  PPG_DTREE_FLUX_FRACTION,
  PPG_DTREE_MERGE_FRACTION,
  PPG_DTREE_MAX_DEPTH,
  PPG_DTREE_INITIAL_DEPTH,
  PPG_MIS_ALPHA,
  PPG_MIS_ALPHA_MIN,
  PPG_MIS_ALPHA_MAX,
  PPG_MAX_SPATIAL_CELLS,
} from './ppg/ppgConstants.js';
export type { AABB, STreeNode, DTreeNode, DTree, STree, PPGModelHandle } from './ppg/types.js';
// WGSL kernel strings (for host inspection or test assertions).
export { PPG_UPDATE_WGSL } from './ppg/ppgUpdate.wgsl.js';
export { PPG_GUIDE_WGSL } from './ppg/ppgGuide.wgsl.js';
// W9 — serialisation (CPU producers + GPU-equivalent traversal oracles).
export {
  serialiseDTree,
  serialiseSTree,
  gpuTraverseDTreeLeaf,
  gpuTraverseSTreeLeaf,
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  STREE_HEADER_F32,
  STREE_NODE_F32,
} from './ppg/serialise.js';
export type { SerialisedSTree } from './ppg/serialise.js';

// ─── Neural denoiser (T2.H2) ──────────────────────────────────────────────────
// InferenceGraph + weights loader for the U-Net neural denoiser.
// The 'neural' denoiser mode in HybridEngine is opt-in (default: atrous-variance).
// Load weights via loadWeightsFromArrayBuffer() from a .vitrum-model binary.
export { InferenceGraph } from './neural/InferenceGraph.js';
export { buildUNetSpec, WALKAROUND_DENOISER_UNET_SPEC } from './neural/unetArchitecture.js';
export type { UNetSpec, LayerSpec, LayerKind, LayerWeightLayout, LayerParams } from './neural/unetArchitecture.js';
export {
  loadWeightsFromArrayBuffer,
  serializeWeightsToArrayBuffer,
  VITRUM_MODEL_MAGIC,
  VITRUM_MODEL_VERSION,
} from './neural/weights.js';
export type { ModelWeights, LayerWeights } from './neural/weights.js';
