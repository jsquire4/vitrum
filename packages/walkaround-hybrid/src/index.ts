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

// ─── Sprint 13 — Neural denoiser ─────────────────────────────────────────────
// InferenceGraph: WebGPU compute-shader inference graph for UNet-style neural denoising.
// Mode scope: walkaround only (PT final uses Sprint 10b OIDN).

// Inference graph orchestrator and types.
export { InferenceGraph } from './neural/InferenceGraph.js';
export type {
  InferenceLayer,
  InferenceLayerKind,
  InferenceGraphSpec,
  ModelWeights,
} from './neural/InferenceGraph.js';

// UNet architecture spec and constants.
export {
  WALKAROUND_DENOISER_UNET_SPEC,
  buildUNetSpec,
  UNET_INPUT_CHANNELS,
  UNET_OUTPUT_CHANNELS,
  UNET_ENCODER_CHANNELS,
  UNET_DECODER_CHANNELS,
  UNET_TOTAL_PARAMETERS,
  UNET_WEIGHT_BYTES,
  UNET_INPUT_TENSOR_NAMES,
  UNET_OUTPUT_TENSOR_NAMES,
} from './neural/unetArchitecture.js';

// WGSL primitive kernels (exported for host inspection and headless testing).
export { CONV2D_WGSL } from './neural/wgsl/conv2d.wgsl.js';
export { TRANSPOSED_CONV2D_WGSL } from './neural/wgsl/transposedConv2d.wgsl.js';
export { RELU_WGSL } from './neural/wgsl/relu.wgsl.js';
export { SKIP_CONNECTION_WGSL } from './neural/wgsl/skipConnection.wgsl.js';
export { BILINEAR_UPSAMPLE_WGSL } from './neural/wgsl/bilinearUpsample.wgsl.js';

// ─── Sprint 11 — PPG (path guiding) ──────────────────────────────────────────
// PPG is walkaround-only. WebGL2 PT has no compute shaders and cannot
// maintain the kd-tree update pass.

// Type definitions (CPU-side) + GPU layout constants.
export type { PPGDirectionalBin, PPGQuadTreeNode, PPGSpatialCell, PPGBufferOptions } from './ppg/types.js';
export {
  PPG_MAX_SPATIAL_CELLS,
  PPG_DIRECTIONS,
  PPG_CELL_BYTE_STRIDE,
  PPG_LEAF_BYTE_STRIDE,
  PPG_KD_NODE_BYTE_STRIDE,
  PPG_KD_MAX_NODES,
} from './ppg/types.js';

// PPG update-pass WGSL (the live training kernel; dispatched from
// WalkaroundGPUPipeline when PPG is enabled). The companion sample-pass
// fragment was deleted in P3-C.2 — `shadePpgGuide.wgsl.ts` provides the
// guided indirect bounce via marker-injection into shade.wgsl with
// real @group(3) bindings.
export { PPG_UPDATE_WGSL } from './ppg/wgsl/ppgUpdate.wgsl.js';

// Buffer allocation helpers.
export {
  createPPGBuffers,
  destroyPPGBuffers,
  writePpgKdTree,
} from './pipeline/resourceManager.js';
export {
  buildPpgKdTreeGpuBytes,
  encodePpgKdDisabledRoot,
  ppgNearestCellIndexKd,
  ppgNearestCellIndexBrute,
} from './ppg/buildPpgKdTree.js';
export {
  aabbFromBvhPositions,
  buildPpgUniformGridCells,
  encodePpgCellGpuBytes,
} from './ppg/ppgCellUpload.js';
export type { PpgCellPosition } from './ppg/ppgCellUpload.js';
export type { PPGBuffers, FrameResourceOptions } from './pipeline/resourceManager.js';
