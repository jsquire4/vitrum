export {
  MAX_TILE_GRID,
  TileVariancePass,
  computeAdaptiveTileRepeatFactors,
  linearTileIndexFromVarianceReadPixelsPy,
} from './legacy/three/adaptiveTileWeights.js';
export { ZERO_SAMPLE_COUNT_EPSILON } from './accumulationSampleEpsilon.js';
export { readAccumulationRgbFloat, accumulationFloatRgbaToRgb } from './legacy/three/readbackHdr.js';
export {
  HDR_ACCUM_GOLDEN_BASE64,
  HDR_ACCUM_GOLDEN_EXPECTED_RGB_DIVIDE,
  HDR_ACCUM_GOLDEN_PIXEL_COUNT,
  HDR_ACCUM_GOLDEN_BYTE_LENGTH,
  decodeHdrAccumGoldenBin,
  hdrAccumGoldenBinFromBase64,
} from './hdrGoldenFixture.js';
export { applyFrameToPerspectiveCamera } from './legacy/three/frameCamera.js';
export { packCameUBO } from './cameUniformUploader.js';
export type { CameSegment, CameNode, CameUploadOptions, CamePackedUBO } from './cameUniformUploader.js';

export { PTEngineWebGL2, createPTEngine_WebGL2 } from './ptEngineWebGL2.js';
export type {
  PTEngineWebGL2Options,
  PTEngineWebGL2QualityMode,
  PTEngineWebGL2Telemetry,
  PTEngineWebGL2FrameOutput,
  PTEngineWebGL2Surface,
} from './ptEngineWebGL2.js';

// Task 4.4 Theme A — scheduler state machine + the `updatePrimitive` patch
// router extracted out of the engine god-class.
export {
  AdaptiveScheduler,
  DEFAULT_TILE_SIZE,
} from './adaptiveScheduler.js';
export type {
  SchedulerOptions,
  SchedulerDeviceLimits,
  RenderSizePlan,
} from './adaptiveScheduler.js';
export { routePrimitivePatch } from './legacy/three/scenePatch.js';
export type {
  PrimitivePatchContext,
  RoutePrimitivePatchOutcome,
} from './legacy/three/scenePatch.js';

// W11 follow-up — OIDN final-pass dispatcher (internal kick-and-return state
// machine; hosts query the engine via `getDenoisedFrame()` rather than touching
// the dispatcher directly, but the types are exported for test harnesses).
export { OIDNFinalDispatcher } from './legacy/three/oidnFinalDispatcher.js';
export type {
  OIDNFinalDispatcherOptions,
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
} from './legacy/three/oidnFinalDispatcher.js';

export * from './constants.js';
// sunGeometry now lives in @vitrum/scene-lighting (re-exported below via
// `export * from '@vitrum/scene-lighting'`).
export { IblBakerCache } from './legacy/three/iblBaker.js';
export type { IblBakerCacheOptions } from './legacy/three/iblBaker.js';
export { debounceMsForEditRate, PT_DEBOUNCE_MS_NORMAL, PT_DEBOUNCE_MS_BURST } from './debounce.js';

// Re-exported from @vitrum/scene-lighting — the four lighting-state modules
// (sun geometry, time-of-day → SkyParams, intensity table, unified LightingState)
// were previously colocated here but are equally consumed by walkaround-hybrid;
// they live in their own backend-agnostic package now. The re-exports below
// keep the pt-webgl public surface unchanged for existing callers.
export * from '@vitrum/scene-lighting';

// Sprint 10c — BDPT option types for host callers that drive fork uniforms directly.
export type { ForkBridgeBdptOptions, ForkBridgeCausticOptions } from './legacy/three/forkUniformBridge.js';
export { driveForkMaterialUniforms } from './legacy/three/forkUniformBridge.js';

// Sprint 10c follow-up (C3, 2026-05-19) — host-side light-path texture
// helper for BDPT. The engine's bdptAdvanceFrame() method takes the
// helper's texture and forwards it to the fork's uniforms per frame.
export { BdptLightPathBuffer } from './legacy/three/bdptLightPathBuffer.js';
export type { BdptLightPathBufferOptions } from './legacy/three/bdptLightPathBuffer.js';
export { bdptLightPathTextureType } from './legacy/three/bdpt/lightPathTextureType.js';
export { fillBdptLightPathWebGL } from './legacy/three/bdpt/fillBdptLightPathWebGL.js';
export { runBdptLightSubpathPass } from './legacy/three/bdpt/runBdptLightSubpathPass.js';
export type { BdptLightSubpathTracer } from './legacy/three/bdpt/runBdptLightSubpathPass.js';
export { sampleBdptBounce0FromScene } from './legacy/three/bdpt/bdptSceneEmittersCpu.js';

export { auditPtWebglSceneForTlas } from './sceneTlasAudit.js';
export type { PtWebglTlasAudit } from './sceneTlasAudit.js';
