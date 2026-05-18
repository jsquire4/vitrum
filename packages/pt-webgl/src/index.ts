export {
  MAX_TILE_GRID,
  TileVariancePass,
  computeAdaptiveTileRepeatFactors,
  linearTileIndexFromVarianceReadPixelsPy,
} from './adaptiveTileWeights.js';
export { ZERO_SAMPLE_COUNT_EPSILON } from './accumulationSampleEpsilon.js';
export { readAccumulationRgbFloat, accumulationFloatRgbaToRgb } from './readbackHdr.js';
export {
  HDR_ACCUM_GOLDEN_BASE64,
  HDR_ACCUM_GOLDEN_EXPECTED_RGB_DIVIDE,
  HDR_ACCUM_GOLDEN_PIXEL_COUNT,
  HDR_ACCUM_GOLDEN_BYTE_LENGTH,
  decodeHdrAccumGoldenBin,
  hdrAccumGoldenBinFromBase64,
} from './hdrGoldenFixture.js';
export { applyFrameToPerspectiveCamera } from './frameCamera.js';
export { packCameUBO } from './cameUniformUploader.js';
export type { CameSegment, CameNode, CameUploadOptions, CamePackedUBO } from './cameUniformUploader.js';

export { PTEngineWebGL2, createPTEngine_WebGL2 } from './ptEngineWebGL2.js';
export type {
  PTEngineWebGL2Options,
  PTEngineWebGL2QualityMode,
  PTEngineWebGL2Telemetry,
  PTEngineWebGL2FrameOutput,
} from './ptEngineWebGL2.js';

// W11 follow-up — OIDN final-pass dispatcher (internal kick-and-return state
// machine; hosts query the engine via `getDenoisedFrame()` rather than touching
// the dispatcher directly, but the types are exported for test harnesses).
export { OIDNFinalDispatcher } from './oidnFinalDispatcher.js';
export type {
  OIDNFinalDispatcherOptions,
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
} from './oidnFinalDispatcher.js';

export * from './constants.js';
export * from './sunGeometry.js';
export { IblBakerCache, bakeSkyEquirect, clearSkyEquirectCache } from './iblBaker.js';
export type { IblBakerCacheOptions } from './iblBaker.js';
export { debounceMsForEditRate, PT_DEBOUNCE_MS_NORMAL, PT_DEBOUNCE_MS_BURST } from './debounce.js';
export { computeLightingState } from './lightingState.js';
export type { LightingState, LightingStateInputs } from './lightingState.js';
export { skyParamsFor, worldSunPosition, SUN_LIGHT_DISTANCE } from './skyParams.js';
export type { SkyParams } from './skyParams.js';
export {
  COLOR_TEMP_HEX,
  SUN_INTENSITY,
  getSunIntensity,
  pointIntensityFromLumens,
  rectAreaIntensityFromLumens,
} from './lightingIntensityTable.js';

// Sprint 10c — BDPT option types for host callers that drive fork uniforms directly.
export type { ForkBridgeBdptOptions, ForkBridgeCausticOptions } from './forkUniformBridge.js';
export { driveForkMaterialUniforms } from './forkUniformBridge.js';
