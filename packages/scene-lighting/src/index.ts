/**
 * @vitrum/scene-lighting — host-side lighting-state primitives shared across
 * vitrum render backends.
 *
 * The four modules below are backend-agnostic: pure TypeScript computing
 * scene-lighting values that each renderer uploads to its own UBO or light
 * representation.
 *
 * Consumed by host applications and tools that need lighting-state primitives
 * (e.g. time-of-day sun position, sky params, intensity tables). The runtime
 * backends do not import this package directly — they receive pre-computed
 * lighting values via their own EngineOptions / UBO upload paths.
 */

export * from './sunGeometry.js';

export { computeLightingState } from './lightingState.js';
export type { LightingState, LightingStateInputs } from './lightingState.js';

export { skyParamsFor, worldSunPosition, SUN_LIGHT_DISTANCE, SUN_Z_DEPTH_SCALE } from './skyParams.js';
export type { SkyParams, SkyParamsOptions } from './skyParams.js';

export {
  COLOR_TEMP_HEX,
  SUN_INTENSITY,
  getSunIntensity,
  pointIntensityFromLumens,
  rectAreaIntensityFromLumens,
} from './lightingIntensityTable.js';
