/**
 * @vitrum/scene-lighting — host-side lighting-state primitives shared across
 * vitrum render backends.
 *
 * The four modules below are backend-agnostic: pure TypeScript computing
 * scene-lighting values that each renderer uploads to its own UBO or light
 * representation.
 *
 * Currently consumed by the native path-tracing and walkaround renderers.
 */

export * from './sunGeometry.js';

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
