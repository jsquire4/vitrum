/**
 * @vitrum/scene-lighting — host-side lighting-state primitives shared across
 * vitrum render backends (`@vitrum/pt-webgl`, `@vitrum/walkaround-hybrid`, …).
 *
 * The four modules below were previously colocated inside `@vitrum/pt-webgl/src/`
 * even though walkaround-hybrid and pt-webgl both consume them. They are
 * backend-agnostic: pure TypeScript computing scene-lighting values that each
 * renderer then uploads to its own UBO / three.js light.
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
