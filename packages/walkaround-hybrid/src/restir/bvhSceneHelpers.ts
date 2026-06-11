/**
 * Back-compat re-export barrel for bvhSceneHelpers.
 *
 * The implementation was split into:
 *   - sceneHelpers.ts       (core scene-graph utilities)
 *   - emitterHelpers.ts     (emitter-packing functions)
 *
 * All existing imports of this module continue to work unchanged.
 */

export { enrichMeshVertexRangesWithCoreMatrix } from './sceneHelpers.js';

export type { ExtraEmitterTri, PackedAnalyticLights } from './emitterHelpers.js';
export {
  packEmitterTrisForDDGI,
  collectRectAreaEmitterTrisFromCore,
  collectMeshAreaEmitterTrisFromCore,
  packAnalyticPointSpotEmitters,
} from './emitterHelpers.js';
