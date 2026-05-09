import type { Object3D, Scene } from 'three';

/**
 * Root object(s) accepted by `@vitrum/shared-bvh` / ReSTIR `buildSceneBVH`.
 * Today this is always a THREE scene graph; the alias documents the seam for
 * a future raw-buffer or other-host implementation.
 */
export type WalkaroundBVHSceneRoot = Scene | Object3D;

/**
 * DDGI’s probe update path expects a full THREE.Scene (traverse, materials,
 * BVH from merged geometry). Hosts **must** pass a real `Scene` until DDGI’s
 * `SceneBvh` is generalized.
 */
export type WalkaroundDDGIScene = Scene;

/**
 * Convenience: hosts that use three.js supply one `Scene` for both BVH and
 * DDGI. `HybridEngineOptions.threeScene` should satisfy this.
 */
export type WalkaroundThreeHostScene = Scene;
