// Optional THREE/TSL bridge for @vitrum/walkaround-hybrid.
// Import this subpath only from hosts that already depend on three/webgpu +
// three/tsl. The package root intentionally exposes engine-facing APIs only.

import type { HybridEngineOptions } from './HybridEngineOptions.js';
import type { WalkaroundThreeHostScene } from './three/hostSceneTypes.js';

export { HybridEngine, createWalkaroundEngine_Hybrid } from './HybridEngine.js';
export { applyDDGIShading, disposeApplyDDGIShadingCache } from './three/applyDDGIShading.js';
export { upgradeToNodeMaterial } from './three/nodeMaterialUpgrade.js';
export type { HybridEngineGISurface } from './HybridEnginePublic.js';
export type {
  WalkaroundBVHSceneRoot,
  WalkaroundDDGIScene,
  WalkaroundThreeHostScene,
} from './three/hostSceneTypes.js';

export type HybridEngineThreeOptions = Omit<HybridEngineOptions, 'threeScene'> & {
  readonly threeScene?: WalkaroundThreeHostScene;
};
