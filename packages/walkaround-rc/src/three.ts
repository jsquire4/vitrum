// Optional THREE/TSL bridge for @vitrum/walkaround-rc.
// Import this subpath only from hosts that already depend on three/webgpu +
// three/tsl. The package root intentionally stays raw-WebGPU/runtime safe.

export { allocateCascades, disposeCascades, fillCascadeDebug } from './three/cascadePyramidThree.js';
export type { CascadeBuffers } from './three/cascadePyramidThree.js';
export { CascadeBufferManager } from './three/cascadeBuffers.js';
export { GIReceiver } from './three/giReceiver.js';
export type { GIReceiverExclusionPredicate, GIReceiverOptions } from './three/giReceiver.js';
export { buildWalkaroundLightingNode } from './three/walkaroundDiffuseLighting.js';
export type { WalkaroundLightingNodes } from './three/walkaroundDiffuseLighting.js';
