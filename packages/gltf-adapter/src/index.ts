// @vitrum/gltf-adapter — glTF 2.0 → @vitrum/core Scene adapter.
//
// Primary entry point: gltfToScene.
// animationNodeId builds the stable channel-target node id (`gltf-node-<i>`)
// used by result.animations / result.animationTargets.
// Re-exports the GltfJson type for hosts that parse glTF JSON before passing it in.
// RawImageHandle is exported so hosts can type-narrow in non-browser environments.

export { gltfToScene } from './gltfToScene.js';
export type { GltfToSceneOptions, GltfToSceneResult } from './gltfToScene.js';
export { animationNodeId } from './animations.js';
export type { GltfJson } from './gltfTypes.js';
export type { DecodeImageFn, RawImageHandle } from './textures.js';
