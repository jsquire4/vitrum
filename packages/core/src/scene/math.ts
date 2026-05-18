// Scene description — backend-agnostic.
//
// Math primitives (these are exported for hosts to construct against).

export type Vec2 = readonly [number, number];
export type Vec3 = readonly [number, number, number];
export type Vec4 = readonly [number, number, number, number];

/** Column-major 4×4 matrix, 16 elements. Matches Three.js + WebGPU/WebGL convention. */
export type Mat4 = Float32Array;

/** A monotonic, host-supplied identifier. Stable across `setScene` calls so
 *  backends can do incremental updates. Hosts should use whatever their scene
 *  graph uses — three.js `Object3D.uuid`, integer counters, etc. */
export type SceneNodeId = string;
