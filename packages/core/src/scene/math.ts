// Scene description — backend-agnostic.
//
// Math primitives (these are exported for hosts to construct against).

export type Vec2 = readonly [number, number];
export type Vec3 = readonly [number, number, number];
export type Vec4 = readonly [number, number, number, number];

/** Column-major 4×4 matrix, 16 elements. Matches Three.js + WebGPU/WebGL convention. */
declare const MAT4_BRAND: unique symbol;
export type Mat4 = Float32Array & { readonly [MAT4_BRAND]: 'Mat4' };

/** Validate/cast a typed array into a branded Mat4. */
export function asMat4(value: Float32Array | ReadonlyArray<number>): Mat4 {
  if (value.length !== 16) {
    throw new RangeError(`Mat4 requires 16 elements, got ${value.length}`);
  }
  if (value instanceof Float32Array) {
    return value as Mat4;
  }
  return new Float32Array(value) as Mat4;
}

/** Runtime guard for externally-provided matrix values. */
export function isMat4(value: unknown): value is Mat4 {
  return value instanceof Float32Array && value.length === 16;
}

/** A monotonic, host-supplied identifier. Stable across `setScene` calls so
 *  backends can do incremental updates. Hosts should use whatever their scene
 *  graph uses — three.js `Object3D.uuid`, integer counters, etc. */
export type SceneNodeId = string;
