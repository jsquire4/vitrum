// Scene description — backend-agnostic.
//
// Primitives — geometry that occupies space.

import type { Mat4, SceneNodeId } from './math.js';
import type { MaterialSpec } from './material.js';

/** Triangle mesh. Position/normal/uv arrays follow three.js convention:
 *  flat Float32Arrays where consecutive triples (or pairs for uv) describe
 *  one vertex. `indices` is optional; without it, vertices are interpreted
 *  as triangle-list. */
export interface MeshPrimitive {
  readonly kind: 'mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly tangents?: Float32Array;       // xyzw per vertex; w = bitangent sign
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: MaterialSpec;
  readonly transform?: Mat4;              // identity if absent
  readonly castShadow?: boolean;          // default true
  readonly receiveShadow?: boolean;       // default true
}

/** Same geometry repeated at many transforms. Backend may build a single BVH
 *  once and traverse via instance transforms. */
export interface InstancedMeshPrimitive {
  readonly kind: 'instanced-mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly tangents?: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: MaterialSpec;
  readonly instances: ReadonlyArray<Mat4>;
}

/** Closed-form ray-primitive intersection. Backend-supported shapes only;
 *  unsupported shapes log a warning and degrade to skip (or to mesh
 *  tessellation if a fallback geometry is provided).
 *
 *  Phase 6 sprint 5 introduces 'h-channel-came' for our analytic came/solder
 *  geometry. Future shapes (gemstones via 'ellipsoid', pillars via 'capsule',
 *  etc.) extend this discriminated union without breaking existing scenes.
 */
export interface AnalyticPrimitive {
  readonly kind: 'analytic';
  readonly id: SceneNodeId;
  readonly shape: AnalyticShape;
  readonly params: Float32Array;          // shape-specific layout, see AnalyticShape
  readonly material: MaterialSpec;
  readonly transform?: Mat4;
  readonly fallbackMesh?: Omit<MeshPrimitive, 'kind' | 'id' | 'material' | 'transform'>;
}

export type AnalyticShape =
  | 'sphere'           // params: [cx, cy, cz, radius]
  | 'box'              // params: [cx, cy, cz, hx, hy, hz]
  | 'capsule'          // params: [ax, ay, az, bx, by, bz, radius]
  | 'cylinder'         // params: [cx, cy, cz, radius, halfHeight]
  | 'h-channel-came';  // params: [length, railWidth, blockHeight, webThickness] — H-channel rail primitive, Phase 6 sprint 5

export type ScenePrimitive =
  | MeshPrimitive
  | InstancedMeshPrimitive
  | AnalyticPrimitive;
