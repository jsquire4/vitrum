// Scene description — backend-agnostic.
//
// Emitters — anything that gives off light.

import type { SceneNodeId, Vec3 } from './math.js';

export type SceneEmitter =
  | DirectionalEmitter
  | DiscAreaEmitter
  | RectAreaEmitter
  | PointEmitter
  | SpotEmitter
  | MeshAreaEmitter;

export interface EmitterBase {
  readonly id: SceneNodeId;
  readonly color: Vec3;
  readonly intensity: number;
  /** Whether this emitter's shadow rays are cast. Default true.
   *  @reserved Accepted; not yet consumed by any backend (road-to-100 shadow tier). */
  readonly castShadow?: boolean;
}

export interface DirectionalEmitter extends EmitterBase {
  readonly kind: 'directional';
  readonly direction: Vec3;               // unit vector pointing AT the light
  /** Optional: angular subtense for soft shadows. 0 = perfectly directional.
   *  @reserved Accepted; not yet consumed by any backend (road-to-100 soft-shadow tier). */
  readonly angularDiameter?: number;
}

export interface DiscAreaEmitter extends EmitterBase {
  readonly kind: 'disc-area';
  readonly position: Vec3;
  readonly normal: Vec3;
  readonly radius: number;
}

export interface RectAreaEmitter extends EmitterBase {
  readonly kind: 'rect-area';
  readonly position: Vec3;
  readonly uAxis: Vec3;                   // half-width vector
  readonly vAxis: Vec3;                   // half-height vector (uAxis × vAxis = normal)
}

export interface PointEmitter extends EmitterBase {
  readonly kind: 'point';
  readonly position: Vec3;
  readonly distance?: number;             // attenuation falloff distance
  readonly decay?: number;                // 0 = no decay, 2 = physical inverse-square
}

export interface SpotEmitter extends EmitterBase {
  readonly kind: 'spot';
  readonly position: Vec3;
  readonly direction: Vec3;
  readonly angle: number;                 // half-cone angle in radians
  readonly penumbra?: number;             // 0–1; 0 = hard edge, 1 = full penumbra
  readonly distance?: number;
  readonly decay?: number;
}

export interface MeshAreaEmitter extends EmitterBase {
  readonly kind: 'mesh-area';
  /** References a `MeshPrimitive` in the scene by id. The emitter samples
   *  surface points on that mesh; the mesh's material's emissive contributes
   *  to the radiance. Used for textured panel cells (e.g., stained-glass
   *  cells where each cell contributes its baked emissive). */
  readonly meshId: SceneNodeId;
}
