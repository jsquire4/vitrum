/**
 * lights.ts — THREE.js light → @vitrum/core SceneEmitter converters.
 *
 * Handles DirectionalLight, RectAreaLight, PointLight, SpotLight. AmbientLight
 * and HemisphereLight are not yet supported and emit a per-call warning via
 * `warnOnce`. All other light types throw.
 */

import type * as THREE from 'three';
import type { SceneEmitter, Vec3, DirectionalEmitter, RectAreaEmitter, PointEmitter, SpotEmitter } from '@vitrum/core';
import { colorToVec3 } from './material.js';

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function normalizeVec3(
  x: number,
  y: number,
  z: number,
  label?: string,
  fallback: Vec3 = [0, 0, -1],
): Vec3 {
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len === 0) {
    if (label != null) {
      console.warn(`@vitrum/three-bindings: light "${label}" has zero-length direction; using fallback [${fallback.join(',')}].`);
    }
    return fallback;
  }
  return [x / len, y / len, z / len];
}

function worldPositionOf(obj: THREE.Object3D): Vec3 {
  obj.updateWorldMatrix(true, false);
  const e = obj.matrixWorld.elements;
  return [e[12] ?? 0, e[13] ?? 0, e[14] ?? 0];
}

function rectHalfAxisFromWorldColumn(
  me: THREE.Matrix4['elements'],
  offset: 0 | 4,
  halfSize: number,
  label: string,
  fallback: Vec3,
): Vec3 {
  const axis = normalizeVec3(
    me[offset] ?? 0,
    me[offset + 1] ?? 0,
    me[offset + 2] ?? 0,
    label,
    fallback,
  );
  return [axis[0] * halfSize, axis[1] * halfSize, axis[2] * halfSize];
}

// ────────────────────────────────────────────────────────────────────────────
// Per-call warning dedup
// ────────────────────────────────────────────────────────────────────────────

/**
 * Warn once per call about an unsupported light type. The `warnedTypes` set
 * is passed in from `sceneFromThreeJS` so it is scoped to a single call —
 * not module-global state. This preserves the "warn once per call about each
 * type" semantic while allowing repeated calls to warn again independently.
 *
 * (M-2 fix: moved from module-level singleton to per-call parameter.)
 */
function warnOnce(warnedTypes: Set<string>, typeName: string, label: string): void {
  if (warnedTypes.has(typeName)) return;
  warnedTypes.add(typeName);
  console.warn(
    `@vitrum/three-bindings: skipping unsupported light type "${typeName}" at "${label}". Supported types are listed in the backend's EngineCapabilities.`,
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Light converters
// ────────────────────────────────────────────────────────────────────────────

/**
 * Convert a THREE.Light to a @vitrum/core SceneEmitter.
 *
 * @param light - The light to convert.
 * @param warnedTypes - Per-call set tracking which unsupported types have
 *   already emitted a warning this call.
 * @returns The converted emitter, or `null` if the light type is unsupported
 *   but skippable (AmbientLight, HemisphereLight).
 * @throws Error if the light type is entirely unrecognised.
 */
export function convertLight(
  light: THREE.Light,
  warnedTypes: Set<string>,
): SceneEmitter | null {
  const label = light.name || light.uuid;
  light.updateWorldMatrix(true, false);
  const color = colorToVec3(light.color);
  const id = light.uuid;

  if ((light as THREE.AmbientLight).isAmbientLight === true) {
    warnOnce(warnedTypes, 'AmbientLight', label);
    return null;
  }

  if ((light as THREE.HemisphereLight).isHemisphereLight === true) {
    warnOnce(warnedTypes, 'HemisphereLight', label);
    return null;
  }

  if ((light as THREE.DirectionalLight).isDirectionalLight === true) {
    const dl = light as THREE.DirectionalLight;
    const pos = worldPositionOf(dl);
    const target = worldPositionOf(dl.target);
    const dx = pos[0] - target[0];
    const dy = pos[1] - target[1];
    const dz = pos[2] - target[2];
    const direction = normalizeVec3(dx, dy, dz, label);
    const emitter: DirectionalEmitter = {
      kind: 'directional',
      id,
      color,
      intensity: dl.intensity,
      direction,
    };
    return emitter;
  }

  if ((light as THREE.RectAreaLight).isRectAreaLight === true) {
    const rl = light as THREE.RectAreaLight;
    // RectAreaLight faces -Z in local space; derive uAxis/vAxis from world matrix.
    const me = rl.matrixWorld.elements;
    // Column 0 = local X orientation, column 1 = local Y orientation. Width and
    // height carry the physical size; object scale is not baked into the core axes.
    const hw = rl.width / 2;
    const hh = rl.height / 2;
    const uAxis = rectHalfAxisFromWorldColumn(me, 0, hw, label, [1, 0, 0]);
    const vAxis = rectHalfAxisFromWorldColumn(me, 4, hh, label, [0, 1, 0]);
    const position: Vec3 = [me[12], me[13], me[14]];
    const emitter: RectAreaEmitter = {
      kind: 'rect-area',
      id,
      color,
      intensity: rl.intensity,
      position,
      uAxis,
      vAxis,
    };
    return emitter;
  }

  if ((light as THREE.PointLight).isPointLight === true) {
    const pl = light as THREE.PointLight;
    const position = worldPositionOf(pl);
    const emitter: PointEmitter = {
      kind: 'point',
      id,
      color,
      intensity: pl.intensity,
      position,
      distance: pl.distance,
      decay: pl.decay,
    };
    return emitter;
  }

  if ((light as THREE.SpotLight).isSpotLight === true) {
    const sl = light as THREE.SpotLight;
    const position = worldPositionOf(sl);
    const target = worldPositionOf(sl.target);
    const dx = position[0] - target[0];
    const dy = position[1] - target[1];
    const dz = position[2] - target[2];
    const direction = normalizeVec3(dx, dy, dz, label);
    const emitter: SpotEmitter = {
      kind: 'spot',
      id,
      color,
      intensity: sl.intensity,
      position,
      direction,
      angle: sl.angle,
      penumbra: sl.penumbra,
      distance: sl.distance,
      decay: sl.decay,
    };
    return emitter;
  }

  throw new Error(
    `Unsupported THREE type at "${label}": ${light.constructor.name}. Supported types are listed in the backend's EngineCapabilities.`,
  );
}
