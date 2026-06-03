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

function normalizeVec3(x: number, y: number, z: number, label?: string): Vec3 {
  const len = Math.sqrt(x * x + y * y + z * z);
  if (len === 0) {
    if (label != null) {
      console.warn(`@vitrum/three-bindings: light "${label}" has zero-length direction; using fallback [0,0,-1].`);
    }
    return [0, 0, -1];
  }
  return [x / len, y / len, z / len];
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
    const dx = dl.position.x - dl.target.position.x;
    const dy = dl.position.y - dl.target.position.y;
    const dz = dl.position.z - dl.target.position.z;
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
    // Column 0 = local X (half-width), column 1 = local Y (half-height).
    const hw = rl.width / 2;
    const hh = rl.height / 2;
    const uAxis: Vec3 = [me[0] * hw, me[1] * hw, me[2] * hw];
    const vAxis: Vec3 = [me[4] * hh, me[5] * hh, me[6] * hh];
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
    const position: Vec3 = [pl.position.x, pl.position.y, pl.position.z];
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
    const position: Vec3 = [sl.position.x, sl.position.y, sl.position.z];
    const dx = sl.position.x - sl.target.position.x;
    const dy = sl.position.y - sl.target.position.y;
    const dz = sl.position.z - sl.target.position.z;
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
