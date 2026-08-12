// Scale-derived defaults and backend selection for createEngine().

import type { MaterialSpec, Scene, Vec3 } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS } from '@vitrum/core';

export type EnginePreference = 'realtime' | 'quality' | 'quality-webgpu' | 'auto';
export type EngineBackendId = 'walkaround-hybrid' | 'pt-webgpu' | 'pt-webgl2';

export const DEFAULT_PRIMARY_LIGHT_DIR: Vec3 = Object.freeze([0.3, -0.7, 0.6]);
export const DEFAULT_PRIMARY_LIGHT_INTENSITY = 1.0;
export const DEFAULT_SKY_TINT: Vec3 = Object.freeze([0.5, 0.7, 1.0]);
export const DEFAULT_SKY_IRRADIANCE = 0.3;

export interface ScaleDefaults {
  readonly cameraMoveResetThresholdSq: number;
  readonly temporalAccumAlpha: number;
  readonly emitterDist2Floor: number;
  readonly triIntersectEpsilon: number;
}

export interface SceneMaterialBackendRecommendation {
  readonly backend: EngineBackendId;
  readonly fields: readonly string[];
}

export function deriveScaleDefaults(D: number): ScaleDefaults {
  return {
    cameraMoveResetThresholdSq: (D * 1e-3) ** 2,
    temporalAccumAlpha: 0.01,
    emitterDist2Floor: (D * 1e-4) ** 2,
    triIntersectEpsilon: D * 1e-6,
  };
}

export function pickBackend(
  prefer: EnginePreference,
  hasWebGPU: boolean,
  triangleCount: number,
  needsTlas = false,
  gltfRecommendedBackend?: EngineBackendId,
  materialRecommendedBackend?: EngineBackendId,
): EngineBackendId {
  // Triangle count is retained for call-site compatibility. `prefer:'auto'` no
  // longer uses a 500k-triangle walkaround cutoff; createEngine tries the
  // progressive viewer first and uses this picker only as the single-engine
  // fallback (and for explicit prefer / glTF / material routing).
  void triangleCount;
  if (prefer === 'auto' && gltfRecommendedBackend !== undefined) {
    return resolveGltfRecommendedBackend(gltfRecommendedBackend, hasWebGPU);
  }
  if (prefer === 'auto' && materialRecommendedBackend !== undefined) {
    return resolveGltfRecommendedBackend(materialRecommendedBackend, hasWebGPU);
  }
  if (prefer === 'quality-webgpu') return hasWebGPU ? 'pt-webgpu' : 'pt-webgl2';
  if (prefer === 'quality') {
    if (needsTlas && hasWebGPU) return 'pt-webgpu';
    return 'pt-webgl2';
  }
  if (prefer === 'realtime') {
    if (!hasWebGPU) return 'pt-webgl2';
    return 'walkaround-hybrid';
  }
  // Single-engine auto fallback: PT on WebGPU (full or lite by device), WebGL2 otherwise.
  if (hasWebGPU) {
    return 'pt-webgpu';
  }
  // WebGL-only host: merged BVH is the only pt-webgl2 path; caller should warn
  // when needsTlas is true (handled at the createEngine call site).
  return 'pt-webgl2';
}

/**
 * Whether `createEngine({ prefer:'auto' })` should try the progressive
 * walkaround→PT viewer before falling back to {@link pickBackend}.
 *
 * Progressive needs the Class-A limit union (not lite). Skip when the host
 * asked for a single engine, when walkaround cannot accept the scene, or when
 * the glTF planner already named WebGL2.
 */
export function shouldAttemptProgressiveViewer(
  prefer: EnginePreference,
  hasWebGPU: boolean,
  gltfRecommendedBackend?: EngineBackendId,
  materialRecommendedBackend?: EngineBackendId,
): boolean {
  if (prefer !== 'auto') return false;
  if (!hasWebGPU) return false;
  if (materialRecommendedBackend != null) return false;
  if (gltfRecommendedBackend === 'pt-webgl2') return false;
  return true;
}

export function recommendBackendForSceneMaterials(
  scene: Scene,
  hasWebGPU: boolean,
): SceneMaterialBackendRecommendation | null {
  const fields = collectWalkaroundUnsupportedMaterialFieldsWithPtSupport(scene);
  if (fields.length === 0) return null;
  return {
    backend: hasWebGPU ? 'pt-webgpu' : 'pt-webgl2',
    fields,
  };
}

function resolveGltfRecommendedBackend(
  backend: EngineBackendId,
  hasWebGPU: boolean,
): EngineBackendId {
  if (backend === 'pt-webgl2') return 'pt-webgl2';
  if (hasWebGPU) return backend;
  return 'pt-webgl2';
}

function collectWalkaroundUnsupportedMaterialFieldsWithPtSupport(scene: Scene): readonly string[] {
  const out = new Set<string>();
  const walkaroundRows = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.materials ?? {};
  const ptWebgl2Rows = BACKEND_PROMISE_LEDGER['pt-webgl2'].supportDetails.materials ?? {};
  const ptWebgpuRows = BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials ?? {};

  for (const primitive of scene.primitives) {
    const material = (primitive as { readonly material?: MaterialSpec }).material;
    if (material == null) continue;
    for (const field of MATERIAL_SPEC_FIELDS) {
      if (walkaroundRows[field] !== 'unsupported') continue;
      if (!isMaterialFieldAuthored(material, field)) continue;
      const gl2Support = ptWebgl2Rows[field];
      const gpuSupport = ptWebgpuRows[field];
      if (gl2Support !== 'unsupported' || gpuSupport !== 'unsupported') out.add(field);
    }
  }
  return [...out].sort();
}

function isMaterialFieldAuthored(
  material: MaterialSpec,
  field: (typeof MATERIAL_SPEC_FIELDS)[number],
): boolean {
  const value = material[field];
  if (value == null) return false;
  if (typeof value === 'number') return Number.isFinite(value) && value !== 0;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') return value.length > 0;
  if (ArrayBuffer.isView(value)) return value.byteLength > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value).length > 0;
  return true;
}
