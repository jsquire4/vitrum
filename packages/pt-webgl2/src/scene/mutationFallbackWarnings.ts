// mutationFallbackWarnings — pure classification + EngineWarning builders for the
// pt-webgl2 scene-mutation fallback path (extracted from index.ts, T3-D / D11-1).
//
// When a targeted mutation (addPrimitive/removePrimitive/updatePrimitive) cannot be
// applied as a native texture patch, the engine falls back to a broader
// texture/BVH/scene repack and emits a structured `EngineWarning` naming the reason.
// This module owns the *classification* of a patch's field set into a fallback
// reason + the construction of the warning object; the engine retains the
// once-only Set and the emit. BEHAVIOR-PRESERVING: the returned EngineWarning
// objects (code/message/details) are byte-identical to the pre-extraction inline
// construction; the once-key semantics are unchanged (the caller owns the Set).

import type { EngineWarning, ScenePrimitive } from '@vitrum/core';

export function primitivePatchFields(patch: Partial<ScenePrimitive>): string[] {
  return Object.keys(patch)
    .filter((key) => key !== 'id' && key !== 'kind')
    .sort();
}

export function materialPatchFields(patch: Partial<ScenePrimitive>): string[] {
  if (patch.material == null) return [];
  return Object.keys(patch.material as unknown as Record<string, unknown>).sort();
}

export const GEOMETRY_REBUILD_PATCH_FIELDS = new Set([
  'transform',
  'positions',
  'normals',
  'indices',
  'uvs',
  'uv1',
  'uvSets',
  'tangents',
  'colors',
  'instances',
  'shape',
  'params',
  'morphTargets',
  'morphTargetNormals',
  'morphTargetTangents',
  'morphTargetUvs',
  'morphTargetUv1s',
  'morphTargetUvSets',
  'morphWeights',
]);

export const DISPLACEMENT_GEOMETRY_MATERIAL_FIELDS = new Set([
  'displacementMap',
  'displacementScale',
  'displacementBias',
]);

export const FULL_SCENE_REPACK_PATCH_FIELDS = new Set([
  'shape',
  'params',
]);

export const ANIMATION_REBUILD_PATCH_FIELDS = new Set([
  'bones',
  'boneInverses',
  'skinIndices',
  'skinWeights',
  'skinInfluencesPerVertex',
  'morphTargets',
  'morphTargetNormals',
  'morphTargetTangents',
  'morphTargetUvs',
  'morphTargetUv1s',
  'morphTargetUvSets',
  'morphWeights',
]);

export function primitiveFallbackReason(fields: readonly string[]): {
  readonly fallbackReason: string;
  readonly nativePatchMissing: string;
  readonly animationFields?: readonly string[];
  readonly fullUploadFields?: readonly string[];
} | null {
  const animationFields = fields.filter((field) => ANIMATION_REBUILD_PATCH_FIELDS.has(field));
  if (animationFields.length > 0) {
    return {
      fallbackReason: 'animation-geometry-rebuild',
      nativePatchMissing: 'targeted-skinned-or-morph-geometry-update',
      animationFields,
    };
  }
  if (fields.includes('colors')) {
    return {
      fallbackReason: 'geometry-material-texture-rebuild',
      nativePatchMissing: 'targeted-vertex-color-attribute-material-flag-update',
    };
  }
  if (fields.some((field) => GEOMETRY_REBUILD_PATCH_FIELDS.has(field))) {
    const fullUploadFields = fields.filter((field) => FULL_SCENE_REPACK_PATCH_FIELDS.has(field));
    if (fullUploadFields.length > 0) {
      return {
        fallbackReason: 'primitive-scene-texture-repack',
        nativePatchMissing: 'targeted-primitive-layout-or-analytic-update',
        fullUploadFields,
      };
    }
    return {
      fallbackReason: 'geometry-bvh-texture-rebuild',
      nativePatchMissing: 'targeted-primitive-geometry-splice',
    };
  }
  return null;
}

/**
 * Build the structured `EngineWarning` for an addPrimitive/removePrimitive
 * fallback rebuild. Byte-identical to the pre-extraction inline construction in
 * `#warnPrimitiveListFallback`.
 */
export function buildPrimitiveListFallbackWarning(
  method: 'addPrimitive' | 'removePrimitive',
  primitiveId: string,
  fallbackReason: 'primitive-list-texture-refresh' | 'primitive-list-scene-repack',
  textureRefreshMode?: string,
): EngineWarning {
  const refreshed = fallbackReason === 'primitive-list-texture-refresh';
  return {
    code: 'pt-webgl2.primitive-list-fallback-rebuild',
    backend: 'pt-webgl2',
    phase: 'mutation',
    method,
    message:
      `[vitrum/pt-webgl2] ${method}("${primitiveId}") rebuilds the backend ` +
      (refreshed ? 'geometry/material/atlas/BVH texture pack' : 'scene-texture/BVH pack') +
      '. This is supported, but it is not a targeted ' +
      'native geometry patch.',
    details: {
      primitiveId,
      operation: method,
      fallbackReason,
      nativePatchMissing: 'targeted-primitive-list-splice',
      ...(textureRefreshMode !== undefined ? { textureRefreshMode } : {}),
    },
  };
}

/**
 * Classify an updatePrimitive patch into a stable once-key + the structured
 * `EngineWarning` for a mutation fallback rebuild. Byte-identical to the
 * pre-extraction inline construction in `#warnPrimitiveMutationFallback`.
 *
 * @param materialTextureFields the material-texture-map field subset (resolved by
 *   `materialTextureMapPatchFields` in the mutateSceneTextures module — passed in
 *   to keep this module free of the GL-texture dependency).
 */
export function buildPrimitiveMutationFallbackWarning(
  id: string,
  patch: Partial<ScenePrimitive>,
  materialTextureFields: readonly string[],
  mutationFallback?: {
    readonly fallbackReason: string;
    readonly nativePatchMissing: string;
    readonly textureRefreshMode?: string;
  },
): { readonly key: string; readonly warning: EngineWarning } {
  const fields = primitivePatchFields(patch);
  const materialFields = materialPatchFields(patch);
  const materialDisplacementFields = materialFields.filter((field) =>
    DISPLACEMENT_GEOMETRY_MATERIAL_FIELDS.has(field),
  );
  const patchFallback = primitiveFallbackReason(fields);
  const signature = fields.length > 0 ? fields.join(',') : '<none>';
  const key = `updatePrimitive:${id}:${signature}`;
  const details: Record<string, unknown> = { primitiveId: id, fields };
  if (materialFields.length > 0) details.materialFields = materialFields;
  if (materialTextureFields.length > 0) {
    details.materialTextureFields = materialTextureFields;
    details.fallbackReason = 'texture-map-material-patch';
    details.nativePatchMissing = 'targeted-material-atlas-texture-update';
  } else if (materialDisplacementFields.length > 0) {
    details.displacementFields = materialDisplacementFields;
    details.fallbackReason = 'displacement-geometry-repack';
    details.nativePatchMissing = 'targeted-displacement-geometry-update';
  } else if (
    mutationFallback != null &&
    (patchFallback == null || patchFallback.fallbackReason === 'geometry-bvh-texture-rebuild')
  ) {
    details.fallbackReason = mutationFallback.fallbackReason;
    details.nativePatchMissing = mutationFallback.nativePatchMissing;
    if (mutationFallback.textureRefreshMode !== undefined) {
      details.textureRefreshMode = mutationFallback.textureRefreshMode;
    }
    if (patchFallback?.animationFields !== undefined) {
      details.animationFields = patchFallback.animationFields;
    }
    if (patchFallback?.fullUploadFields !== undefined) {
      details.fullUploadFields = patchFallback.fullUploadFields;
    }
  } else if (patchFallback != null) {
    details.fallbackReason = patchFallback.fallbackReason;
    details.nativePatchMissing = patchFallback.nativePatchMissing;
    if (patchFallback.animationFields !== undefined) {
      details.animationFields = patchFallback.animationFields;
    }
    if (patchFallback.fullUploadFields !== undefined) {
      details.fullUploadFields = patchFallback.fullUploadFields;
    }
  }
  return {
    key,
    warning: {
      code: 'pt-webgl2.primitive-mutation-fallback-rebuild',
      backend: 'pt-webgl2',
      phase: 'mutation',
      method: 'updatePrimitive',
      message:
        `[vitrum/pt-webgl2] updatePrimitive("${id}") fields [${signature}] ` +
        'are supported by rebuilding the backend scene-texture/BVH pack rather ' +
        'than by a targeted native patch.',
      details,
    },
  };
}
