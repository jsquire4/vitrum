import type { BackendSupportMode, MaterialSpec } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS } from '@vitrum/core';
import type { PtWebgpuTraceTier } from './traceTier.js';

// CAP-01 — the remaining material fields the full tier drops, derived from the
// ledger's per-field support matrix so warning + capability rows can never drift.
// `extensions` is the contract-sanctioned host-discretionary escape hatch
// (no warning).
export const UNSUPPORTED_MATERIAL_FIELDS: readonly (keyof MaterialSpec)[] = MATERIAL_SPEC_FIELDS.filter(
  (field) =>
    BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials[field] === 'unsupported' &&
    field !== 'extensions',
);

export const PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS = [
  // The lite trace shader composes no full-tier group-3 material texture
  // bindings. These fields are therefore unrendered on lite even when the full
  // pt-webgpu tier supports them.
  'baseColorMap',
  'normalMap',
  'normalScale',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'aoMapIntensity',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'clearcoatNormalScale',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'bumpScale',
  'lightMap',
  'lightMapIntensity',
  // Lite also omits the full-tier alpha-test, per-material environment scale,
  // and anisotropic-BSDF routes.
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'envMapIntensity',
  'anisotropy',
  'anisotropyRotation',
] as const satisfies readonly (keyof MaterialSpec)[];

export const PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS = Object.freeze([
  ...new Set([
    ...UNSUPPORTED_MATERIAL_FIELDS,
    ...PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS,
  ]),
]);

export const PT_WEBGPU_LITE_MATERIALS = Object.freeze({
  ...BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials,
  ...Object.fromEntries(
    PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS.map((field) =>
      [field, 'unsupported' as BackendSupportMode],
    ),
  ),
  // Lite still packs scalar layered-lobe controls inherited from the full tier,
  // but it omits the group-3 texture bindings used by layer normal maps/scales.
  frontLayer: 'approximate' as BackendSupportMode,
  backLayer:  'approximate' as BackendSupportMode,
});

export function collectUnsupportedLayerNormalFields(
  fields: Set<string>,
  prefix: 'frontLayer' | 'backLayer',
  layer: MaterialSpec['frontLayer'] | MaterialSpec['backLayer'] | undefined,
): void {
  if (layer?.normalMap != null) fields.add(`${prefix}.normalMap`);
  if (layer?.normalScale != null) fields.add(`${prefix}.normalScale`);
}

export function collectUnsupportedMaterialFieldsForTraceTier(
  material: Partial<MaterialSpec>,
  traceTier: PtWebgpuTraceTier,
): string[] {
  const unsupportedFields = traceTier === 'lite'
    ? PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS
    : UNSUPPORTED_MATERIAL_FIELDS;
  const fields = new Set<string>();
  for (const field of unsupportedFields) {
    if (material[field] != null) fields.add(field);
  }
  if (traceTier === 'lite') {
    collectUnsupportedLayerNormalFields(fields, 'frontLayer', material.frontLayer);
    collectUnsupportedLayerNormalFields(fields, 'backLayer', material.backLayer);
  }
  return Array.from(fields).sort();
}
