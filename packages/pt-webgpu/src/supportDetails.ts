import type { MaterialSpec } from '@vitrum/core';
import {
  PT_WEBGPU_FULL_SUPPORT_MANIFEST,
  PT_WEBGPU_LITE_SUPPORT_MANIFEST,
} from './supportManifest.js';
import type { PtWebgpuTraceTier } from './traceTier.js';

function materialFieldsWithMode(
  materials: typeof PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials,
  mode: 'unsupported',
): (keyof MaterialSpec)[] {
  return (Object.keys(materials) as (keyof MaterialSpec)[])
    .filter((field) => materials[field] === mode);
}

// `extensions` is the contract-sanctioned host-discretionary escape hatch and
// is deliberately ignored without warning even though no shader consumes it.
export const UNSUPPORTED_MATERIAL_FIELDS = Object.freeze(
  materialFieldsWithMode(
    PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials,
    'unsupported',
  ).filter((field) => field !== 'extensions'),
);

export const PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS = Object.freeze(
  (Object.keys(PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials) as (keyof MaterialSpec)[])
    .filter((field) =>
      PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials[field] === 'unsupported' &&
      PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials[field] !== 'unsupported'),
);

export const PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS = Object.freeze([
  ...new Set([
    ...UNSUPPORTED_MATERIAL_FIELDS,
    ...PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS,
  ]),
]);

export const PT_WEBGPU_FULL_MATERIALS =
  PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials;
export const PT_WEBGPU_LITE_MATERIALS =
  PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials;

export function collectUnsupportedLayerNormalFields(
  fields: Set<string>,
  prefix: 'frontLayer' | 'backLayer',
  layer: MaterialSpec['frontLayer']   | undefined,
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
  return Array.from(fields).sort();
}
