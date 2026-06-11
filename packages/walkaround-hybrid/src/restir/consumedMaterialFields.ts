/**
 * Allowlist of `MaterialSpec` fields that the walkaround-hybrid package
 * actually reads during scene ingestion (BVH packing, emitter list building,
 * DDGI material upload). Every field NOT in this set is silently dropped —
 * callers get a `console.warn` on the first `setScene` per engine instance
 * that surfaces those fields.
 *
 * Derived by reading the actual ingestion code (2026-06-10):
 *
 *  Field                  Site
 *  ---------------------  -----------------------------------------------------
 *  baseColor              packingHelpers.ts – packBVHIndexWFromCore via
 *                           materialSpecTriColor (also fallback in
 *                           classifyTriangleCoreEmitter for the transmissive
 *                           secondary-emitter branch)
 *  roughness              packingHelpers.ts – packBVHRoughMetalFromCore via
 *                           resolveRoughMetal
 *  metallic               packingHelpers.ts – packBVHIndexWFromCore (isMetal
 *                           flag) + packBVHRoughMetalFromCore
 *  emissive               packingHelpers.ts – packBVHEmissiveLeFromCore via
 *                           materialSpecEmissiveLe
 *  emissiveIntensity      same as emissive
 *  transmission           packingHelpers.ts – packBVHIndexWFromCore (trans4
 *                           lane) + resolveRoughMetal (glass-roughness branch)
 *  attenuationColor       shared-bvh materialSpecTriColor / Beer-Lambert lane
 *                           (bvh_beer buffer)
 *  attenuationDistance    same Beer-Lambert path
 *  thickness              same Beer-Lambert path
 *  ior                    shared-bvh coreMaterialToMaterialEntry →
 *                           ddgi/probeUpdateMaterials.ts (DDGI material upload)
 *  extensions             materialSpecSurfaceTextureId (extensions.surfaceTextureId
 *                           → texType3 lane in bvhIndex.w) +
 *                           materialSpecSkipEmitter (extensions.skipEmitter)
 *
 * Everything else — ALL TextureRef maps, Disney BSDF scalars, spectral curves,
 * volume scattering, thin-film stacks, layered BSDF, anisotropy, alpha/coverage,
 * and specular extension scalars — is IGNORED.
 */

/** The set of `MaterialSpec` keys actually consumed by walkaround-hybrid. */
export const CONSUMED_MATERIAL_FIELDS: ReadonlySet<string> = new Set<string>([
  'baseColor',
  'roughness',
  'metallic',
  'emissive',
  'emissiveIntensity',
  'transmission',
  'attenuationColor',
  'attenuationDistance',
  'thickness',
  'ior',
  'extensions',
]);

/**
 * Scan every mesh/skinned-mesh/instanced-mesh primitive's material in `scene`
 * and return the union of fields that are present in the scene but NOT in
 * {@link CONSUMED_MATERIAL_FIELDS}. The check is per-field-key; a field counts
 * as "present" when it is defined and non-null on at least one material.
 *
 * Returns an empty array when the scene has no unconsumed material fields.
 */
export function collectUnconsumedMaterialFields(
  primitives: ReadonlyArray<{
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
): string[] {
  const supplied = new Set<string>();
  for (const prim of primitives) {
    if (
      prim.kind !== 'mesh' &&
      prim.kind !== 'skinned-mesh' &&
      prim.kind !== 'instanced-mesh'
    ) {
      continue;
    }
    const mat = prim.material;
    if (!mat) continue;
    for (const key of Object.keys(mat)) {
      if (CONSUMED_MATERIAL_FIELDS.has(key)) continue;
      const val = (mat)[key];
      if (val !== undefined && val !== null) {
        supplied.add(key);
      }
    }
  }
  return Array.from(supplied).sort();
}
