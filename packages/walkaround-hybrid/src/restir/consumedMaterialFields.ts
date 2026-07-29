import type { MaterialSpec } from '@vitrum/core';

/**
 * Allowlist of `MaterialSpec` fields that the walkaround-hybrid package
 * actually reads during scene ingestion (BVH packing, material atlas upload,
 * emitter list building, DDGI material upload). Every field NOT in this set is
 * not consumed by the renderer; callers get a structured warning the first
 * time an engine instance sees those fields.
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
 *  shadingModel           packingHelpers.ts – packBVHRoughMetalFromCore
 *                           encodes unlit as a material flag; shade.wgsl
 *                           emits base color directly for unlit surfaces
 *  alphaMode              packingHelpers.ts – packBVHRoughMetalFromCore
 *                           encodes scalar mask / fully-transparent blend
 *                           discard into bvh_material bit 2; sceneTraversal
 *                           skips those triangles for first-hit traversal
 *  alphaCutoff            same scalar cutout path (`mask` cutoff default 0.5)
 *  opacity                scalar mask coverage plus ordered front-to-back OIT
 *                           for primary blend layers; secondary transport uses
 *                           independently seeded stochastic coverage.
 *  doubleSided            dedicated side metadata filters opaque back faces in
 *                           hybrid, DDGI, and RC traversal; TLAS orientation is
 *                           corrected for mirrored instance transforms while
 *                           transmissive exits remain admissible.
 *  emissive               packingHelpers.ts – packBVHEmissiveLeFromCore via
 *                           materialSpecEmissiveLe
 *  emissiveIntensity      same as emissive
 *  emissiveMap            materialAtlas.wgsl samples accepted CPU-readable maps
 *                           for camera-visible glow; ReSTIR-DI splits every
 *                           radiating mapped triangle into exact constant-texel
 *                           sub-triangles shared by the CDF/alias/light tree and
 *                           p-hat. GPU-only/unreadable maps or missing UV sets
 *                           fail synchronously before scene publication.
 *  transmission           packingHelpers.ts – packBVHIndexWFromCore (trans4
 *                           lane) + resolveRoughMetal (glass-roughness branch)
 *  transmissionMap        materialAtlas.wgsl samples atlas-backed linear R-channel
 *                           maps for shade/RIS/GI glass gating; emitter/GI
 *                           payloads still use scalar packed lanes.
 *  attenuationColor       shared-bvh materialSpecTriColor / Beer-Lambert lane
 *                           (bvh_beer buffer)
 *  attenuationDistance    same Beer-Lambert path
 *  thickness              same Beer-Lambert path
 *  thicknessMap           materialAtlas.wgsl samples atlas-backed linear KHR
 *                           volume maps from G and exponentiates the scalar
 *                           Beer-Lambert tint by thicknessTexture.g.
 *  ior                    shared-bvh coreMaterialToMaterialEntry →
 *                           ddgi/probeUpdateMaterials.ts (DDGI material upload)
 *  extensions             stained-glass validateSurfaceTextureId
 *                           (extensions.surfaceTextureId → texType3 lane in
 *                           bvhIndex.w) +
 *                           materialSpecSkipEmitter (extensions.skipEmitter)
 *  baseColorMap           pipeline/materialTextureAtlas.ts packs CPU pixel or
 *                           nominal GPU TextureRefs into a full-mip RGBA32F array;
 *                           shade.wgsl applies authored wrap/filter/transform and
 *                           a bounded projected-triangle footprint LOD.
 *  normalMap              same material atlas path; shade.wgsl derives a
 *                           per-triangle tangent frame from positions + uv0/uv1
 *                           and perturbs the camera-visible smooth normal.
 *  normalScale            stored in normal-map atlas metadata and applied to
 *                           tangent-space xy before normal reconstruction.
 *  bumpMap                same material atlas path; shade.wgsl samples a
 *                           atlas-backed linear height field and finite-differences
 *                           it into a camera-visible normal perturbation after
 *                           normalMap application.
 *  bumpScale              stored in bump-map atlas metadata and applied to the
 *                           finite-difference height gradient.
 *  roughnessMap           same material atlas + metadata path; shade.wgsl
 *                           samples the glTF G channel and multiplies the
 *                           scalar roughness factor for visible BRDF terms.
 *  metallicMap            same material atlas + metadata path; shade.wgsl
 *                           samples the glTF B channel and multiplies the
 *                           scalar metallic factor for visible BRDF terms.
 *  aoMap                  same material atlas + metadata path; shade.wgsl
 *                           samples the glTF R channel and multiplies it into
 *                           the runtime GTAO factor for diffuse lighting.
 *  aoMapIntensity         packBVHRoughMetalFromCore stores the glTF
 *                           occlusion strength in material-word bits 3-7.
 *  alphaMap               materialAtlas.wgsl samples readable alpha maps in
 *                           primary traversal, RIS, and GI bounce casts; mask
 *                           uses opacity * baseColorMap.a * alphaMap.r <
 *                           alphaCutoff. Fractional blend camera composition is
 *                           handled by ordered transparent OIT; secondary
 *                           ReSTIR/GI casts use independently seeded stochastic
 *                           coverage and alpha-aware shadow transmittance.
 *  lightMap               materialAtlas.wgsl samples atlas-backed linear light maps
 *                           as receiver-local irradiance, converts it to
 *                           outgoing Lambertian radiance exactly once, and
 *                           carries that source through ReSTIR-GI, DDGI, and RC.
 *  lightMapIntensity      stored in light-map atlas metadata and multiplied into
 *                           the receiver-local baked irradiance term.
 *  envMapIntensity        stored in material atlas metadata and applied to
 *                           shade-owned HDRI ReSTIR-DI environment lighting,
 *                           including canonical p-hat evaluation for temporal
 *                           and spatial DI reuse.
 *  specularColor          stored in material atlas metadata and applied to the
 *                           dielectric GGX F0 tint in shade-owned direct,
 *                           analytic, sun, specular-indirect, and DI/GI suffix
 *                           material paths.
 *  specularIntensity      same scalar specular metadata path.
 *  specularColorMap      atlas-backed sRGB maps multiply scalar `specularColor`
 *                           before shade-owned GGX evaluation.
 *  specularIntensityMap  atlas-backed linear maps multiply scalar
 *                           `specularIntensity` from their alpha channel.
 *  clearcoat             stored in material atlas metadata and added as a
 *                           fixed-F0 GGX top-coat lobe in shade-owned direct,
 *                           analytic, sun, specular-indirect, DI/GI suffix,
 *                           and GI receiver-target material paths.
 *  clearcoatRoughness    same scalar clearcoat metadata path.
 *  clearcoatMap          atlas-backed linear maps multiply scalar `clearcoat`
 *                           from their red channel before top-coat evaluation.
 *  clearcoatRoughnessMap atlas-backed linear maps multiply scalar
 *                           `clearcoatRoughness` from their green channel.
 *  clearcoatNormalMap    atlas-backed normal maps perturb the shade-owned
 *                           clearcoat lobe plus DI/GI suffix and receiver
 *                           material payloads through the derived-TBN atlas path.
 *  clearcoatNormalScale  metadata scale for `clearcoatNormalMap`.
 *  sheen                 stored in material atlas metadata and added as a
 *                           Charlie/Neubelt-Pettineo sheen lobe in shade-owned
 *                           direct, analytic, sun, specular-indirect, and DI/GI
 *                           suffix/receiver-target material paths.
 *  sheenColor            same scalar sheen metadata path.
 *  sheenRoughness        same scalar sheen metadata path.
 *  sheenColorMap         atlas-backed sRGB maps multiply scalar `sheenColor`.
 *  sheenRoughnessMap     atlas-backed linear maps multiply scalar
 *                           `sheenRoughness` from their alpha channel.
 *  anisotropy            stored in material atlas metadata and switches
 *                           shade-owned GGX evals to an anisotropic branch.
 *  anisotropyRotation    metadata rotation for the anisotropic GGX frame.
 *  anisotropyMap         atlas-backed linear KHR anisotropy maps multiply
 *                           strength from B and direction from RG.
 *  iridescence           stored in material atlas metadata and modifies
 *                           shade-owned GGX F0 with a thin-film approximation.
 *  iridescenceIor        metadata thin-film IOR for the F0 approximation.
 *  iridescenceThicknessRange metadata min/max thickness in nanometres.
 *  iridescenceMap        atlas-backed linear KHR iridescence maps multiply
 *                           scalar iridescence from the red channel.
 *  iridescenceThicknessMap atlas-backed linear thickness maps select thickness
 *                           from the green channel.
 *  spectralAttenuation    preintegrated 32-sample CIE/D65 attenuation, shared
 *                           by shade, OIT, ReSTIR GI/DI, DDGI, and RC transport.
 *  dispersionAbbeNumber   Cauchy/Abbe RGB IOR reduction drives transmissive
 *                           refraction and generic refractive caustics.
 *  thinFilmStack          full CPU TMM spectral integration pre-binned by
 *                           incidence angle and consumed in all material paths.
 *
 * Everything else — TextureRef maps other than baseColorMap / normalMap /
 * roughnessMap / metallicMap / aoMap / alphaMap / emissiveMap /
 * transmissionMap / thicknessMap / lightMap / specular maps / clearcoat maps /
 * sheen maps / anisotropyMap / iridescence maps / bumpMap / displacementMap,
 * scalar displacement controls other than displacementScale /
 * displacementBias / displacementSubdivisions,
 * and unlisted future maps/extension families
 * — is rejected by the
 * warning/truthfulness surface rather than silently rendered as native.
 */

/**
 * Keyed doc record: each consumed `MaterialSpec` field → a one-line note on
 * where/how walkaround-hybrid reads it. This is the SINGLE SOURCE of the
 * allowlist — {@link CONSUMED_MATERIAL_FIELDS} is derived from its keys, so a
 * field can never be in the allowlist without a doc entry (or vice versa). The
 * full narrative sites live in this module's header JSDoc; these entries are the
 * machine-checkable index (guarded by `consumedMaterialFieldDocs.test.ts`, which
 * asserts key parity with the Set — D6-5).
 */
export const CONSUMED_MATERIAL_FIELD_DOCS = {
  baseColor: 'packingHelpers packBVHIndexWFromCore via materialSpecTriColor',
  roughness: 'packingHelpers packBVHRoughMetalFromCore via resolveRoughMetal',
  metallic: 'packBVHIndexWFromCore isMetal flag + packBVHRoughMetalFromCore',
  shadingModel: 'packBVHRoughMetalFromCore encodes unlit as a material flag',
  emissive: 'packingHelpers packBVHEmissiveLeFromCore via materialSpecEmissiveLe',
  emissiveIntensity: 'same as emissive',
  emissiveMap: 'materialAtlas.wgsl camera-visible emitter glow + exact texel sub-triangles',
  lightMap: 'receiver-local irradiance → albedo/π outgoing source in ReSTIR-GI/DDGI/RC',
  lightMapIntensity: 'receiver-local light-map irradiance multiplier',
  envMapIntensity: 'material atlas metadata → shade HDRI ReSTIR-DI env lighting',
  spectralAttenuation: '32-sample CIE/D65 attenuation → shade/OIT/ReSTIR/DDGI/RC',
  dispersionAbbeNumber: 'Cauchy/Abbe RGB IOR → transmissive transport and caustics',
  thinFilmStack: 'spectral TMM preintegration → angle-binned forward/reverse optical response',
  specularColor: 'material atlas metadata → dielectric GGX F0 tint',
  specularIntensity: 'scalar specular metadata path',
  specularColorMap: 'atlas-backed sRGB maps multiply scalar specularColor',
  specularIntensityMap: 'atlas-backed linear maps multiply specularIntensity from alpha',
  clearcoat: 'material atlas metadata → fixed-F0 GGX top-coat lobe',
  clearcoatRoughness: 'scalar clearcoat metadata path',
  clearcoatMap: 'atlas-backed linear maps multiply scalar clearcoat from red',
  clearcoatRoughnessMap: 'atlas-backed linear maps multiply clearcoatRoughness from green',
  clearcoatNormalMap: 'atlas-backed normal maps perturb the clearcoat lobe (derived TBN)',
  clearcoatNormalScale: 'metadata scale for clearcoatNormalMap',
  sheen: 'material atlas metadata → Charlie sheen lobe',
  sheenColor: 'scalar sheen metadata path',
  sheenRoughness: 'scalar sheen metadata path',
  sheenColorMap: 'atlas-backed sRGB maps multiply scalar sheenColor',
  sheenRoughnessMap: 'atlas-backed linear maps multiply sheenRoughness from alpha',
  anisotropy: 'material atlas metadata → anisotropic GGX branch',
  anisotropyRotation: 'metadata rotation for the anisotropic GGX frame',
  anisotropyMap: 'atlas-backed linear KHR anisotropy maps (strength B, direction RG)',
  iridescence: 'material atlas metadata → thin-film F0 approximation',
  iridescenceIor: 'metadata thin-film IOR',
  iridescenceThicknessRange: 'metadata min/max thickness (nm)',
  iridescenceMap: 'atlas-backed linear KHR iridescence maps multiply from red',
  iridescenceThicknessMap: 'atlas-backed linear thickness maps select from green',
  alphaMode: 'packBVHRoughMetalFromCore encodes mask/blend into bvh_material bit 2',
  alphaCutoff: 'scalar cutout path (mask cutoff default 0.5)',
  opacity: 'scalar mask coverage + ordered primary OIT + seeded secondary coverage',
  doubleSided: 'dedicated side metadata → parity-correct hybrid/DDGI/RC traversal filtering',
  transmission: 'packBVHIndexWFromCore trans4 lane + resolveRoughMetal glass branch',
  transmissionMap: 'materialAtlas.wgsl atlas-backed linear R glass gating',
  attenuationColor: 'shared-bvh Beer-Lambert lane (bvh_beer buffer)',
  attenuationDistance: 'same Beer-Lambert path',
  thickness: 'same Beer-Lambert path',
  thicknessMap: 'materialAtlas.wgsl atlas-backed linear KHR volume maps from G',
  scatteringCoefficient: 'RGB homogeneous-medium sigma_s (scalar fallback) in camera/ReSTIR/DDGI/RC',
  scatteringCoefficientRGB: 'per-channel homogeneous-medium sigma_s in camera/ReSTIR/DDGI/RC',
  scatteringAnisotropy: 'normalized Henyey-Greenstein phase in camera/ReSTIR/DDGI/RC',
  frontLayer: 'front-face RGB transmission, roughness, and layer-local normal map transport',
  backLayer: 'back-face RGB transmission, roughness, and layer-local normal map transport',
  ior: 'shared-bvh coreMaterialToMaterialEntry → DDGI material upload',
  extensions: 'stained-glass-validated surfaceTextureId → texType3 lane; skipEmitter',
  baseColorMap: 'mipmapped RGBA32F atlas with CPU/GPU sources and compact arbitrary-texCoord affine charts',
  normalMap: 'material atlas path; shade.wgsl per-triangle TBN perturbation',
  normalScale: 'normal-map atlas metadata applied to tangent-space xy',
  bumpMap: 'atlas-backed linear height field finite-differenced into a normal perturbation',
  bumpScale: 'bump-map atlas metadata applied to the height gradient',
  displacementMap: 'shared-bvh vertex displacement (mesh microdisplacement)',
  displacementScale: 'displacement magnitude scalar',
  displacementBias: 'displacement offset scalar',
  displacementSubdivisions: 'displacement subdivision level',
  roughnessMap: 'material atlas + metadata; shade.wgsl multiplies scalar roughness (glTF G)',
  metallicMap: 'material atlas + metadata; shade.wgsl multiplies scalar metallic (glTF B)',
  aoMap: 'material atlas + metadata; shade.wgsl multiplies runtime GTAO (glTF R)',
  aoMapIntensity: 'packBVHRoughMetalFromCore stores occlusion strength in material-word bits 3-7',
  alphaMap: 'materialAtlas.wgsl alpha maps in primary/RIS/GI casts; mask + OIT',
} as const satisfies Readonly<Partial<Record<keyof MaterialSpec, string>>>;

/**
 * Compile-time-complete disposition of the canonical MaterialSpec contract.
 * Every field is represented by the renderer's ingestion/transport path above.
 * Adding a core material field therefore breaks this package's typecheck until
 * the backend deliberately consumes or rejects it.
 */
export const MATERIAL_FIELD_DISPOSITIONS = {
  ...CONSUMED_MATERIAL_FIELD_DOCS,
} as const satisfies Readonly<Record<keyof MaterialSpec, string>>;

/** The set of `MaterialSpec` keys actually consumed by walkaround-hybrid.
 *  Derived from {@link CONSUMED_MATERIAL_FIELD_DOCS} keys so the allowlist and
 *  the doc index cannot drift (D6-5). */
export const CONSUMED_MATERIAL_FIELDS: ReadonlySet<string> =
  new Set<string>(Object.keys(CONSUMED_MATERIAL_FIELD_DOCS));

/** Numerical zero used by the delta-interface transport paths. Roughness is
 * packed to u8 on GPU, so every positive representable shader value is above
 * this epsilon; it is not an artistic smooth-glass threshold. */
export const MAX_DELTA_DIELECTRIC_ROUGHNESS_EPSILON = 1e-4;

export interface UnconsumedMaterialPrimitiveFields {
  readonly primitiveId: string;
  readonly fields: readonly string[];
}

function materialBearingPrimitiveKind(kind: string): boolean {
  return (
    kind === 'mesh' ||
    kind === 'skinned-mesh' ||
    kind === 'instanced-mesh' ||
    kind === 'analytic'
  );
}

/**
 * Scan one material and return the fields that are present but NOT in
 * {@link CONSUMED_MATERIAL_FIELDS}. The check is per-field-key; a field counts
 * as "present" when it is defined and non-null.
 */
export function collectUnconsumedMaterialFieldsForMaterial(
  material: Record<string, unknown> | undefined,
): string[] {
  if (!material) return [];
  const supplied = new Set<string>();
  for (const key of Object.keys(material)) {
    if (CONSUMED_MATERIAL_FIELDS.has(key)) continue;
    const val = material[key];
    if (val !== undefined && val !== null) {
      supplied.add(key);
    }
  }
  return Array.from(supplied).sort();
}

export type UnconsumedMaterialFieldCategory =
  | 'geometry'
  | 'spectral'
  | 'volume'
  | 'layered'
  | 'unknown';

/** Group unconsumed material keys into stable semantic buckets for diagnostics. */
export function categorizeUnconsumedMaterialFields(
  fields: readonly string[],
): Partial<Record<UnconsumedMaterialFieldCategory, readonly string[]>> {
  const grouped: Record<UnconsumedMaterialFieldCategory, string[]> = {
    geometry: [],
    spectral: [],
    volume: [],
    layered: [],
    unknown: [],
  };
  for (const field of fields) {
    // Every currently standardized MaterialSpec field is consumed by this
    // backend. Any residual key is therefore an extension/future field whose
    // semantics are not known here; report it honestly as unknown rather than
    // maintaining an empty uploaded-but-unread classification table.
    grouped.unknown.push(field);
  }

  const out: Partial<Record<UnconsumedMaterialFieldCategory, readonly string[]>> = {};
  for (const category of ['geometry', 'spectral', 'volume', 'layered', 'unknown'] as const) {
    if (grouped[category].length > 0) {
      out[category] = grouped[category].sort();
    }
  }
  return out;
}

/**
 * Scan every material-bearing primitive's material in `scene`
 * and return the union of fields that are present in the scene but NOT in
 * {@link CONSUMED_MATERIAL_FIELDS}. The check is per-field-key; a field counts
 * as "present" when it is defined and non-null on at least one material.
 *
 * Returns an empty array when the scene has no unconsumed material fields.
 */
export function collectUnconsumedMaterialFields(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
): string[] {
  const supplied = new Set<string>();
  for (const entry of collectUnconsumedMaterialPrimitiveFields(primitives)) {
    for (const key of entry.fields) supplied.add(key);
  }
  return Array.from(supplied).sort();
}

/**
 * Return per-primitive authorship for unconsumed material fields. This keeps the
 * aggregate warning compact while still giving hosts a stable pointer to the
 * exact scene primitive that triggered a parked unsupported material family.
 */
export function collectUnconsumedMaterialPrimitiveFields(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
): UnconsumedMaterialPrimitiveFields[] {
  const out: UnconsumedMaterialPrimitiveFields[] = [];
  for (const prim of primitives) {
    if (!materialBearingPrimitiveKind(prim.kind)) {
      continue;
    }
    const fields = collectUnconsumedMaterialFieldsForMaterial(prim.material);
    if (fields.length > 0) {
      out.push({
        primitiveId: prim.id ?? '(unnamed)',
        fields,
      });
    }
  }
  return out.sort((a, b) => a.primitiveId.localeCompare(b.primitiveId));
}

/**
 * Strict scene/update boundary for material authorship. Core validation owns
 * the canonical MaterialSpec vocabulary; this guard is deliberately retained
 * as a second line of defence for untyped callers and future core additions.
 * No authored field may degrade to a warning-and-ignore path.
 */
export function assertNoUnconsumedMaterialFields(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
  method: 'setScene' | 'updatePrimitive',
): void {
  const unconsumed = collectUnconsumedMaterialPrimitiveFields(primitives);
  if (unconsumed.length === 0) return;
  const details = unconsumed
    .map(({ primitiveId, fields }) => `${primitiveId} [${fields.join(', ')}]`)
    .join('; ');
  throw new TypeError(
    `[vitrum/walkaround-hybrid] ${method}: material fields are not represented ` +
    `by this backend and cannot be ignored. Unsupported primitive fields: ${details}.`,
  );
}
