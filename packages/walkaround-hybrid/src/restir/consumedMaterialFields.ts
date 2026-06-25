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
 *  opacity                same scalar cutout path; fractional `blend` remains
 *                           approximate and is diagnosed by HybridEngine
 *  emissive               packingHelpers.ts – packBVHEmissiveLeFromCore via
 *                           materialSpecEmissiveLe
 *  emissiveIntensity      same as emissive
 *  emissiveMap            materialAtlas.wgsl samples readable sRGB emissive
 *                           maps for camera-visible emitter glow; ReSTIR-DI direct
 *                           emitter lists split eligible CPU-readable maps into
 *                           exact texel-cell sub-triangles, while GI/RC/DDGI and
 *                           fallback paths still report all-path texel-PDF
 *                           approximation.
 *  transmission           packingHelpers.ts – packBVHIndexWFromCore (trans4
 *                           lane) + resolveRoughMetal (glass-roughness branch)
 *  transmissionMap        materialAtlas.wgsl samples readable linear R-channel
 *                           maps for shade/RIS/GI glass gating; emitter/GI
 *                           payloads still use scalar packed lanes.
 *  attenuationColor       shared-bvh materialSpecTriColor / Beer-Lambert lane
 *                           (bvh_beer buffer)
 *  attenuationDistance    same Beer-Lambert path
 *  thickness              same Beer-Lambert path
 *  thicknessMap           materialAtlas.wgsl samples readable linear KHR
 *                           volume maps from G and exponentiates the scalar
 *                           Beer-Lambert tint by thicknessTexture.g.
 *  ior                    shared-bvh coreMaterialToMaterialEntry →
 *                           ddgi/probeUpdateMaterials.ts (DDGI material upload)
 *  extensions             materialSpecSurfaceTextureId (extensions.surfaceTextureId
 *                           → texType3 lane in bvhIndex.w) +
 *                           materialSpecSkipEmitter (extensions.skipEmitter)
 *  baseColorMap           pipeline/materialTextureAtlas.ts packs readable raw
 *                           uv0 TextureRefs into an RGBA32F array texture;
 *                           shade.wgsl samples it with wrap + transform and
 *                           multiplies the visible scalar baseColor.
 *  normalMap              same material atlas path; shade.wgsl derives a
 *                           per-triangle tangent frame from positions + uv0/uv1
 *                           and perturbs the camera-visible smooth normal.
 *  normalScale            stored in normal-map atlas metadata and applied to
 *                           tangent-space xy before normal reconstruction.
 *  bumpMap                same material atlas path; shade.wgsl samples a
 *                           readable linear height field and finite-differences
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
 *                           handled by the transparent-OIT pass; direct-light
 *                           OIT shadows attenuate blend coverage, while
 *                           ReSTIR/GI participation remains approximate.
 *  lightMap               materialAtlas.wgsl samples readable linear light maps
 *                           as camera-visible baked outgoing radiance only.
 *  lightMapIntensity      stored in light-map atlas metadata and multiplied into
 *                           the camera-visible baked light-map term.
 *  envMapIntensity        stored in material atlas metadata and applied to
 *                           shade-owned HDRI ReSTIR-DI environment lighting,
 *                           including canonical p-hat evaluation for temporal
 *                           and spatial DI reuse.
 *  specularColor          stored in material atlas metadata and applied to the
 *                           dielectric GGX F0 tint in shade-owned direct,
 *                           analytic, sun, specular-indirect, and DI/GI suffix
 *                           material paths.
 *  specularIntensity      same scalar specular metadata path; approximate
 *                           pending rich-material GI GPU A/B promotion, not
 *                           because the receiver target ignores the lobe.
 *  specularColorMap      readable sRGB maps multiply scalar `specularColor`
 *                           before shade-owned GGX evaluation.
 *  specularIntensityMap  readable linear maps multiply scalar
 *                           `specularIntensity` from their alpha channel.
 *  clearcoat             stored in material atlas metadata and added as a
 *                           fixed-F0 GGX top-coat lobe in shade-owned direct,
 *                           analytic, sun, specular-indirect, DI/GI suffix,
 *                           and GI receiver-target material paths.
 *  clearcoatRoughness    same scalar clearcoat metadata path; approximate
 *                           pending rich-material GI GPU A/B promotion.
 *  clearcoatMap          readable linear maps multiply scalar `clearcoat`
 *                           from their red channel before top-coat evaluation.
 *  clearcoatRoughnessMap readable linear maps multiply scalar
 *                           `clearcoatRoughness` from their green channel.
 *  clearcoatNormalMap    readable normal maps perturb the shade-owned
 *                           clearcoat lobe plus DI/GI suffix and receiver
 *                           material payloads through the derived-TBN atlas path.
 *  clearcoatNormalScale  metadata scale for `clearcoatNormalMap`.
 *  sheen                 stored in material atlas metadata and added as a
 *                           Charlie/Neubelt-Pettineo sheen lobe in shade-owned
 *                           direct, analytic, sun, specular-indirect, and DI/GI
 *                           suffix/receiver-target material paths.
 *  sheenColor            same scalar sheen metadata path; approximate pending
 *                           rich-material GI GPU A/B promotion.
 *  sheenRoughness        same scalar sheen metadata path.
 *  sheenColorMap         readable sRGB maps multiply scalar `sheenColor`.
 *  sheenRoughnessMap     readable linear maps multiply scalar
 *                           `sheenRoughness` from their alpha channel.
 *  anisotropy            stored in material atlas metadata and switches
 *                           shade-owned GGX evals to an anisotropic branch.
 *  anisotropyRotation    metadata rotation for the anisotropic GGX frame.
 *  anisotropyMap         readable linear KHR anisotropy maps multiply
 *                           strength from B and direction from RG.
 *  iridescence           stored in material atlas metadata and modifies
 *                           shade-owned GGX F0 with a thin-film approximation.
 *  iridescenceIor        metadata thin-film IOR for the F0 approximation.
 *  iridescenceThicknessRange metadata min/max thickness in nanometres.
 *  iridescenceMap        readable linear KHR iridescence maps multiply
 *                           scalar iridescence from the red channel.
 *  iridescenceThicknessMap readable linear thickness maps select thickness
 *                           from the green channel.
 *  frontLayer             material atlas stores per-face transmission, roughness
 *                           override, and layer-local normal maps. Shade,
 *                           transparent-OIT, ReSTIR-DI p-hat/reuse, ReSTIR-GI
 *                           suffix/receiver targets, RC probe rays, and DDGI
 *                           probe-hit material all apply the selected face layer.
 *  backLayer              same per-face material atlas path as frontLayer.
 *
 * Everything else — TextureRef maps other than baseColorMap / normalMap /
 * roughnessMap / metallicMap / aoMap / alphaMap / emissiveMap /
 * transmissionMap / thicknessMap / lightMap / specular maps / clearcoat maps /
 * sheen maps / anisotropyMap / iridescence maps / bumpMap / displacementMap,
 * scalar displacement controls other than displacementScale /
 * displacementBias / displacementSubdivisions, spectral curves, thin-film stacks,
 * and unlisted future maps/extension families
 * — is rejected by the
 * warning/truthfulness surface rather than silently rendered as native.
 */

/** The set of `MaterialSpec` keys actually consumed by walkaround-hybrid. */
export const CONSUMED_MATERIAL_FIELDS: ReadonlySet<string> = new Set<string>([
  'baseColor',
  'roughness',
  'metallic',
  'shadingModel',
  'emissive',
  'emissiveIntensity',
  'emissiveMap',
  'lightMap',
  'lightMapIntensity',
  'envMapIntensity',
  'specularColor',
  'specularIntensity',
  'specularColorMap',
  'specularIntensityMap',
  'clearcoat',
  'clearcoatRoughness',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'clearcoatNormalScale',
  'sheen',
  'sheenColor',
  'sheenRoughness',
  'sheenColorMap',
  'sheenRoughnessMap',
  'anisotropy',
  'anisotropyRotation',
  'anisotropyMap',
  'iridescence',
  'iridescenceIor',
  'iridescenceThicknessRange',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'frontLayer',
  'frontLayer.normalMap',
  'frontLayer.normalScale',
  'backLayer',
  'backLayer.normalMap',
  'backLayer.normalScale',
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'transmission',
  'transmissionMap',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'scatteringCoefficientRGB',
  'attenuationColor',
  'attenuationDistance',
  'thickness',
  'thicknessMap',
  'ior',
  'extensions',
  'baseColorMap',
  'normalMap',
  'normalScale',
  'bumpMap',
  'bumpScale',
  'displacementMap',
  'displacementScale',
  'displacementBias',
  'displacementSubdivisions',
  'roughnessMap',
  'metallicMap',
  'aoMap',
  'aoMapIntensity',
  'alphaMap',
]);

/** Structured payload for the residual emissive-map texel-PDF warning. */
export const EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS = {
  directEmitterPdf: 'exact-texel-cell-subtriangles-when-eligible',
  fallbackDirectEmitterPdf: 'uv-local-barycentric-micro-emitter-selection',
  giSuffixEmission: 'uv-local-emissive-texel-sampled-on-hit',
  probeHitEmission: 'uv-local-emissive-texel-sampled-on-direct-probe-hit',
  residualApproximation: 'global-texel-selection-pdf',
  missing: 'global-exact-texel-alias-pdf',
  exactDirectEmitterConditions: [
    'cpu-readable-emissive-map',
    'non-mirrored-wrap',
    'non-degenerate-uvs',
    'bounded-covered-texel-cells',
    'readable-covered-texels',
  ],
  approximatePaths: [
    'ReSTIR-GI-texel-selection-pdf',
    'RC-non-direct-texel-selection-pdf',
    'DDGI-non-direct-texel-selection-pdf',
    'fallback-direct-emitter',
  ],
} as const;

/** Structured payload for the camera-visible-only light-map approximation. */
export const LIGHT_MAP_CAMERA_VISIBLE_APPROXIMATION_DETAILS = {
  lightMapUsage: 'camera-visible-baked-outgoing-radiance',
  directLighting: 'not-sampled-as-scene-light',
  giPropagation: 'not-injected-into-restir-gi-ddgi-rc',
  residualApproximation: 'first-hit-baked-light',
  missing: 'global-transport-light-map-participation',
} as const;

/** Structured payload for consumed-but-still-promoted-by-proof rich material rows. */
export const RICH_MATERIAL_GI_APPROXIMATION_DETAILS = {
  consumedBy: [
    'shade-owned-direct-light',
    'ReSTIR-DI-candidate-and-resolve-BRDF',
    'ReSTIR-GI-suffix-and-receiver-lobe-targets',
    'DDGI/RC-probe-hit-compact-BRDF',
  ],
  approximation: 'compact-rich-material-lobe-response',
  missing: 'material-furnace-reference-ab-and-full-rich-material-gi-promotion',
  proofTail: [
    'GPU-material-furnace',
    'rich-material-GI-A/B',
    'reference-render-sweep',
  ],
} as const;

export interface ApproximateRichMaterialPrimitiveFields {
  readonly primitiveId: string;
  readonly fields: readonly string[];
}

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

const UNCONSUMED_MATERIAL_FIELD_CATEGORIES: Readonly<Record<string, UnconsumedMaterialFieldCategory>> = {
  spectralAttenuation: 'spectral',
  dispersionAbbeNumber: 'spectral',
  thinFilmStack: 'layered',
};

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
    const category = UNCONSUMED_MATERIAL_FIELD_CATEGORIES[field] ?? 'unknown';
    grouped[category].push(field);
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
 * Return primitive ids whose material asks for nontrivial `alphaMode:'blend'`.
 * The scalar alpha traversal path can faithfully discard fully-transparent
 * blend endpoints (`opacity <= 0`) and mask cutouts (`opacity < alphaCutoff`),
 * and the transparent-OIT pass camera-composites partial coverage. HybridEngine
 * still emits a structured warning because camera-visible light-map/emissive
 * terms are first-hit approximations, finite-emitter direct light is
 * fixed-stratified, and ReSTIR/GI participation remains approximate.
 */
export function collectApproximateAlphaBlendPrimitiveIds(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
    readonly positions?: ArrayLike<number> | undefined;
    readonly colors?: ArrayLike<number> | undefined;
  }>,
): string[] {
  const ids: string[] = [];
  for (const prim of primitives) {
    if (!materialBearingPrimitiveKind(prim.kind)) {
      continue;
    }
    const mat = prim.material;
    if (!mat || mat.alphaMode !== 'blend') continue;
    const opacity = effectiveScalarBlendOpacity(mat);
    const hasTextureAlphaSource = baseColorMapCanReduceAlpha(mat.baseColorMap) || mat.alphaMap != null;
    const hasVertexAlphaSource = hasFractionalVertexAlpha(prim.positions, prim.colors);
    if (opacity > 0 && (opacity < 1 || hasTextureAlphaSource || hasVertexAlphaSource)) {
      ids.push(prim.id ?? '(unnamed)');
    }
  }
  return ids.sort();
}

/**
 * Return primitive ids whose material-backed emissive maps still need a residual
 * all-path texel-PDF warning. This is a truthfulness surface, not a rejection:
 * walkaround samples readable emissive maps for visible glow and can split
 * eligible ReSTIR-DI finite emitters into exact texel-cell sub-triangles.
 * ReSTIR-GI suffix hits and direct probe hits sample the readable texel at the
 * hit UV. The remaining approximation is global texel-selection/PDF promotion
 * across every GI/RC/DDGI/fallback path.
 */
export function collectApproximateEmissiveMapTexelPdfPrimitiveIds(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
  emitters: ReadonlyArray<{
    readonly id?: unknown;
    readonly kind: string;
    readonly meshId?: unknown;
    readonly color?: ArrayLike<unknown>;
    readonly intensity?: unknown;
  }> = [],
): string[] {
  const ids = new Set<string>();
  const litMeshAreaIds = collectLitMeshAreaEmitterMeshIds(emitters);
  for (const prim of primitives) {
    if (!materialBearingPrimitiveKind(prim.kind)) {
      continue;
    }
    const mat = prim.material;
    if (!mat || mat.emissiveMap == null) continue;
    const id = prim.id ?? '(unnamed)';
    if (emissiveEnergyIsNonZero(mat) || litMeshAreaIds.has(String(id))) {
      ids.add(id);
    }
  }
  return [...ids].sort();
}

/**
 * Return primitive ids whose light map is rendered only as a camera-visible baked
 * outgoing-radiance term. This is consumed, but deliberately approximate: the
 * map is not treated as a scene light source and is not propagated through GI.
 */
export function collectApproximateLightMapPrimitiveIds(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
): string[] {
  const ids: string[] = [];
  for (const prim of primitives) {
    if (!materialBearingPrimitiveKind(prim.kind)) continue;
    const mat = prim.material;
    if (!mat || mat.lightMap == null) continue;
    const intensity = typeof mat.lightMapIntensity === 'number' && Number.isFinite(mat.lightMapIntensity)
      ? Math.max(0, mat.lightMapIntensity)
      : 1;
    if (intensity > 0) ids.push(prim.id ?? '(unnamed)');
  }
  return ids.sort();
}

/**
 * Return material-bearing primitives that exercise rich-material lobes that are
 * consumed by walkaround but still approximate pending material-furnace /
 * rich-GI reference promotion. Default scalar metadata does not warn; maps and
 * non-default active lobes do.
 */
export function collectApproximateRichMaterialPrimitiveFields(
  primitives: ReadonlyArray<{
    readonly id?: string;
    readonly kind: string;
    readonly material?: Record<string, unknown>;
  }>,
): ApproximateRichMaterialPrimitiveFields[] {
  const out: ApproximateRichMaterialPrimitiveFields[] = [];
  for (const prim of primitives) {
    if (!materialBearingPrimitiveKind(prim.kind)) continue;
    const fields = collectApproximateRichMaterialFieldsForMaterial(prim.material);
    if (fields.length > 0) {
      out.push({ primitiveId: prim.id ?? '(unnamed)', fields });
    }
  }
  return out.sort((a, b) => a.primitiveId.localeCompare(b.primitiveId));
}

export function collectApproximateRichMaterialFieldsForMaterial(
  material: Record<string, unknown> | undefined,
): string[] {
  if (!material) return [];
  const fields = new Set<string>();
  const has = (key: string): boolean => material[key] !== undefined && material[key] !== null;
  const scalar = (key: string, fallback: number): number => {
    const value = material[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  };
  const rgbDiffers = (key: string, fallback: readonly [number, number, number]): boolean => {
    const value = material[key];
    if (!Array.isArray(value)) return false;
    return Math.abs(Number(value[0] ?? fallback[0]) - fallback[0]) > 1e-6 ||
      Math.abs(Number(value[1] ?? fallback[1]) - fallback[1]) > 1e-6 ||
      Math.abs(Number(value[2] ?? fallback[2]) - fallback[2]) > 1e-6;
  };
  const vec2Differs = (key: string, fallback: readonly [number, number]): boolean => {
    const value = material[key];
    if (!Array.isArray(value)) return false;
    return Math.abs(Number(value[0] ?? fallback[0]) - fallback[0]) > 1e-6 ||
      Math.abs(Number(value[1] ?? fallback[1]) - fallback[1]) > 1e-6;
  };

  if (rgbDiffers('specularColor', [1, 1, 1])) fields.add('specularColor');
  if (has('specularColorMap')) fields.add('specularColorMap');
  if (Math.abs(scalar('specularIntensity', 1) - 1) > 1e-6) fields.add('specularIntensity');
  if (has('specularIntensityMap')) fields.add('specularIntensityMap');

  const clearcoatActive = scalar('clearcoat', 0) > 1e-6 || has('clearcoatMap');
  if (scalar('clearcoat', 0) > 1e-6) fields.add('clearcoat');
  if (has('clearcoatMap')) fields.add('clearcoatMap');
  if (clearcoatActive && Math.abs(scalar('clearcoatRoughness', 0) - 0) > 1e-6) fields.add('clearcoatRoughness');
  if (has('clearcoatRoughnessMap')) fields.add('clearcoatRoughnessMap');
  if (has('clearcoatNormalMap')) fields.add('clearcoatNormalMap');
  if (has('clearcoatNormalMap') && Math.abs(scalar('clearcoatNormalScale', 1) - 1) > 1e-6) fields.add('clearcoatNormalScale');

  const sheenActive = scalar('sheen', 0) > 1e-6 || has('sheenColorMap') || has('sheenRoughnessMap');
  if (scalar('sheen', 0) > 1e-6) fields.add('sheen');
  if (sheenActive && rgbDiffers('sheenColor', [0, 0, 0])) fields.add('sheenColor');
  if (sheenActive && Math.abs(scalar('sheenRoughness', 0) - 0) > 1e-6) fields.add('sheenRoughness');
  if (has('sheenColorMap')) fields.add('sheenColorMap');
  if (has('sheenRoughnessMap')) fields.add('sheenRoughnessMap');

  const anisotropyActive = Math.abs(scalar('anisotropy', 0)) > 1e-6 || has('anisotropyMap');
  if (Math.abs(scalar('anisotropy', 0)) > 1e-6) fields.add('anisotropy');
  if (anisotropyActive && Math.abs(scalar('anisotropyRotation', 0)) > 1e-6) fields.add('anisotropyRotation');
  if (has('anisotropyMap')) fields.add('anisotropyMap');

  const iridescenceActive = scalar('iridescence', 0) > 1e-6 || has('iridescenceMap') || has('iridescenceThicknessMap');
  if (scalar('iridescence', 0) > 1e-6) fields.add('iridescence');
  if (iridescenceActive && Math.abs(scalar('iridescenceIor', 1.3) - 1.3) > 1e-6) fields.add('iridescenceIor');
  if (iridescenceActive && vec2Differs('iridescenceThicknessRange', [100, 400])) fields.add('iridescenceThicknessRange');
  if (has('iridescenceMap')) fields.add('iridescenceMap');
  if (has('iridescenceThicknessMap')) fields.add('iridescenceThicknessMap');

  return [...fields].sort();
}

function collectLitMeshAreaEmitterMeshIds(
  emitters: ReadonlyArray<{
    readonly kind: string;
    readonly meshId?: unknown;
    readonly color?: ArrayLike<unknown>;
    readonly intensity?: unknown;
  }>,
): ReadonlySet<string> {
  const meshIds = new Set<string>();
  for (const emitter of emitters) {
    if (emitter.kind !== 'mesh-area' || emitter.meshId === undefined) continue;
    if (!emitterEnergyIsNonZero(emitter)) continue;
    meshIds.add(String(emitter.meshId));
  }
  return meshIds;
}

function effectiveScalarBlendOpacity(material: Record<string, unknown>): number {
  const opacity = unitAlpha(material.opacity, 1);
  const baseColor = material.baseColor as ArrayLike<unknown> | undefined;
  const baseAlpha = baseColor && typeof baseColor.length === 'number' && baseColor.length >= 4
    ? unitAlpha(baseColor[3], 1)
    : 1;
  return Math.min(1, Math.max(0, opacity * baseAlpha));
}

function asTextureHandle(value: unknown): unknown | null {
  if (value == null) return null;
  if (typeof value === 'object' && 'handle' in value) {
    return (value as { readonly handle?: unknown }).handle ?? null;
  }
  return value;
}

function textureHint(handle: unknown): {
  readonly channels?: number;
  readonly dataType?: string;
} | undefined {
  if (handle == null || typeof handle !== 'object') return undefined;
  const h = handle as {
    readonly __vitrum_hint__?: { readonly channels?: number; readonly dataType?: string };
    readonly channels?: number;
    readonly dataType?: string;
  };
  return h.__vitrum_hint__ ?? (
    h.channels != null || h.dataType != null
      ? {
          ...(h.channels != null ? { channels: h.channels } : {}),
          ...(h.dataType != null ? { dataType: h.dataType } : {}),
        }
      : undefined
  );
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function decodedTextureAlpha(value: number, src: ArrayLike<number>, hintDataType: string | undefined): number {
  const isFloat = src instanceof Float32Array;
  const useHalf = hintDataType != null
    ? hintDataType === 'float16' || hintDataType === 'half-float'
    : false;
  const useUint16 = hintDataType != null ? hintDataType === 'uint16' : src instanceof Uint16Array;
  const useFloat = hintDataType != null ? hintDataType === 'float32' : isFloat;
  const bpe = (src as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useUint16 || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  return useHalf ? halfToFloat(value) :
    useFloat ? value :
      useUint16 ? Math.min(1, Math.max(0, value / 65535)) :
      intMax > 0 ? value / intMax : value;
}

function baseColorMapCanReduceAlpha(value: unknown): boolean {
  const handle = asTextureHandle(value);
  if (handle == null || typeof handle !== 'object') return false;
  const h = handle as {
    readonly width?: number;
    readonly height?: number;
    readonly data?: ArrayLike<number>;
    readonly image?: { readonly width?: number; readonly height?: number; readonly data?: ArrayLike<number> };
  };
  const src = h.data ?? h.image?.data;
  const width = Number(h.width ?? h.image?.width ?? 0);
  const height = Number(h.height ?? h.image?.height ?? 0);
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) return false;

  const hint = textureHint(handle);
  const pixelCount = Math.max(1, width * height);
  const heuristicStride = Math.max(1, Math.round(src.length / pixelCount));
  const stride = hint?.channels ?? heuristicStride;
  if (stride < 4) return false;

  for (let p = 0; p < pixelCount; p += 1) {
    const alpha = decodedTextureAlpha(Number(src[p * stride + 3] ?? 1), src, hint?.dataType);
    if (Number.isFinite(alpha) && alpha < 0.999) return true;
  }
  return false;
}

function hasFractionalVertexAlpha(
  positions: ArrayLike<number> | undefined,
  colors: ArrayLike<number> | undefined,
): boolean {
  if (positions == null || colors == null) return false;
  const vertexCount = Math.floor(positions.length / 3);
  if (vertexCount <= 0) return false;
  const stride = colors.length / vertexCount;
  if (!Number.isInteger(stride) || stride < 4) return false;
  for (let v = 0; v < vertexCount; v += 1) {
    const alpha = unitAlpha(colors[v * stride + 3], 1);
    if (alpha > 0 && alpha < 1) return true;
  }
  return false;
}

function unitAlpha(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : fallback;
}

function emissiveEnergyIsNonZero(material: Record<string, unknown>): boolean {
  const intensity = typeof material.emissiveIntensity === 'number' && Number.isFinite(material.emissiveIntensity)
    ? Math.max(0, material.emissiveIntensity)
    : 1;
  if (intensity <= 0) return false;
  const emissive = material.emissive as ArrayLike<unknown> | undefined;
  if (!emissive || typeof emissive.length !== 'number') return false;
  for (let i = 0; i < Math.min(3, emissive.length); i += 1) {
    const channel = emissive[i];
    if (typeof channel === 'number' && Number.isFinite(channel) && channel > 0) {
      return true;
    }
  }
  return false;
}

function emitterEnergyIsNonZero(emitter: {
  readonly color?: ArrayLike<unknown>;
  readonly intensity?: unknown;
}): boolean {
  const intensity = typeof emitter.intensity === 'number' && Number.isFinite(emitter.intensity)
    ? Math.max(0, emitter.intensity)
    : 0;
  if (intensity <= 0) return false;
  const color = emitter.color;
  if (!color || typeof color.length !== 'number') return false;
  for (let i = 0; i < Math.min(3, color.length); i += 1) {
    const channel = color[i];
    if (typeof channel === 'number' && Number.isFinite(channel) && channel > 0) {
      return true;
    }
  }
  return false;
}
