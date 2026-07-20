/**
 * Material-approximation warning subsystem — extracted from `HybridEngine.ts`
 * (T3-A). Mirrors the `HybridEngineDdgiSync` / `HybridEngineGIState` extraction
 * pattern: a focused helper the engine owns as a single field and delegates to.
 *
 * OWNS: the 9 once-only dedup `Set`s + every material-approximation /
 * lifecycle-truthfulness `warnX` method that HybridEngine used to inline. Every
 * warning code, message, `details` payload, dedup-key derivation, and emission
 * ORDER is preserved byte-identically (pinned by
 * `materialApproximationWarner.characterization.test.ts`).
 *
 * The atlas-diagnostic warning — previously a triple-repeated 5-arm nested
 * ternary — is now a single lookup table keyed by `diagnostic.code`
 * ({@link ATLAS_DIAGNOSTIC_TABLE}), so the message/code/fallback for each arm is
 * declared once.
 *
 * The engine injects a `warn(warning)` sink (its existing `_warn` — console +
 * subscriber fan-out); the warner never touches the console/subscribers itself.
 */

import type { EngineWarning } from '@vitrum/core';
import type { MaterialTextureAtlasDiagnostic } from './pipeline/materialTextureAtlas.js';
import {
  categorizeUnconsumedMaterialFields,
  EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS,
  LIGHT_MAP_CAMERA_VISIBLE_APPROXIMATION_DETAILS,
  RICH_MATERIAL_GI_APPROXIMATION_DETAILS,
  VOLUME_LAYER_TRANSPORT_APPROXIMATION_DETAILS,
  type ApproximateRichMaterialPrimitiveFields,
  type ApproximateVolumeLayerPrimitiveFields,
  type UnconsumedMaterialPrimitiveFields,
} from './restir/consumedMaterialFields.js';

/** The method label carried on the emitted warning (`phase`/`method`). */
type WarnMethod = 'setScene' | 'updatePrimitive';
/** The emissive-map warner additionally accepts the `updateEmitter` entry. */
type EmissiveWarnMethod = WarnMethod | 'updateEmitter';

/**
 * Per-arm declarations for the material-texture-atlas diagnostic warning. Keyed
 * by `diagnostic.code`; replaces the former triple-repeated nested ternary.
 * `message` receives the fully-resolved diagnostic + the (already computed)
 * `sourcePathSuffix` (`" at <path>"` or `""`) so each arm formats itself once.
 */
interface AtlasDiagnosticArm {
  readonly warningCode:
    | 'walkaround-hybrid.unsupported-material-texture-texcoord'
    | 'walkaround-hybrid.ambiguous-material-texture-stride'
    | 'walkaround-hybrid.invalid-material-texture-transform'
    | 'walkaround-hybrid.material-texture-sampler-policy-approximation'
    | 'walkaround-hybrid.unreadable-material-texture-map';
  readonly fallback: string;
  readonly message: (
    d: MaterialTextureAtlasDiagnostic,
    method: WarnMethod,
    sourcePathSuffix: string,
  ) => string;
}

const ATLAS_DIAGNOSTIC_TABLE: Readonly<
  Record<MaterialTextureAtlasDiagnostic['code'], AtlasDiagnosticArm>
> = {
  'unsupported-material-texture-texcoord': {
    warningCode: 'walkaround-hybrid.unsupported-material-texture-texcoord',
    fallback: 'map ignored',
    message: (d, method, suffix) =>
      `[vitrum/walkaround-hybrid] ${method}: ${d.field} on material slot ` +
      `${d.materialIndex}${suffix} ` +
      `uses texCoord ${d.texCoord}; the material atlas only supports UV sets 0 and 1, so the map is ignored.`,
  },
  'ambiguous-material-texture-stride': {
    warningCode: 'walkaround-hybrid.ambiguous-material-texture-stride',
    fallback: 'heuristic pixel stride',
    message: (d, method, suffix) =>
      `[vitrum/walkaround-hybrid] ${method}: ${d.field} on material slot ` +
      `${d.materialIndex}${suffix} ` +
      `has ambiguous raw pixel stride ${d.pixelStride} ` +
      `(${d.valueCount} values / ${d.width}x${d.height} pixels); ` +
      `the atlas decoded it heuristically. Attach __vitrum_hint__ = { channels: N } ` +
      `to make texture ingestion deterministic.`,
  },
  'invalid-material-texture-transform': {
    warningCode: 'walkaround-hybrid.invalid-material-texture-transform',
    fallback: 'identity texture transform fallback',
    message: (d, method, suffix) =>
      `[vitrum/walkaround-hybrid] ${method}: ${d.field} on material slot ` +
      `${d.materialIndex}${suffix} ` +
      `has non-finite texture transform component(s) ` +
      `${d.transformComponents?.join(', ') ?? '(unknown)'}; invalid components are replaced ` +
      `with the identity texture transform fallback and the map remains atlas-backed.`,
  },
  'material-texture-sampler-policy-approximation': {
    warningCode: 'walkaround-hybrid.material-texture-sampler-policy-approximation',
    fallback: 'base-level atlas sampler',
    message: (d, method, suffix) =>
      `[vitrum/walkaround-hybrid] ${method}: ${d.field} on material slot ` +
      `${d.materialIndex}${suffix} ` +
      `requests sampler policy ` +
      `mag=${d.magFilter ?? 'default'}, min=${d.minFilter ?? 'default'}, ` +
      `mip=${d.mipFilter ?? 'default'}; the material atlas honors footprint-independent ` +
      `nearest/linear filtering, but this policy needs implicit LOD or min/mag footprint selection ` +
      `in compute passes, so the map remains atlas-backed with approximate mip/footprint filtering.`,
  },
  'unreadable-material-texture-map': {
    warningCode: 'walkaround-hybrid.unreadable-material-texture-map',
    fallback: 'map ignored',
    message: (d, method, suffix) =>
      `[vitrum/walkaround-hybrid] ${method}: ${d.field} on material slot ` +
      `${d.materialIndex}${suffix} ` +
      `has a texture handle that is not CPU-readable; ` +
      `the map is ignored by the material atlas. Provide a raw {width,height,data} ` +
      `or DataTexture-shaped handle before setScene/updatePrimitive for native map sampling.`,
  },
};

/**
 * Owns the once-only material-approximation / truthfulness warnings for one
 * `HybridEngine` instance. The engine holds one of these and delegates.
 */
export class MaterialApproximationWarner {
  /** Structured-warning sink (the engine's `_warn`: console + subscribers). */
  private readonly _warn: (warning: EngineWarning) => void;

  /** Tracks which unconsumed-material-field sets have already been warned about
   *  (keyed by sorted join of the field names). Prevents duplicate console.warn
   *  calls across incremental `setScene` calls with the same ignored fields. */
  private readonly _warnedMaterialFields = new Set<string>();
  /** Tracks which fractional alpha-blend primitive sets have already warned. */
  private readonly _warnedAlphaBlendApproximationIds = new Set<string>();
  /** Tracks which emissive-map texel-PDF approximation primitive sets have warned. */
  private readonly _warnedEmissiveMapTexelPdfApproximationIds = new Set<string>();
  /** Tracks which light-map camera-visible approximation primitive sets have warned. */
  private readonly _warnedLightMapApproximationIds = new Set<string>();
  /** Tracks which rich-material approximation primitive/field sets have warned. */
  private readonly _warnedRichMaterialApproximationIds = new Set<string>();
  /** Tracks which volume/layer transport approximation primitive/field sets have warned. */
  private readonly _warnedVolumeLayerTransportApproximationIds = new Set<string>();
  /** Tracks atlas-backed material texture drops already reported to hosts. */
  private readonly _warnedMaterialTextureAtlasDiagnostics = new Set<string>();
  /** Tracks invalid setSize dimensions already reported to hosts. */
  private readonly _warnedInvalidSetSize = new Set<string>();
  /** Tracks unknown primitive patch field sets already reported to hosts. */
  private readonly _warnedUnknownPrimitivePatchFields = new Set<string>();

  constructor(warn: (warning: EngineWarning) => void) {
    this._warn = warn;
  }

  warnUnconsumedMaterialFields(
    fields: readonly string[],
    method: WarnMethod,
    primitiveFields: readonly UnconsumedMaterialPrimitiveFields[] = [],
  ): void {
    if (fields.length === 0) return;
    const sortedFields = Array.from(fields).sort();
    const key = sortedFields.join(',');
    if (this._warnedMaterialFields.has(key)) return;
    this._warnedMaterialFields.add(key);
    const categories = categorizeUnconsumedMaterialFields(sortedFields);
    this._warn({
      code: 'walkaround-hybrid.unconsumed-material-fields',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: the following material fields are ` +
        `supplied but not consumed by this backend: ${sortedFields.join(', ')}. ` +
        `See consumedMaterialFields.ts for the full allowlist.`,
      details: {
        fields: sortedFields,
        categories,
        primitiveFields,
      },
    });
  }

  warnApproximateAlphaBlendPrimitiveIds(
    primitiveIds: readonly string[],
    method: WarnMethod,
  ): void {
    if (primitiveIds.length === 0) return;
    const key = primitiveIds.join(',');
    if (this._warnedAlphaBlendApproximationIds.has(key)) return;
    this._warnedAlphaBlendApproximationIds.add(key);
    this._warn({
      code: 'walkaround-hybrid.alpha-blend-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: fractional or texture-driven alphaMode:'blend' ` +
        `is camera-composited by the transparent OIT pass, but transparent-layer ` +
        `ReSTIR/GI participation remains approximate; finite emitters are ` +
        `camera-visible fixed-stratified direct lights, not reservoir participants; ` +
        `primitives: ${primitiveIds.join(', ')}.`,
      details: { primitiveIds },
    });
  }

  warnApproximateEmissiveMapTexelPdfPrimitiveIds(
    primitiveIds: readonly string[],
    method: EmissiveWarnMethod,
  ): void {
    if (primitiveIds.length === 0) return;
    const key = primitiveIds.join(',');
    if (this._warnedEmissiveMapTexelPdfApproximationIds.has(key)) return;
    this._warnedEmissiveMapTexelPdfApproximationIds.add(key);
    this._warn({
      code: 'walkaround-hybrid.emissive-map-texel-pdf-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: material-backed emissiveMap ` +
        `surfaces are rendered; eligible ReSTIR-DI finite emitters are split ` +
        `into exact texel-cell sub-triangles, and GI/probe hit shading samples ` +
        `the readable texel at the hit UV, but full texel-space alias tables/PDFs ` +
        `are not guaranteed across every GI, RC, DDGI, and fallback sampling path; ` +
        `primitives: ${primitiveIds.join(', ')}.`,
      details: {
        primitiveIds,
        ...EMISSIVE_MAP_TEXEL_PDF_APPROXIMATION_DETAILS,
      },
    });
  }

  warnApproximateLightMapPrimitiveIds(
    primitiveIds: readonly string[],
    method: WarnMethod,
  ): void {
    if (primitiveIds.length === 0) return;
    const key = primitiveIds.join(',');
    if (this._warnedLightMapApproximationIds.has(key)) return;
    this._warnedLightMapApproximationIds.add(key);
    this._warn({
      code: 'walkaround-hybrid.light-map-camera-visible-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: lightMap is consumed as camera-visible baked ` +
        `outgoing radiance, but it is not sampled as a scene light and is not propagated ` +
        `through ReSTIR-GI, DDGI, or RC transport; primitives: ${primitiveIds.join(', ')}.`,
      details: {
        primitiveIds,
        ...LIGHT_MAP_CAMERA_VISIBLE_APPROXIMATION_DETAILS,
      },
    });
  }

  warnApproximateRichMaterialPrimitiveFields(
    primitiveFields: readonly ApproximateRichMaterialPrimitiveFields[],
    method: WarnMethod,
  ): void {
    if (primitiveFields.length === 0) return;
    const normalized = primitiveFields
      .map((entry) => ({
        primitiveId: entry.primitiveId,
        fields: [...entry.fields].sort(),
      }))
      .sort((a, b) => a.primitiveId.localeCompare(b.primitiveId));
    const key = normalized.map((entry) => `${entry.primitiveId}:${entry.fields.join('|')}`).join(',');
    if (this._warnedRichMaterialApproximationIds.has(key)) return;
    this._warnedRichMaterialApproximationIds.add(key);
    const fieldSet = [...new Set(normalized.flatMap((entry) => entry.fields))].sort();
    this._warn({
      code: 'walkaround-hybrid.rich-material-gi-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: rich material lobes are consumed ` +
        `by the realtime material path, but specular/clearcoat/sheen/anisotropy/` +
        `iridescence GI remains approximate pending material-furnace/reference A/B; ` +
        `primitives: ${normalized.map((entry) => entry.primitiveId).join(', ')}.`,
      details: {
        primitiveFields: normalized,
        fields: fieldSet,
        ...RICH_MATERIAL_GI_APPROXIMATION_DETAILS,
      },
    });
  }

  warnApproximateVolumeLayerPrimitiveFields(
    primitiveFields: readonly ApproximateVolumeLayerPrimitiveFields[],
    method: WarnMethod,
  ): void {
    if (primitiveFields.length === 0) return;
    const normalized = primitiveFields
      .map((entry) => ({
        primitiveId: entry.primitiveId,
        fields: [...entry.fields].sort(),
      }))
      .sort((a, b) => a.primitiveId.localeCompare(b.primitiveId));
    const key = normalized.map((entry) => `${entry.primitiveId}:${entry.fields.join('|')}`).join(',');
    if (this._warnedVolumeLayerTransportApproximationIds.has(key)) return;
    this._warnedVolumeLayerTransportApproximationIds.add(key);
    const fieldSet = [...new Set(normalized.flatMap((entry) => entry.fields))].sort();
    this._warn({
      code: 'walkaround-hybrid.volume-layer-transport-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: volume scattering and face-layer ` +
        `material fields are consumed by the compact realtime material path, but ` +
        `full participating-media and layered-stack transport remains approximate; ` +
        `primitives: ${normalized.map((entry) => entry.primitiveId).join(', ')}.`,
      details: {
        primitiveFields: normalized,
        fields: fieldSet,
        ...VOLUME_LAYER_TRANSPORT_APPROXIMATION_DETAILS,
      },
    });
  }

  warnReservedReceiveShadowPrimitiveIds(
    primitiveIds: readonly string[],
    method: WarnMethod,
  ): void {
    if (primitiveIds.length === 0) return;
    this._warn({
      code: 'walkaround-hybrid.reserved-receive-shadow',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: receiveShadow:false is reserved and not ` +
        `consumed by any backend (non-physical for GI); primitives: ${primitiveIds.join(', ')}.`,
      details: { primitiveIds },
    });
  }

  warnMaterialTextureAtlasDiagnostics(
    diagnostics: readonly MaterialTextureAtlasDiagnostic[],
    method: WarnMethod,
  ): void {
    for (const diagnostic of diagnostics) {
      const sourcePath = diagnostic.sourcePath;
      const key =
        `${method}:${diagnostic.code}:${diagnostic.materialIndex}:${diagnostic.field}:` +
        `${diagnostic.colorSpace}:${sourcePath ?? ''}:${diagnostic.texCoord ?? ''}:` +
        `${diagnostic.transformComponents?.join(',') ?? ''}:` +
        `${diagnostic.magFilter ?? ''}:${diagnostic.minFilter ?? ''}:${diagnostic.mipFilter ?? ''}:` +
        `${diagnostic.pixelStride ?? ''}:${diagnostic.valueCount ?? ''}`;
      if (this._warnedMaterialTextureAtlasDiagnostics.has(key)) continue;
      this._warnedMaterialTextureAtlasDiagnostics.add(key);

      const arm = ATLAS_DIAGNOSTIC_TABLE[diagnostic.code];
      const sourcePathSuffix = sourcePath !== undefined ? ` at ${sourcePath}` : '';
      this._warn({
        code: arm.warningCode,
        backend: 'walkaround-hybrid',
        phase: method,
        method,
        message: arm.message(diagnostic, method, sourcePathSuffix),
        details: {
          materialIndex: diagnostic.materialIndex,
          field: diagnostic.field,
          colorSpace: diagnostic.colorSpace,
          ...(diagnostic.texCoord !== undefined ? { texCoord: diagnostic.texCoord } : {}),
          ...(diagnostic.pixelStride !== undefined ? { pixelStride: diagnostic.pixelStride } : {}),
          ...(diagnostic.valueCount !== undefined ? { valueCount: diagnostic.valueCount } : {}),
          ...(diagnostic.width !== undefined ? { width: diagnostic.width } : {}),
          ...(diagnostic.height !== undefined ? { height: diagnostic.height } : {}),
          ...(diagnostic.transformComponents !== undefined
            ? { transformComponents: diagnostic.transformComponents }
            : {}),
          ...(diagnostic.magFilter !== undefined ? { magFilter: diagnostic.magFilter } : {}),
          ...(diagnostic.minFilter !== undefined ? { minFilter: diagnostic.minFilter } : {}),
          ...(diagnostic.mipFilter !== undefined ? { mipFilter: diagnostic.mipFilter } : {}),
          ...(sourcePath !== undefined ? { sourcePath } : {}),
          ...(diagnostic.textureIndex !== undefined ? { textureIndex: diagnostic.textureIndex } : {}),
          ...(diagnostic.imageIndex !== undefined ? { imageIndex: diagnostic.imageIndex } : {}),
          ...(diagnostic.samplerIndex !== undefined ? { samplerIndex: diagnostic.samplerIndex } : {}),
          ...(diagnostic.imageUri !== undefined ? { imageUri: diagnostic.imageUri } : {}),
          ...(diagnostic.imageMimeType !== undefined ? { imageMimeType: diagnostic.imageMimeType } : {}),
          ...(diagnostic.textureSourceExtension !== undefined
            ? { textureSourceExtension: diagnostic.textureSourceExtension }
            : {}),
          fallback: arm.fallback,
        },
      });
    }
  }

  warnUnknownPrimitivePatchFields(id: string, fields: readonly string[]): void {
    if (fields.length === 0) return;
    const sortedFields = Array.from(new Set(fields)).sort();
    const key = `${id}:${sortedFields.join(',')}`;
    if (this._warnedUnknownPrimitivePatchFields.has(key)) return;
    this._warnedUnknownPrimitivePatchFields.add(key);
    this._warn({
      code: 'walkaround-hybrid.unknown-primitive-patch-fields',
      backend: 'walkaround-hybrid',
      phase: 'mutation',
      method: 'updatePrimitive',
      message:
        `[vitrum/walkaround-hybrid] updatePrimitive("${id}"): patch fields are not ` +
        `recognized by this backend and were ignored: ${sortedFields.join(', ')}.`,
      details: { primitiveId: id, fields: sortedFields },
    });
  }

  warnInvalidSetSize(width: number, height: number): void {
    const key = `${width}x${height}`;
    if (this._warnedInvalidSetSize.has(key)) return;
    this._warnedInvalidSetSize.add(key);
    this._warn({
      code: 'walkaround-hybrid.invalid-set-size',
      backend: 'walkaround-hybrid',
      phase: 'lifecycle',
      method: 'setSize',
      message:
        `[vitrum/walkaround-hybrid] setSize(${width}, ${height}) ignored: ` +
        `width and height must both be positive before GPU frame resources can be resized.`,
      details: { width, height },
    });
  }
}
