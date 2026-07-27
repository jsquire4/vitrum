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
 * The atlas-diagnostic warning — previously a triple-repeated nested
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
  type UnconsumedMaterialPrimitiveFields,
} from './restir/consumedMaterialFields.js';

/** The method label carried on the emitted warning (`phase`/`method`). */
type WarnMethod = 'setScene' | 'updatePrimitive';

/**
 * Per-arm declarations for the material-texture-atlas diagnostic warning. Keyed
 * by `diagnostic.code`; replaces the former triple-repeated nested ternary.
 * `message` receives the fully-resolved diagnostic + the (already computed)
 * `sourcePathSuffix` (`" at <path>"` or `""`) so each arm formats itself once.
 */
interface AtlasDiagnosticArm {
  readonly warningCode:
    | 'walkaround-hybrid.ambiguous-material-texture-stride'
    | 'walkaround-hybrid.invalid-material-texture-transform'
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
  'unreadable-material-texture-map': {
    warningCode: 'walkaround-hybrid.unreadable-material-texture-map',
    fallback: 'map ignored',
    message: (d, method, suffix) =>
      `[vitrum/walkaround-hybrid] ${method}: ${d.field} on material slot ` +
      `${d.materialIndex}${suffix} ` +
      `has a texture handle that is neither CPU-readable nor a nominal ` +
      `WalkaroundWebGpuTextureSource descriptor; the map is ignored by the material atlas. Provide a raw ` +
      `{width,height,data}, DataTexture-shaped handle, or createWalkaroundWebGpuTextureSource ` +
      `descriptor before setScene/updatePrimitive for native map sampling.`,
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
