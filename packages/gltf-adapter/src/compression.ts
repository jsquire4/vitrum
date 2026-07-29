// compression.ts — KHR_draco_mesh_compression + EXT/KHR_meshopt_compression
// resolution via built-in or host-supplied decoder hooks (GLTF-02).
//
// The adapter supplies lazy Draco and meshoptimizer decoders by default. Hosts
// may override them through `GltfToSceneOptions.dracoDecode` /
// `.meshoptDecode`; this module rewrites
// the (cloned) glTF JSON + buffers map so the rest of the pipeline sees plain
// uncompressed data:
//
//   - EXT/KHR_meshopt_compression sits on BUFFER VIEWS. Each compressed bufferView
//     is decoded into a synthetic buffer and the bufferView is repointed at it,
//     so ALL downstream consumers (attribute/index/animation accessors, image
//     bufferViews) transparently read decompressed bytes.
//   - KHR_draco_mesh_compression sits on MESH PRIMITIVES. Decoded typed arrays
//     become synthetic buffers/bufferViews/accessors and the cloned primitive
//     is repointed at them. A missing base index accessor is synthesized from
//     Draco's decoded face list rather than dropping connectivity.
//
// Failure-mode contract:
//   - required extension decode failure always throws;
//   - optional decode failure uses only a fully validated spec fallback;
//   - optional data without an exact fallback throws rather than publishing
//     malformed, partial, or unresolved geometry.
//
// Async: `gltfToScene` is already async, so hooks may return their result
// either synchronously or as a Promise — both are awaited.
//
// References:
//   - KHR_draco_mesh_compression
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_draco_mesh_compression/README.md
//   - EXT_meshopt_compression
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md
//   - KHR_meshopt_compression
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_meshopt_compression/README.md

import type { GltfJson } from './gltfTypes.js';
import { gltfPrimitiveKey, type GltfSceneReachability } from './sceneScope.js';
import { validateGltfPropertyExtensions } from './gltfPropertyValidation.js';
import { validateDeclaredBufferRange } from './bufferRangeValidation.js';
import {
  chargeCompressionHookOutput,
  checkedCompressionSum,
  CompressionAllocationLedger,
  compressionTypedArrayInfo,
  type CompressionDecodeState,
} from './compressionLimits.js';
import {
  preflightDracoCompressionDeclarations,
  resolveDracoStrict,
} from './strictDracoCompression.js';
import {
  gltfArrayBufferByteLength,
  GltfResourceLimitError,
  type ImportResourceLedger,
} from './importResourceBudget.js';

const DRACO_EXT = 'KHR_draco_mesh_compression';
const MESHOPT_EXT = 'EXT_meshopt_compression';
const MESHOPT_KHR_EXT = 'KHR_meshopt_compression';
const MESHOPT_EXTENSIONS = [MESHOPT_EXT, MESHOPT_KHR_EXT] as const;

// ── Public hook types ────────────────────────────────────────────────────────

/** Typed arrays a Draco decode hook may return per attribute / for indices. */
export type DracoTypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array | Float32Array;

export interface DracoDecodeResult {
  /**
   * Decoded vertex attributes keyed by glTF semantic (POSITION, NORMAL,
   * TEXCOORD_0, …). Each array must either:
   *   - match the declared accessor's `componentType` exactly (the adapter
   *     then applies the accessor's `normalized` flag itself), or
   *   - be a `Float32Array` of already-dequantized values (accepted for any
   *     declared componentType except `JOINTS_n`; normalization is considered
   *     done). Joint indices must preserve the accessor's unsigned integer
   *     representation so skinning indices cannot be rounded or normalized.
   * Length must equal `accessor.count × components(accessor.type)`.
   */
  readonly attributes: Readonly<Record<string, DracoTypedArray>>;
  /**
   * Decoded triangle-list face indices. TRIANGLE_STRIP source primitives are
   * converted to TRIANGLES with a fresh accessor count when Draco's face list
   * differs from the source strip accessor.
   */
  readonly indices: Uint8Array | Uint16Array | Uint32Array;
}

export type DracoAccessorComponentType = 5120 | 5121 | 5122 | 5123 | 5126;

export interface DracoAttributeDecodeSchema {
  readonly componentType: DracoAccessorComponentType;
  readonly normalized: boolean;
  readonly count: number;
  readonly type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4';
}

export interface DracoDecodeContext {
  /** Declared accessor schema for every Draco-owned attribute semantic. */
  readonly attributes: Readonly<Record<string, DracoAttributeDecodeSchema>>;
}

/**
 * Host-supplied Draco decode hook (KHR_draco_mesh_compression).
 *
 * @param compressed   - The raw compressed blob (the extension's bufferView).
 * @param attributeIds - The extension's `attributes` map: glTF semantic →
 *                       Draco attribute unique id (pass each id to
 *                       `decoder.GetAttributeByUniqueId`).
 * @param context      - Adapter-supplied declared accessor schemas. Optional so
 *                       existing two-argument host hooks remain compatible.
 * May return synchronously or as a Promise.
 */
export type DracoDecodeFn = (
  compressed: Uint8Array,
  attributeIds: Readonly<Record<string, number>>,
  context?: DracoDecodeContext,
) => DracoDecodeResult | Promise<DracoDecodeResult>;

/** EXT/KHR_meshopt_compression `mode` values. */
export type MeshoptMode = 'ATTRIBUTES' | 'TRIANGLES' | 'INDICES';
/** EXT/KHR_meshopt_compression `filter` values. */
export type MeshoptFilter = 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL' | 'COLOR';

/**
 * Host-supplied meshopt decode hook (EXT/KHR_meshopt_compression). The signature
 * mirrors `MeshoptDecoder.decodeGltfBuffer(target, count, stride, source,
 * mode, filter)` from the `meshoptimizer` package, minus the target (return
 * the decoded bytes instead): must return exactly `count × byteStride` bytes.
 * May return synchronously or as a Promise.
 */
export type MeshoptDecodeFn = (
  compressed: Uint8Array,
  count: number,
  byteStride: number,
  mode: MeshoptMode,
  filter: MeshoptFilter,
) => Uint8Array | Promise<Uint8Array>;

export interface GltfDecodeHooks {
  readonly dracoDecode?: DracoDecodeFn | undefined;
  readonly meshoptDecode?: MeshoptDecodeFn | undefined;
}

export interface GltfCompressionScope {
  readonly sceneReachability?: GltfSceneReachability | undefined;
  readonly bufferViewIndices?: ReadonlySet<number> | undefined;
  readonly resourceLedger?: ImportResourceLedger | undefined;
  readonly hookOutputPrecharged?: {
    readonly draco?: boolean | undefined;
    readonly meshopt?: boolean | undefined;
  } | undefined;
}

export type GltfCompressionDiagnosticCode =
  | 'draco-fallback-accessors-used'
  | 'meshopt-buffer-unavailable'
  | 'meshopt-codec-version-unsupported'
  | 'meshopt-invalid-bitstream-header'
  | 'meshopt-compression-budget-exceeded'
  | 'meshopt-decode-hook-failed'
  | 'meshopt-decoded-byte-length-mismatch'
  | 'meshopt-fallback-buffer-used';

export interface GltfCompressionDiagnostic {
  readonly severity: 'warning';
  readonly code: GltfCompressionDiagnosticCode;
  readonly path: string;
  readonly message: string;
  readonly extension:
    | 'KHR_draco_mesh_compression'
    | 'EXT_meshopt_compression'
    | 'KHR_meshopt_compression';
  readonly meshIndex?: number;
  readonly primitiveIndex?: number;
  readonly bufferViewIndex?: number;
  readonly accessorIndex?: number;
  readonly semantic?: string;
}

type GltfCompressionDiagnosticSink = (diagnostic: GltfCompressionDiagnostic) => void;

// ── Extension JSON shapes ────────────────────────────────────────────────────

interface MeshoptBufferViewExt {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride: number;
  count: number;
  mode: MeshoptMode;
  filter?: MeshoptFilter;
}

type ValidatedMeshoptExtension = MeshoptBufferViewExt & {
  readonly byteOffset: number;
  readonly filter: MeshoptFilter;
};

// ── Entry point ──────────────────────────────────────────────────────────────

/**
 * Resolve all compressed geometry in `gltf` using the supplied hooks.
 *
 * Returns the input `gltf` untouched when no compression extensions are in
 * use; otherwise returns a CLONE with compressed bufferViews / primitive
 * accessors rewritten to synthetic uncompressed buffers (added to `buffers`).
 * The caller's `gltf` object is never mutated.
 *
 * @throws when required decoding fails, or optional compressed data has no
 *         fully valid spec-defined fallback.
 */
export async function resolveCompression(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  hooks: GltfDecodeHooks,
  warnings: string[],
  onDiagnostic?: GltfCompressionDiagnosticSink,
  scope: GltfCompressionScope = {},
): Promise<GltfJson> {
  const hasMeshopt = (gltf.bufferViews ?? []).some(
    (bv, index) =>
      (scope.bufferViewIndices === undefined || scope.bufferViewIndices.has(index)) &&
      getMeshoptBufferViewExtension(bv) !== undefined,
  );
  const hasDraco = (gltf.meshes ?? []).some((m, meshIndex) =>
    m.primitives.some((p, primitiveIndex) => {
      const extensions = hasOwn(p, 'extensions') ? p.extensions : undefined;
      return (
        (scope.sceneReachability === undefined ||
          scope.sceneReachability.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) &&
        extensions !== undefined &&
        hasOwn(extensions, DRACO_EXT) &&
        extensions[DRACO_EXT] !== undefined
      );
    }),
  );
  if (!hasMeshopt && !hasDraco) return gltf;

  // structuredClone intentionally ignores symbol-keyed properties. Validate
  // the caller-owned declarations first so an enumerable symbol cannot vanish
  // and accidentally turn malformed JSON into accepted input.
  if (hasMeshopt) {
    validateMeshoptFallbackBufferReferences(gltf);
    for (const [index, bufferView] of (gltf.bufferViews ?? []).entries()) {
      if (scope.bufferViewIndices !== undefined && !scope.bufferViewIndices.has(index)) continue;
      const meshopt = getMeshoptBufferViewExtension(bufferView);
      if (meshopt === undefined) continue;
      const ext = validateMeshoptExtension(
        meshopt.value,
        `bufferViews[${index}].extensions.${meshopt.name}`,
        meshopt.name,
      );
      validateDeclaredBufferRange(
        gltf,
        ext.buffer,
        ext.byteOffset,
        ext.byteLength,
        `bufferViews[${index}].extensions.${meshopt.name} source`,
      );
      const parent = validateMeshoptParentLayout(gltf, index, meshopt.name, ext);
      validateMeshoptBufferMarker(gltf, parent.bufferIndex, meshopt.name);
      validateMeshoptBufferMarker(gltf, ext.buffer, meshopt.name);
    }
  }
  if (hasDraco) {
    preflightDracoCompressionDeclarations(
      gltf,
      scope.sceneReachability,
      (gltf.extensionsRequired ?? []).includes(DRACO_EXT),
    );
  }

  // Copy-on-write across BOTH externally visible products. Synthetic buffers
  // are staged in a private Map and published only after every reachable decode
  // and rewrite has succeeded. A late Draco failure therefore cannot leak the
  // meshopt buffers produced earlier in the same call.
  const out = structuredClone(gltf);
  const initialBufferEntries = new Map(buffers);
  const stagedBuffers = new Map(initialBufferEntries);
  const stagedWarnings: string[] = [];
  const stagedDiagnostics: GltfCompressionDiagnostic[] = [];
  // A top-level import already has separate, typed encoded-resource and decoded-
  // geometry ledgers. Do not superimpose the legacy fixed compression ceiling:
  // it would contradict an explicit public zero opt-out (and would also combine
  // two independently governed resource domains). The local ledger still
  // validates every cumulative sum against the safe-integer range. Standalone
  // internal calls without an import ledger retain the conservative default.
  const allocationLedger = new CompressionAllocationLedger(
    scope.resourceLedger === undefined ? undefined : 0,
  );
  const decodeState: CompressionDecodeState = { attemptsDisabled: false };
  const required = new Set(out.extensionsRequired ?? []);

  // meshopt first: it operates at bufferView level, so a (theoretical) Draco
  // blob inside a meshopt-wrapped view would already be decompressed.
  if (hasMeshopt) {
    await _resolveMeshopt(
      out,
      stagedBuffers,
      hooks.meshoptDecode,
      required,
      stagedWarnings,
      (diagnostic) => stagedDiagnostics.push(diagnostic),
      scope.bufferViewIndices,
      allocationLedger,
      decodeState,
      scope.resourceLedger,
      scope.hookOutputPrecharged?.meshopt === true,
    );
  }
  if (hasDraco) {
    await resolveDracoStrict(
      out,
      stagedBuffers,
      hooks.dracoDecode,
      required.has(DRACO_EXT),
      stagedWarnings,
      (diagnostic) => stagedDiagnostics.push(diagnostic),
      scope.sceneReachability,
      allocationLedger,
      decodeState,
      scope.resourceLedger,
      scope.hookOutputPrecharged?.draco === true,
    );
  }
  for (const [index, initialBuffer] of initialBufferEntries) {
    if (!buffers.has(index) || buffers.get(index) !== initialBuffer) {
      throw new Error(
        `[vitrum/gltf-adapter] Caller buffer Map entry ${index} changed during compressed geometry decoding; no adapter-owned buffers were published.`,
      );
    }
  }
  const stagedNewBuffers: Array<readonly [number, ArrayBuffer]> = [];
  for (const [index, buffer] of stagedBuffers) {
    if (initialBufferEntries.has(index)) continue;
    if (buffers.has(index)) {
      throw new Error(
        `[vitrum/gltf-adapter] Caller buffer Map claimed synthetic buffer index ${index} during compressed geometry decoding; no adapter-owned buffers were published.`,
      );
    }
    stagedNewBuffers.push([index, buffer]);
  }
  for (const [index, buffer] of stagedNewBuffers) {
    buffers.set(index, buffer);
  }
  warnings.push(...stagedWarnings);
  for (const diagnostic of stagedDiagnostics) {
    try {
      onDiagnostic?.(diagnostic);
    } catch {
      // Diagnostics are observational and cannot invalidate a completed decode.
    }
  }
  return out;
}

// ── Synthetic buffer plumbing ────────────────────────────────────────────────

function _nextBufferIndex(gltf: GltfJson, buffers: Map<number, ArrayBuffer>): number {
  let next = gltf.buffers?.length ?? 0;
  // Map entries that are not backed by glTF buffer descriptors are host data,
  // not an instruction to allocate every intervening array slot. In
  // particular, never let one stray huge key drive unbounded descriptor
  // padding. Start at the first descriptor-free index and skip only exact
  // occupied collisions.
  while (buffers.has(next)) {
    if (next >= Number.MAX_SAFE_INTEGER) {
      throw new RangeError('[vitrum/gltf-adapter] No safe synthetic buffer index remains.');
    }
    next += 1;
  }
  return next;
}

/** Copy `bytes` into a fresh buffer, register it, and return its index. */
function _addSyntheticBuffer(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  bytes: Uint8Array,
  allocationLedger: CompressionAllocationLedger,
  allocationPath: string,
  resourceLedger: ImportResourceLedger | undefined,
): number {
  const info = compressionTypedArrayInfo(bytes);
  if (info?.kind !== 'Uint8Array' || info.shared) {
    throw new TypeError(`[vitrum/gltf-adapter] ${allocationPath} must be a non-shared Uint8Array.`);
  }
  allocationLedger.charge(info.byteLength, `${allocationPath} retained copy`);
  resourceLedger?.chargeDecodedGeometryBytes(info.byteLength, `${allocationPath} retained copy`);
  const idx = _nextBufferIndex(gltf, buffers);
  const copy = new Uint8Array(bytes); // exact-size copy, detached from source
  buffers.set(idx, copy.buffer);
  // Keep gltf.buffers index-consistent if an exact host-map collision caused a
  // small descriptor gap.
  const list = (gltf.buffers ??= []);
  while (list.length < idx) list.push({ byteLength: 0 });
  list.push({ byteLength: copy.byteLength });
  return idx;
}

function _stripExtension(holder: { extensions?: Record<string, unknown> }, name: string): void {
  if (!holder.extensions) return;
  delete holder.extensions[name];
  if (Object.keys(holder.extensions).length === 0) delete holder.extensions;
}

function emitCompressionDiagnostic(
  warnings: string[],
  onDiagnostic: GltfCompressionDiagnosticSink | undefined,
  diagnostic: GltfCompressionDiagnostic,
): void {
  warnings.push(diagnostic.message);
  try {
    onDiagnostic?.(diagnostic);
  } catch {
    // Host diagnostic callbacks must not abort compression fallback analysis.
  }
}

function getMeshoptBufferViewExtension(bufferView: {
  readonly extensions?: Record<string, unknown>;
}): { readonly name: (typeof MESHOPT_EXTENSIONS)[number]; readonly value: unknown } | undefined {
  const extensions = hasOwn(bufferView, 'extensions') ? bufferView.extensions : undefined;
  const extValue =
    extensions !== undefined && hasOwn(extensions, MESHOPT_EXT)
      ? extensions[MESHOPT_EXT]
      : undefined;
  const khrValue =
    extensions !== undefined && hasOwn(extensions, MESHOPT_KHR_EXT)
      ? extensions[MESHOPT_KHR_EXT]
      : undefined;
  if (extValue !== undefined && khrValue !== undefined) {
    throw new Error(
      `[vitrum/gltf-adapter] ${MESHOPT_EXT} and ${MESHOPT_KHR_EXT} ` +
        'must not coexist on one bufferView.',
    );
  }
  if (extValue !== undefined) return { name: MESHOPT_EXT, value: extValue };
  if (khrValue !== undefined) return { name: MESHOPT_KHR_EXT, value: khrValue };
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function assertExactEnumerableKeys(
  value: object,
  allowed: ReadonlySet<string>,
  path: string,
): void {
  for (const key of Reflect.ownKeys(value)) {
    if (!Object.prototype.propertyIsEnumerable.call(value, key)) continue;
    if (typeof key !== 'string' || !allowed.has(key)) {
      const rendered = typeof key === 'symbol' ? key.toString() : JSON.stringify(key);
      throw new TypeError(
        `[vitrum/gltf-adapter] ${path} contains unsupported enumerable field ${rendered}.`,
      );
    }
  }
}

function requireSafeInteger(value: unknown, path: string, minimum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} must be a safe integer >= ${minimum}; received ${String(value)}.`,
    );
  }
  return value as number;
}

function checkedProduct(a: number, b: number, path: string): number {
  const result = a * b;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path} is not a safe integer.`);
  }
  return result;
}

function formatUnknownError(error: unknown): string {
  try {
    return String(error);
  } catch {
    return 'unknown decoder failure';
  }
}

function validateMeshoptExtension(
  raw: unknown,
  path: string,
  name: (typeof MESHOPT_EXTENSIONS)[number],
): ValidatedMeshoptExtension {
  if (!isRecord(raw)) {
    throw new TypeError(`[vitrum/gltf-adapter] ${path} must be an object.`);
  }
  assertExactEnumerableKeys(
    raw,
    new Set([
      'buffer',
      'byteOffset',
      'byteLength',
      'byteStride',
      'count',
      'mode',
      'filter',
      'extensions',
      'extras',
    ]),
    path,
  );
  validateGltfPropertyExtensions(raw, path);
  for (const key of ['buffer', 'byteLength', 'byteStride', 'count', 'mode'] as const) {
    if (!hasOwn(raw, key)) {
      throw new TypeError(`[vitrum/gltf-adapter] ${path}.${key} must be an own property.`);
    }
  }
  const buffer = requireSafeInteger(raw.buffer, `${path}.buffer`, 0);
  const byteOffset =
    !hasOwn(raw, 'byteOffset') || raw.byteOffset === undefined
      ? 0
      : requireSafeInteger(raw.byteOffset, `${path}.byteOffset`, 0);
  const byteLength = requireSafeInteger(raw.byteLength, `${path}.byteLength`, 1);
  const byteStride = requireSafeInteger(raw.byteStride, `${path}.byteStride`, 1);
  if (byteStride > 256) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path}.byteStride must be <= 256.`);
  }
  const count = requireSafeInteger(raw.count, `${path}.count`, 1);
  checkedProduct(count, byteStride, `${path}.count * byteStride`);
  if (raw.mode !== 'ATTRIBUTES' && raw.mode !== 'TRIANGLES' && raw.mode !== 'INDICES') {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.mode must be ATTRIBUTES, TRIANGLES, or INDICES.`,
    );
  }
  const filter = hasOwn(raw, 'filter') ? raw.filter : 'NONE';
  if (
    filter !== 'NONE' &&
    filter !== 'OCTAHEDRAL' &&
    filter !== 'QUATERNION' &&
    filter !== 'EXPONENTIAL' &&
    filter !== 'COLOR'
  ) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.filter must be NONE, OCTAHEDRAL, ` +
        'QUATERNION, EXPONENTIAL, or COLOR.',
    );
  }
  if (filter === 'COLOR' && name !== MESHOPT_KHR_EXT) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.filter COLOR is only valid for ${MESHOPT_KHR_EXT}.`,
    );
  }
  if (raw.mode === 'ATTRIBUTES' && byteStride % 4 !== 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path}.byteStride must be a multiple of 4 for ATTRIBUTES.`,
    );
  }
  if (raw.mode !== 'ATTRIBUTES') {
    if (byteStride !== 2 && byteStride !== 4) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${path}.byteStride must be 2 or 4 for ${raw.mode}.`,
      );
    }
    if (filter !== 'NONE') {
      throw new TypeError(`[vitrum/gltf-adapter] ${path}.filter must be NONE for ${raw.mode}.`);
    }
    if (raw.mode === 'TRIANGLES' && count % 3 !== 0) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${path}.count must be divisible by 3 for TRIANGLES.`,
      );
    }
  } else if (filter === 'OCTAHEDRAL' && byteStride !== 4 && byteStride !== 8) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path}.byteStride must be 4 or 8 for OCTAHEDRAL.`);
  } else if (filter === 'QUATERNION' && byteStride !== 8) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path}.byteStride must be 8 for QUATERNION.`);
  } else if (filter === 'EXPONENTIAL' && byteStride % 4 !== 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path}.byteStride must be a multiple of 4 for EXPONENTIAL.`,
    );
  } else if (filter === 'COLOR' && byteStride !== 4 && byteStride !== 8) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path}.byteStride must be 4 or 8 for COLOR.`);
  }
  return {
    buffer,
    byteOffset,
    byteLength,
    byteStride,
    count,
    mode: raw.mode,
    filter,
  };
}

function validateMeshoptBufferMarker(
  gltf: GltfJson,
  bufferIndex: number,
  preferredName: (typeof MESHOPT_EXTENSIONS)[number],
): { readonly fallback?: boolean } | undefined {
  const descriptor = gltf.buffers?.[bufferIndex];
  const extensions =
    descriptor !== undefined && hasOwn(descriptor, 'extensions')
      ? descriptor.extensions
      : undefined;
  const extValue =
    extensions !== undefined && hasOwn(extensions, MESHOPT_EXT)
      ? extensions[MESHOPT_EXT]
      : undefined;
  const khrValue =
    extensions !== undefined && hasOwn(extensions, MESHOPT_KHR_EXT)
      ? extensions[MESHOPT_KHR_EXT]
      : undefined;
  if (extValue !== undefined && khrValue !== undefined) {
    throw new Error(
      `[vitrum/gltf-adapter] ${MESHOPT_EXT} and ${MESHOPT_KHR_EXT} ` +
        `must not coexist on buffer ${bufferIndex}.`,
    );
  }
  const markerName =
    extValue !== undefined ? MESHOPT_EXT : khrValue !== undefined ? MESHOPT_KHR_EXT : preferredName;
  const raw =
    extensions !== undefined && hasOwn(extensions, markerName) ? extensions[markerName] : undefined;
  if (raw === undefined) return undefined;
  const path = `buffers[${bufferIndex}].extensions.${markerName}`;
  if (!isRecord(raw)) {
    throw new TypeError(`[vitrum/gltf-adapter] ${path} must be an object.`);
  }
  assertExactEnumerableKeys(raw, new Set(['fallback', 'extensions', 'extras']), path);
  validateGltfPropertyExtensions(raw, path);
  const fallback = hasOwn(raw, 'fallback') ? raw.fallback : undefined;
  if (fallback !== undefined && typeof fallback !== 'boolean') {
    throw new TypeError(`[vitrum/gltf-adapter] ${path}.fallback must be boolean.`);
  }
  if (fallback === true && markerName !== preferredName) {
    throw new Error(
      `[vitrum/gltf-adapter] buffer ${bufferIndex} uses ${markerName} ` +
        `fallback:true, but its referencing bufferView uses ${preferredName}.`,
    );
  }
  return fallback === undefined ? {} : { fallback };
}

/**
 * Validate the asset-wide ownership rule for buffers tagged `fallback:true`.
 *
 * Both meshopt specifications reserve such a buffer for uncompressed fallback
 * storage: every bufferView that points at it must carry the matching meshopt
 * extension, and a meshopt extension must never point at it as compressed input.
 * This is necessarily global rather than scoped to the currently imported scene.
 */
function validateMeshoptFallbackBufferReferences(gltf: GltfJson): void {
  const fallbackBuffers = new Map<number, (typeof MESHOPT_EXTENSIONS)[number]>();

  for (const [bufferIndex, descriptor] of (gltf.buffers ?? []).entries()) {
    if (descriptor == null) continue;
    const extensions = hasOwn(descriptor, 'extensions') ? descriptor.extensions : undefined;
    const extValue =
      extensions !== undefined && hasOwn(extensions, MESHOPT_EXT)
        ? extensions[MESHOPT_EXT]
        : undefined;
    const khrValue =
      extensions !== undefined && hasOwn(extensions, MESHOPT_KHR_EXT)
        ? extensions[MESHOPT_KHR_EXT]
        : undefined;
    if (extValue !== undefined && khrValue !== undefined) {
      throw new Error(
        `[vitrum/gltf-adapter] ${MESHOPT_EXT} and ${MESHOPT_KHR_EXT} ` +
          `must not coexist on buffer ${bufferIndex}.`,
      );
    }
    const markerName =
      extValue !== undefined ? MESHOPT_EXT : khrValue !== undefined ? MESHOPT_KHR_EXT : undefined;
    if (markerName === undefined) continue;
    const marker = validateMeshoptBufferMarker(gltf, bufferIndex, markerName);
    if (marker?.fallback === true) fallbackBuffers.set(bufferIndex, markerName);
  }

  if (fallbackBuffers.size === 0) return;

  for (const [bufferViewIndex, bufferView] of (gltf.bufferViews ?? []).entries()) {
    const meshopt = getMeshoptBufferViewExtension(bufferView);
    const parentFallbackName = fallbackBuffers.get(bufferView.buffer);
    if (parentFallbackName !== undefined) {
      if (meshopt === undefined) {
        throw new Error(
          `[vitrum/gltf-adapter] buffer ${bufferView.buffer} is marked ` +
            `${parentFallbackName} fallback:true, so bufferViews[${bufferViewIndex}] ` +
            `must carry ${parentFallbackName}.`,
        );
      }
      if (meshopt.name !== parentFallbackName) {
        throw new Error(
          `[vitrum/gltf-adapter] buffer ${bufferView.buffer} is marked ` +
            `${parentFallbackName} fallback:true, but bufferViews[${bufferViewIndex}] ` +
            `carries ${meshopt.name}.`,
        );
      }
    }
    if (meshopt === undefined) continue;
    const extensionPath = `bufferViews[${bufferViewIndex}].extensions.${meshopt.name}`;
    const ext = validateMeshoptExtension(meshopt.value, extensionPath, meshopt.name);
    const sourceFallbackName = fallbackBuffers.get(ext.buffer);
    if (sourceFallbackName !== undefined) {
      throw new Error(
        `[vitrum/gltf-adapter] ${extensionPath}.buffer references buffer ` +
          `${ext.buffer}, which is marked ${sourceFallbackName} fallback:true ` +
          'and cannot be used as compressed source data.',
      );
    }
  }
}

function validateMeshoptParentLayout(
  gltf: GltfJson,
  bufferViewIndex: number,
  name: (typeof MESHOPT_EXTENSIONS)[number],
  ext: ValidatedMeshoptExtension,
): {
  readonly bufferIndex: number;
  readonly byteOffset: number;
  readonly byteLength: number;
} {
  const path = `bufferViews[${bufferViewIndex}]`;
  const bv = gltf.bufferViews?.[bufferViewIndex];
  if (bv == null) throw new Error(`[vitrum/gltf-adapter] ${path} is missing.`);
  if (!hasOwn(bv, 'buffer') || !hasOwn(bv, 'byteLength')) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.buffer and ${path}.byteLength must be own properties.`,
    );
  }
  const bufferIndex = requireSafeInteger(bv.buffer, `${path}.buffer`, 0);
  const byteOffset =
    !hasOwn(bv, 'byteOffset') || bv.byteOffset === undefined
      ? 0
      : requireSafeInteger(bv.byteOffset, `${path}.byteOffset`, 0);
  const byteLength = requireSafeInteger(bv.byteLength, `${path}.byteLength`, 1);
  const expected = checkedProduct(ext.count, ext.byteStride, `${path} decoded byte length`);
  if (byteLength !== expected) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path}.byteLength must equal decoded count * ` +
        `byteStride (${expected}); received ${byteLength}.`,
    );
  }
  const parentStride =
    !hasOwn(bv, 'byteStride') || bv.byteStride === undefined
      ? undefined
      : requireSafeInteger(bv.byteStride, `${path}.byteStride`, 4);
  if (parentStride !== undefined && (parentStride > 252 || parentStride % 4 !== 0)) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path}.byteStride must be a multiple of 4 ` +
        `between 4 and 252; received ${parentStride}.`,
    );
  }
  if (name === MESHOPT_EXT && parentStride !== undefined && parentStride !== ext.byteStride) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path}.byteStride must equal ${ext.byteStride} ` +
        `for ${MESHOPT_EXT}; received ${parentStride}.`,
    );
  }
  validateDeclaredBufferRange(gltf, bufferIndex, byteOffset, byteLength, path);
  return { bufferIndex, byteOffset, byteLength };
}

function checkedByteSlice(
  buffer: ArrayBuffer,
  byteOffset: number,
  byteLength: number,
  path: string,
): Uint8Array {
  const end = byteOffset + byteLength;
  const intrinsicByteLength = gltfArrayBufferByteLength(buffer);
  if (
    intrinsicByteLength === undefined ||
    !Number.isSafeInteger(end) ||
    end > intrinsicByteLength
  ) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} range [${byteOffset}, ${String(end)}) exceeds buffer length ${String(intrinsicByteLength)}.`,
    );
  }
  return new Uint8Array(buffer, byteOffset, byteLength);
}

function meshoptHeaderFailure(
  name: (typeof MESHOPT_EXTENSIONS)[number],
  ext: ValidatedMeshoptExtension,
  compressed: Uint8Array,
): { readonly reason: string; readonly code: GltfCompressionDiagnosticCode } | undefined {
  const header = compressed[0];
  if (ext.mode === 'ATTRIBUTES') {
    if (name === MESHOPT_EXT && header === 0xa1) {
      return {
        reason: `${MESHOPT_EXT} does not permit the meshopt ATTRIBUTES codec v1 header 0xa1`,
        code: 'meshopt-codec-version-unsupported',
      };
    }
    if (name === MESHOPT_EXT ? header !== 0xa0 : header !== 0xa0 && header !== 0xa1) {
      const expected = name === MESHOPT_EXT ? '0xa0' : '0xa0 or 0xa1';
      return {
        reason:
          `${name} ATTRIBUTES data has invalid header ` +
          `${header === undefined ? 'undefined' : `0x${header.toString(16)}`}; expected ${expected}`,
        code: 'meshopt-invalid-bitstream-header',
      };
    }
    return undefined;
  }
  const expected = ext.mode === 'TRIANGLES' ? 0xe1 : 0xd1;
  if (header !== expected) {
    return {
      reason:
        `${name} ${ext.mode} data has invalid header ` +
        `${header === undefined ? 'undefined' : `0x${header.toString(16)}`}; ` +
        `expected 0x${expected.toString(16)}`,
      code: 'meshopt-invalid-bitstream-header',
    };
  }
  return undefined;
}

function validateMeshoptFallback(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  bufferViewIndex: number,
  name: (typeof MESHOPT_EXTENSIONS)[number],
  ext: ValidatedMeshoptExtension,
): {
  readonly bufferIndex: number;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly bytes: Uint8Array;
  readonly markedFallback: boolean;
} {
  const path = `bufferViews[${bufferViewIndex}]`;
  const parent = validateMeshoptParentLayout(gltf, bufferViewIndex, name, ext);
  const marker = validateMeshoptBufferMarker(gltf, parent.bufferIndex, name);
  const fallback = buffers.get(parent.bufferIndex);
  if (fallback == null) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} fallback buffer ${parent.bufferIndex} is unavailable.`,
    );
  }
  const bytes = checkedByteSlice(
    fallback,
    parent.byteOffset,
    parent.byteLength,
    `${path} fallback`,
  );
  return {
    ...parent,
    bytes,
    markedFallback: marker?.fallback === true,
  };
}

function clearMeshoptFallbackMarker(gltf: GltfJson, bufferIndex: number): void {
  const descriptor = gltf.buffers?.[bufferIndex];
  if (descriptor?.extensions === undefined) return;
  for (const name of MESHOPT_EXTENSIONS) {
    const raw = descriptor.extensions[name];
    if (!isRecord(raw) || raw.fallback !== true) continue;
    delete raw.fallback;
    if (Object.keys(raw).length === 0) delete descriptor.extensions[name];
  }
  if (Object.keys(descriptor.extensions).length === 0) {
    delete descriptor.extensions;
  }
}

function hasUnresolvedMeshoptReference(
  gltf: GltfJson,
  bufferIndex: number,
  exceptBufferViewIndex: number,
): boolean {
  return (gltf.bufferViews ?? []).some(
    (candidate, candidateIndex) =>
      candidateIndex !== exceptBufferViewIndex &&
      candidate.buffer === bufferIndex &&
      getMeshoptBufferViewExtension(candidate) !== undefined,
  );
}

function hasUnresolvedMeshoptReferenceOutsideScope(
  gltf: GltfJson,
  bufferIndex: number,
  exceptBufferViewIndex: number,
  scopedBufferViews: ReadonlySet<number> | undefined,
): boolean {
  if (scopedBufferViews === undefined) return false;
  return (gltf.bufferViews ?? []).some(
    (candidate, candidateIndex) =>
      candidateIndex !== exceptBufferViewIndex &&
      !scopedBufferViews.has(candidateIndex) &&
      candidate.buffer === bufferIndex &&
      getMeshoptBufferViewExtension(candidate) !== undefined,
  );
}

// ── EXT/KHR_meshopt_compression ──────────────────────────────────────────────

async function _resolveMeshopt(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  decode: MeshoptDecodeFn | undefined,
  required: ReadonlySet<string>,
  warnings: string[],
  onDiagnostic: GltfCompressionDiagnosticSink | undefined,
  scopedBufferViews: ReadonlySet<number> | undefined,
  allocationLedger: CompressionAllocationLedger,
  decodeState: CompressionDecodeState,
  resourceLedger: ImportResourceLedger | undefined,
  hookOutputPrecharged: boolean,
): Promise<void> {
  const views = gltf.bufferViews ?? [];
  for (let i = 0; i < views.length; i++) {
    if (scopedBufferViews !== undefined && !scopedBufferViews.has(i)) continue;
    const bv = views[i]!;
    const meshopt = getMeshoptBufferViewExtension(bv);
    if (!meshopt) continue;
    const { name } = meshopt;
    const extensionPath = `bufferViews[${i}].extensions.${name}`;
    const ext = validateMeshoptExtension(meshopt.value, extensionPath, name);
    validateDeclaredBufferRange(
      gltf,
      ext.buffer,
      ext.byteOffset,
      ext.byteLength,
      `${extensionPath} source`,
    );
    const parent = validateMeshoptParentLayout(gltf, i, name, ext);
    const isRequired = required.has(name);
    const expected = ext.count * ext.byteStride;
    const useFallback = (reason: string, code: GltfCompressionDiagnosticCode): void => {
      if (isRequired) {
        throw new Error(
          `[vitrum/gltf-adapter] ${name} is listed in extensionsRequired, so bufferView ${i} ` +
            `must decode successfully: ${reason}`,
        );
      }
      let fallback: ReturnType<typeof validateMeshoptFallback>;
      try {
        fallback = validateMeshoptFallback(gltf, buffers, i, name, ext);
      } catch (fallbackError) {
        throw new Error(
          `[vitrum/gltf-adapter] ${name} bufferView ${i} could not decode (${reason}) and has no fully valid uncompressed fallback: ${formatUnknownError(fallbackError)}`,
        );
      }
      emitCompressionDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code,
        path: extensionPath,
        extension: name,
        bufferViewIndex: i,
        message:
          `[vitrum/gltf-adapter] ${name} bufferView ${i} could not decode (${reason}). ` +
          'Using its fully validated uncompressed fallback buffer.',
      });
      // A fallback:true buffer can be shared by reachable and unreachable
      // compressed views. Stripping only this view's extension while keeping
      // it on that buffer would violate the fallback-buffer ownership rule;
      // clearing the shared marker would erase the declaration needed by the
      // unresolved sibling. Materialize just this fallback range instead.
      const hasUnresolvedReference =
        fallback.markedFallback && hasUnresolvedMeshoptReference(gltf, fallback.bufferIndex, i);
      const sharedWithOutOfScopeView =
        fallback.markedFallback &&
        hasUnresolvedMeshoptReferenceOutsideScope(gltf, fallback.bufferIndex, i, scopedBufferViews);
      if (sharedWithOutOfScopeView) {
        bv.buffer = _addSyntheticBuffer(
          gltf,
          buffers,
          fallback.bytes,
          allocationLedger,
          `${extensionPath} materialized fallback`,
          resourceLedger,
        );
        bv.byteOffset = 0;
        bv.byteLength = fallback.byteLength;
      }
      _stripExtension(bv, name);
      if (fallback.markedFallback && !hasUnresolvedReference) {
        clearMeshoptFallbackMarker(gltf, fallback.bufferIndex);
      }
    };

    if (!decode || decodeState.attemptsDisabled) {
      useFallback(
        decode === undefined
          ? 'no meshoptDecode hook was supplied'
          : 'compressed decode attempts were disabled after an earlier allocation-budget failure',
        decode === undefined
          ? 'meshopt-fallback-buffer-used'
          : 'meshopt-compression-budget-exceeded',
      );
      continue;
    }

    let compressed: Uint8Array;
    try {
      const sourceMarker = validateMeshoptBufferMarker(gltf, ext.buffer, name);
      if (sourceMarker?.fallback === true) {
        throw new Error(`compressed source buffer ${ext.buffer} is marked fallback-only`);
      }
      const src = buffers.get(ext.buffer);
      if (src == null) throw new Error(`compressed buffer ${ext.buffer} is unavailable`);
      compressed = checkedByteSlice(src, ext.byteOffset, ext.byteLength, `${extensionPath} source`);
    } catch (error) {
      useFallback(formatUnknownError(error), 'meshopt-buffer-unavailable');
      continue;
    }

    const headerFailure = meshoptHeaderFailure(name, ext, compressed);
    if (headerFailure !== undefined) {
      useFallback(headerFailure.reason, headerFailure.code);
      continue;
    }

    try {
      const totalCost = checkedCompressionSum(
        [ext.byteLength, expected, expected],
        `${extensionPath} decode allocation`,
      );
      allocationLedger.ensureAvailable(totalCost, `${extensionPath} decode`);
    } catch (error) {
      useFallback(formatUnknownError(error), 'meshopt-compression-budget-exceeded');
      continue;
    }

    const decodedGeometryCost = checkedCompressionSum(
      [expected, expected],
      `${extensionPath} decoded geometry allocation`,
    );
    resourceLedger?.ensureDecodedGeometryBytes(
      decodedGeometryCost,
      `${extensionPath} decoded output and retained copy`,
    );

    let candidate: unknown;
    try {
      allocationLedger.charge(ext.byteLength, `${extensionPath} compressed input copy`);
      candidate = await decode(
        new Uint8Array(compressed),
        ext.count,
        ext.byteStride,
        ext.mode,
        ext.filter,
      );
    } catch (error) {
      if (error instanceof GltfResourceLimitError) throw error;
      useFallback(formatUnknownError(error), 'meshopt-decode-hook-failed');
      continue;
    }
    const candidateInfo = compressionTypedArrayInfo(candidate);
    if (candidateInfo !== undefined) {
      if (!hookOutputPrecharged) {
        resourceLedger?.chargeDecodedGeometryBytes(
          candidateInfo.byteLength,
          `${extensionPath} hook output allocation`,
        );
      }
      try {
        chargeCompressionHookOutput(
          allocationLedger,
          decodeState,
          candidateInfo.byteLength,
          `${extensionPath} hook output allocation`,
        );
      } catch (error) {
        useFallback(formatUnknownError(error), 'meshopt-compression-budget-exceeded');
        continue;
      }
    }
    if (candidateInfo?.kind !== 'Uint8Array' || candidateInfo.shared) {
      useFallback(
        `meshoptDecode returned ${candidateInfo?.shared === true ? 'shared storage' : typeof candidate}, not a non-shared Uint8Array`,
        'meshopt-decode-hook-failed',
      );
      continue;
    }
    if (candidateInfo.byteLength !== expected) {
      useFallback(
        `meshoptDecode returned ${candidateInfo.byteLength} bytes; expected count × ` +
          `byteStride = ${ext.count} × ${ext.byteStride} = ${expected}`,
        'meshopt-decoded-byte-length-mismatch',
      );
      continue;
    }
    const decoded = candidate as Uint8Array;

    try {
      const bufIdx = _addSyntheticBuffer(
        gltf,
        buffers,
        decoded,
        allocationLedger,
        `${extensionPath} decoded output`,
        resourceLedger,
      );
      bv.buffer = bufIdx;
      bv.byteOffset = 0;
      bv.byteLength = expected;
      // Preserve the parent bufferView layout. In particular,
      // KHR_meshopt_compression permits its declared codec stride to differ from
      // the parent bufferView byteStride.
      _stripExtension(bv, name);
      const parentMarker = validateMeshoptBufferMarker(gltf, parent.bufferIndex, name);
      if (
        parentMarker?.fallback === true &&
        !hasUnresolvedMeshoptReference(gltf, parent.bufferIndex, i)
      ) {
        clearMeshoptFallbackMarker(gltf, parent.bufferIndex);
      }
    } catch (error) {
      if (error instanceof GltfResourceLimitError) throw error;
      useFallback(formatUnknownError(error), 'meshopt-decode-hook-failed');
    }
  }
}
