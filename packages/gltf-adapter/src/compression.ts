// compression.ts — KHR_draco_mesh_compression + EXT_meshopt_compression
// resolution via HOST-SUPPLIED decoder hooks (GLTF-02).
//
// The adapter stays dependency-free: it never bundles a Draco or meshoptimizer
// decoder. Instead the host injects decode functions through
// `GltfToSceneOptions.dracoDecode` / `.meshoptDecode`, and this module rewrites
// the (cloned) glTF JSON + buffers map so the rest of the pipeline sees plain
// uncompressed data:
//
//   - EXT_meshopt_compression sits on BUFFER VIEWS. Each compressed bufferView
//     is decoded into a synthetic buffer and the bufferView is repointed at it,
//     so ALL downstream consumers (attribute/index/animation accessors, image
//     bufferViews) transparently read decompressed bytes.
//   - KHR_draco_mesh_compression sits on MESH PRIMITIVES. The decoded typed
//     arrays are written into synthetic buffers/bufferViews and the primitive's
//     EXISTING accessors are repointed at them — accessor `count`, `type`,
//     `componentType` and `normalized` still describe the decoded data per
//     spec, so the standard accessor unpacking (incl. normalization) applies
//     unchanged.
//
// Failure-mode contract (honest, mirrors extensionsRequired semantics):
//   - extension in `extensionsRequired` + no hook (and no usable fallback)
//     → throw a clear Error (the asset cannot be represented without it).
//   - extension optional + no hook → use the spec-defined fallback when it
//     exists (Draco: uncompressed fallback accessors; meshopt: the
//     bufferView's own `buffer`, unless that buffer is a `fallback: true`
//     stub), else warn and leave the data unresolved (affected primitives are
//     skipped downstream with their own warnings).
//
// Async: `gltfToScene` is already async, so hooks may return their result
// either synchronously or as a Promise — both are awaited.
//
// References:
//   - KHR_draco_mesh_compression
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_draco_mesh_compression/README.md
//   - EXT_meshopt_compression
//     https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md

import type { GltfJson, GltfAccessor } from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';
import { componentByteSize, typeComponentCount } from './accessors.js';

const DRACO_EXT = 'KHR_draco_mesh_compression';
const MESHOPT_EXT = 'EXT_meshopt_compression';

// ── Public hook types ────────────────────────────────────────────────────────

/** Typed arrays a Draco decode hook may return per attribute / for indices. */
export type DracoTypedArray =
  | Int8Array | Uint8Array | Int16Array | Uint16Array | Uint32Array | Float32Array;

export interface DracoDecodeResult {
  /**
   * Decoded vertex attributes keyed by glTF semantic (POSITION, NORMAL,
   * TEXCOORD_0, …). Each array must either:
   *   - match the declared accessor's `componentType` exactly (the adapter
   *     then applies the accessor's `normalized` flag itself), or
   *   - be a `Float32Array` of already-dequantized values (accepted for any
   *     declared componentType; normalization is considered done).
   * Length must equal `accessor.count × components(accessor.type)`.
   */
  readonly attributes: Readonly<Record<string, DracoTypedArray>>;
  /** Decoded triangle indices (length must equal the index accessor's count). */
  readonly indices?: Uint8Array | Uint16Array | Uint32Array;
}

/**
 * Host-supplied Draco decode hook (KHR_draco_mesh_compression).
 *
 * @param compressed   - The raw compressed blob (the extension's bufferView).
 * @param attributeIds - The extension's `attributes` map: glTF semantic →
 *                       Draco attribute unique id (pass each id to
 *                       `decoder.GetAttributeByUniqueId`).
 * May return synchronously or as a Promise.
 */
export type DracoDecodeFn = (
  compressed: Uint8Array,
  attributeIds: Readonly<Record<string, number>>,
) => DracoDecodeResult | Promise<DracoDecodeResult>;

/** EXT_meshopt_compression `mode` values. */
export type MeshoptMode = 'ATTRIBUTES' | 'TRIANGLES' | 'INDICES';
/** EXT_meshopt_compression `filter` values. */
export type MeshoptFilter = 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL';

/**
 * Host-supplied meshopt decode hook (EXT_meshopt_compression). The signature
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

interface DracoPrimitiveExt {
  bufferView: number;
  attributes: Record<string, number>;
}

// ── Entry point ──────────────────────────────────────────────────────────────

/**
 * Resolve all compressed geometry in `gltf` using the supplied hooks.
 *
 * Returns the input `gltf` untouched when no compression extensions are in
 * use; otherwise returns a CLONE with compressed bufferViews / primitive
 * accessors rewritten to synthetic uncompressed buffers (added to `buffers`).
 * The caller's `gltf` object is never mutated.
 *
 * @throws when an extension is listed in `extensionsRequired`, no hook is
 *         supplied, and no spec-defined fallback data exists (or required
 *         decode fails).
 */
export async function resolveCompression(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  hooks: GltfDecodeHooks,
  warnings: string[],
): Promise<GltfJson> {
  const hasMeshopt = (gltf.bufferViews ?? []).some(
    (bv) => bv.extensions?.[MESHOPT_EXT] !== undefined,
  );
  const hasDraco = (gltf.meshes ?? []).some((m) =>
    m.primitives.some((p) => p.extensions?.[DRACO_EXT] !== undefined),
  );
  if (!hasMeshopt && !hasDraco) return gltf;

  // Copy-on-write: clone the JSON so the caller's object is never mutated.
  const out = structuredClone(gltf) as GltfJson;
  const required = new Set(out.extensionsRequired ?? []);

  // meshopt first: it operates at bufferView level, so a (theoretical) Draco
  // blob inside a meshopt-wrapped view would already be decompressed.
  if (hasMeshopt) {
    await _resolveMeshopt(out, buffers, hooks.meshoptDecode, required.has(MESHOPT_EXT), warnings);
  }
  if (hasDraco) {
    await _resolveDraco(out, buffers, hooks.dracoDecode, required.has(DRACO_EXT), warnings);
  }
  return out;
}

// ── Synthetic buffer plumbing ────────────────────────────────────────────────

function _nextBufferIndex(gltf: GltfJson, buffers: Map<number, ArrayBuffer>): number {
  let next = gltf.buffers?.length ?? 0;
  for (const k of buffers.keys()) if (k >= next) next = k + 1;
  return next;
}

/** Copy `bytes` into a fresh buffer, register it, and return its index. */
function _addSyntheticBuffer(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  bytes: Uint8Array,
): number {
  const idx = _nextBufferIndex(gltf, buffers);
  const copy = new Uint8Array(bytes); // exact-size copy, detached from source
  buffers.set(idx, copy.buffer);
  // Keep gltf.buffers index-consistent (pad any gap with zero-length stubs).
  const list = (gltf.buffers ??= []);
  while (list.length < idx) list.push({ byteLength: 0 });
  list.push({ byteLength: copy.byteLength });
  return idx;
}

/** Register a tightly-packed synthetic bufferView over `bytes`; returns its index. */
function _addSyntheticBufferView(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  bytes: Uint8Array,
): number {
  const bufIdx = _addSyntheticBuffer(gltf, buffers, bytes);
  const views = (gltf.bufferViews ??= []);
  views.push({ buffer: bufIdx, byteOffset: 0, byteLength: bytes.byteLength });
  return views.length - 1;
}

function _viewBytes(a: ArrayBufferView): Uint8Array {
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

function _typedArrayComponentType(a: ArrayBufferView): GltfComponentType | undefined {
  if (a instanceof Int8Array) return GltfComponentType.BYTE;
  if (a instanceof Uint8Array) return GltfComponentType.UNSIGNED_BYTE;
  if (a instanceof Int16Array) return GltfComponentType.SHORT;
  if (a instanceof Uint16Array) return GltfComponentType.UNSIGNED_SHORT;
  if (a instanceof Uint32Array) return GltfComponentType.UNSIGNED_INT;
  if (a instanceof Float32Array) return GltfComponentType.FLOAT;
  return undefined;
}

function _stripExtension(holder: { extensions?: Record<string, unknown> }, name: string): void {
  if (!holder.extensions) return;
  delete holder.extensions[name];
  if (Object.keys(holder.extensions).length === 0) delete holder.extensions;
}

// ── EXT_meshopt_compression ──────────────────────────────────────────────────

async function _resolveMeshopt(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  decode: MeshoptDecodeFn | undefined,
  isRequired: boolean,
  warnings: string[],
): Promise<void> {
  const views = gltf.bufferViews ?? [];
  for (let i = 0; i < views.length; i++) {
    const bv = views[i]!;
    const ext = bv.extensions?.[MESHOPT_EXT] as MeshoptBufferViewExt | undefined;
    if (!ext) continue;

    if (!decode) {
      // Spec fallback: the bufferView's OWN buffer holds uncompressed data,
      // unless that buffer is a `fallback: true` stub (no real payload).
      const fallbackStub =
        (gltf.buffers?.[bv.buffer]?.extensions?.[MESHOPT_EXT] as
          | { fallback?: boolean }
          | undefined)?.fallback === true;
      const fallbackAvailable = !fallbackStub && buffers.has(bv.buffer);
      if (fallbackAvailable) {
        warnings.push(
          `[vitrum/gltf-adapter] BufferView ${i} uses EXT_meshopt_compression but no ` +
            'opts.meshoptDecode hook was supplied. Falling back to the uncompressed ' +
            'fallback buffer (larger download, identical data).',
        );
        _stripExtension(bv, MESHOPT_EXT);
        continue;
      }
      if (isRequired) {
        throw new Error(
          '[vitrum/gltf-adapter] EXT_meshopt_compression is listed in extensionsRequired ' +
            `but no opts.meshoptDecode hook was supplied and bufferView ${i} has no ` +
            'uncompressed fallback buffer. Supply a decode hook (e.g. meshoptimizer’s ' +
            'MeshoptDecoder.decodeGltfBuffer — see the README "Compressed geometry" section).',
        );
      }
      warnings.push(
        `[vitrum/gltf-adapter] BufferView ${i} uses EXT_meshopt_compression with no ` +
          'opts.meshoptDecode hook and no uncompressed fallback buffer. Dependent ' +
          'accessors cannot be read; affected primitives will be skipped.',
      );
      continue;
    }

    const src = buffers.get(ext.buffer);
    if (!src) {
      const msg =
        `[vitrum/gltf-adapter] EXT_meshopt_compression bufferView ${i} references ` +
        `buffer ${ext.buffer} which is not available (supply it via opts.buffers).`;
      if (isRequired) throw new Error(msg);
      warnings.push(msg + ' BufferView left unresolved.');
      continue;
    }

    const compressed = new Uint8Array(src, ext.byteOffset ?? 0, ext.byteLength);
    let decoded: Uint8Array;
    try {
      decoded = await decode(compressed, ext.count, ext.byteStride, ext.mode, ext.filter ?? 'NONE');
    } catch (e) {
      const msg =
        `[vitrum/gltf-adapter] meshoptDecode hook failed for bufferView ${i} ` +
        `(mode=${ext.mode}, filter=${ext.filter ?? 'NONE'}): ${String(e)}`;
      if (isRequired) throw new Error(msg);
      warnings.push(msg + ' BufferView left unresolved.');
      continue;
    }

    const expected = ext.count * ext.byteStride;
    if (decoded.byteLength !== expected) {
      const msg =
        `[vitrum/gltf-adapter] meshoptDecode hook returned ${decoded.byteLength} bytes for ` +
        `bufferView ${i}; expected count × byteStride = ${ext.count} × ${ext.byteStride} = ${expected}.`;
      if (isRequired) throw new Error(msg);
      warnings.push(msg + ' BufferView left unresolved.');
      continue;
    }

    const bufIdx = _addSyntheticBuffer(gltf, buffers, decoded);
    bv.buffer = bufIdx;
    bv.byteOffset = 0;
    bv.byteLength = expected;
    // Decoded ATTRIBUTES data is interleaved at the extension's byteStride;
    // TRIANGLES / INDICES data is tightly packed scalar elements.
    if (ext.mode === 'ATTRIBUTES') bv.byteStride = ext.byteStride;
    else delete bv.byteStride;
    _stripExtension(bv, MESHOPT_EXT);
  }
}

// ── KHR_draco_mesh_compression ───────────────────────────────────────────────

async function _resolveDraco(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  decode: DracoDecodeFn | undefined,
  isRequired: boolean,
  warnings: string[],
): Promise<void> {
  const meshes = gltf.meshes ?? [];
  for (let mi = 0; mi < meshes.length; mi++) {
    const mesh = meshes[mi]!;
    const label = mesh.name ?? mi;
    for (const prim of mesh.primitives) {
      const ext = prim.extensions?.[DRACO_EXT] as DracoPrimitiveExt | undefined;
      if (!ext) continue;

      if (!decode) {
        _handleDracoNoHook(gltf, prim, label, isRequired, warnings);
        continue;
      }

      // Read the compressed blob.
      const bv = gltf.bufferViews?.[ext.bufferView];
      const buf = bv ? buffers.get(bv.buffer) : undefined;
      if (!bv || !buf) {
        const msg =
          `[vitrum/gltf-adapter] KHR_draco_mesh_compression on mesh "${label}" references ` +
          `bufferView ${ext.bufferView} whose data is not available.`;
        if (isRequired) throw new Error(msg);
        warnings.push(msg + ' Primitive left unresolved (will be skipped).');
        continue;
      }
      const compressed = new Uint8Array(buf, bv.byteOffset ?? 0, bv.byteLength);

      let result: DracoDecodeResult;
      try {
        result = await decode(compressed, ext.attributes);
      } catch (e) {
        const msg =
          `[vitrum/gltf-adapter] dracoDecode hook failed for mesh "${label}": ${String(e)}`;
        if (isRequired) throw new Error(msg);
        warnings.push(msg + ' Primitive left unresolved (will be skipped).');
        continue;
      }

      let failed = false;

      // Attributes: repoint each declared accessor at the decoded data. The
      // accessor keeps describing the decoded data (count/type/componentType/
      // normalized per spec), so unpackAccessorFloat applies normalization etc.
      for (const semantic of Object.keys(ext.attributes)) {
        const accIdx = prim.attributes[semantic];
        if (accIdx === undefined) {
          warnings.push(
            `[vitrum/gltf-adapter] Draco extension on mesh "${label}" declares attribute ` +
              `"${semantic}" with no matching primitive attribute. Ignored.`,
          );
          continue;
        }
        const acc = gltf.accessors?.[accIdx];
        if (!acc) {
          warnings.push(
            `[vitrum/gltf-adapter] Draco attribute "${semantic}" on mesh "${label}" ` +
              `references missing accessor ${accIdx}. Ignored.`,
          );
          if (semantic === 'POSITION') failed = true;
          continue;
        }
        const arr = result.attributes[semantic];
        if (!arr) {
          warnings.push(
            `[vitrum/gltf-adapter] dracoDecode hook did not return attribute "${semantic}" ` +
              `for mesh "${label}".` +
              (acc.bufferView !== undefined
                ? ' Using the accessor’s uncompressed fallback data.'
                : ''),
          );
          if (semantic === 'POSITION' && acc.bufferView === undefined) failed = true;
          continue;
        }
        if (!_rewriteDracoAccessor(gltf, buffers, acc, arr, semantic, label, warnings)) {
          if (semantic === 'POSITION') failed = true;
        }
      }

      // Indices.
      if (prim.indices !== undefined) {
        const idxAcc = gltf.accessors?.[prim.indices];
        if (idxAcc) {
          if (!result.indices) {
            warnings.push(
              `[vitrum/gltf-adapter] dracoDecode hook did not return indices for mesh ` +
                `"${label}".` +
                (idxAcc.bufferView !== undefined
                  ? ' Using the accessor’s uncompressed fallback data.'
                  : ' The index accessor has no fallback; primitive will be skipped.'),
            );
            if (idxAcc.bufferView === undefined) failed = true;
          } else if (result.indices.length !== idxAcc.count) {
            warnings.push(
              `[vitrum/gltf-adapter] dracoDecode hook returned ${result.indices.length} indices ` +
                `for mesh "${label}"; the index accessor declares count=${idxAcc.count}. ` +
                'Indices rejected.',
            );
            if (idxAcc.bufferView === undefined) failed = true;
          } else {
            // Re-encode into the accessor's declared componentType so the
            // standard index unpacking path applies unchanged.
            const ct = idxAcc.componentType;
            const typed =
              ct === GltfComponentType.UNSIGNED_BYTE ? new Uint8Array(result.indices)
              : ct === GltfComponentType.UNSIGNED_SHORT ? new Uint16Array(result.indices)
              : new Uint32Array(result.indices);
            if (ct !== GltfComponentType.UNSIGNED_BYTE &&
                ct !== GltfComponentType.UNSIGNED_SHORT &&
                ct !== GltfComponentType.UNSIGNED_INT) {
              idxAcc.componentType = GltfComponentType.UNSIGNED_INT;
            }
            idxAcc.bufferView = _addSyntheticBufferView(gltf, buffers, _viewBytes(typed));
            idxAcc.byteOffset = 0;
          }
        }
      }

      if (failed) {
        warnings.push(
          `[vitrum/gltf-adapter] Draco decode for mesh "${label}" did not yield usable ` +
            'POSITION/index data. Primitive left unresolved (will be skipped).',
        );
        if (isRequired) {
          throw new Error(
            `[vitrum/gltf-adapter] KHR_draco_mesh_compression is listed in extensionsRequired ` +
              `but the decode of mesh "${label}" did not yield usable geometry (see warnings).`,
          );
        }
        continue; // keep the extension marker → downstream skip
      }
      _stripExtension(prim, DRACO_EXT);
    }
  }
}

/**
 * No-hook handling for a Draco primitive: use the spec fallback (uncompressed
 * accessors, valid when the extension is NOT in extensionsRequired and every
 * referenced accessor carries a bufferView), throw when required, else warn
 * and leave the primitive unresolved.
 */
function _handleDracoNoHook(
  gltf: GltfJson,
  prim: { attributes: Record<string, number | undefined>; indices?: number; extensions?: Record<string, unknown> },
  label: string | number,
  isRequired: boolean,
  warnings: string[],
): void {
  const accessorIndices = [
    ...Object.values(prim.attributes).filter((v): v is number => v !== undefined),
    ...(prim.indices !== undefined ? [prim.indices] : []),
  ];
  const hasFallback =
    accessorIndices.length > 0 &&
    accessorIndices.every((ai) => gltf.accessors?.[ai]?.bufferView !== undefined);

  if (isRequired) {
    throw new Error(
      '[vitrum/gltf-adapter] KHR_draco_mesh_compression is listed in extensionsRequired ' +
        `but no opts.dracoDecode hook was supplied for mesh "${label}". ` +
        (hasFallback
          ? 'The primitive has uncompressed fallback accessors, but required Draco assets must decode the required extension. '
          : 'The primitive also has no uncompressed fallback accessors. ') +
        'Supply a decode hook (e.g. via draco3d — see the README ' +
        '"Compressed geometry" section).',
    );
  }
  if (hasFallback) {
    warnings.push(
      `[vitrum/gltf-adapter] Mesh "${label}" uses KHR_draco_mesh_compression but no ` +
        'opts.dracoDecode hook was supplied. Using the primitive’s uncompressed ' +
        'fallback accessors.',
    );
    _stripExtension(prim, DRACO_EXT);
    return;
  }
  warnings.push(
    `[vitrum/gltf-adapter] Mesh "${label}" uses KHR_draco_mesh_compression with no ` +
      'opts.dracoDecode hook and no uncompressed fallback accessors. Primitive will be skipped.',
  );
}

/**
 * Point `acc` at a synthetic bufferView holding the decoded attribute data.
 * Accepts arrays matching the accessor's componentType (normalization stays
 * with the standard unpack path) or already-dequantized Float32Arrays.
 * Returns false (with a warning) when the array is unusable.
 */
function _rewriteDracoAccessor(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  acc: GltfAccessor,
  arr: DracoTypedArray,
  semantic: string,
  label: string | number,
  warnings: string[],
): boolean {
  const comps = typeComponentCount(acc.type);
  const expectedElems = acc.count * comps;
  if (arr.length !== expectedElems) {
    warnings.push(
      `[vitrum/gltf-adapter] dracoDecode returned ${arr.length} elements for "${semantic}" ` +
        `on mesh "${label}"; the accessor declares count × components = ` +
        `${acc.count} × ${comps} = ${expectedElems}. Attribute rejected.`,
    );
    return false;
  }

  const arrCt = _typedArrayComponentType(arr);
  if (arrCt === acc.componentType) {
    // Exact match: the accessor's normalized flag is applied by the standard
    // unpack path. Sanity-check byte size (guaranteed by the element check).
    void componentByteSize(acc.componentType);
  } else if (arr instanceof Float32Array) {
    // Decoder dequantized for us — describe the data as raw floats.
    acc.componentType = GltfComponentType.FLOAT;
    acc.normalized = false;
  } else {
    warnings.push(
      `[vitrum/gltf-adapter] dracoDecode returned a ${arr.constructor.name} for "${semantic}" ` +
        `on mesh "${label}" but the accessor declares componentType ${acc.componentType}. ` +
        'Return an array matching the accessor componentType, or a dequantized ' +
        'Float32Array. Attribute rejected.',
    );
    return false;
  }

  acc.bufferView = _addSyntheticBufferView(gltf, buffers, _viewBytes(arr));
  acc.byteOffset = 0;
  return true;
}
