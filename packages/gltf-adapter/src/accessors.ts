// accessors.ts — glTF accessor unpacking to typed arrays.
//
// Reads component-typed, optionally-normalized data from bufferViews.
// Sparse accessors are supported: indices + values patches are applied on top
// of the base (or zero) data. This is required for correctness since
// KHR_materials_* test assets use sparse normals.
//
// Reference: glTF 2.0 spec §3.6.2 (Accessors)

import type { GltfJson, GltfAccessor } from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';

/** Number of scalar elements per accessor type. */
const TYPE_COMPONENT_COUNT: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

/**
 * Number of scalar components for an accessor `type` string.
 * Exported for compression.ts (Draco decoded-attribute validation).
 */
export function typeComponentCount(type: string): number {
  const n = TYPE_COMPONENT_COUNT[type];
  if (n === undefined) {
    throw new Error(`[vitrum/gltf-adapter] Unknown accessor type "${type}"`);
  }
  return n;
}

/** Byte size of each component type. Exported for compression.ts. */
export function componentByteSize(ct: GltfComponentType): number {
  switch (ct) {
    case GltfComponentType.BYTE:
    case GltfComponentType.UNSIGNED_BYTE:
      return 1;
    case GltfComponentType.SHORT:
    case GltfComponentType.UNSIGNED_SHORT:
      return 2;
    case GltfComponentType.UNSIGNED_INT:
    case GltfComponentType.FLOAT:
      return 4;
    default:
      throw new Error(`[vitrum/gltf-adapter] Unknown componentType ${String(ct)}`);
  }
}

/**
 * Read a single scalar element from a DataView at the given byte offset,
 * applying normalization if requested.
 */
function readScalar(
  view: DataView,
  byteOffset: number,
  ct: GltfComponentType,
  normalized: boolean,
): number {
  switch (ct) {
    case GltfComponentType.BYTE: {
      const v = view.getInt8(byteOffset);
      return normalized ? Math.max(v / 127, -1) : v;
    }
    case GltfComponentType.UNSIGNED_BYTE: {
      const v = view.getUint8(byteOffset);
      return normalized ? v / 255 : v;
    }
    case GltfComponentType.SHORT: {
      const v = view.getInt16(byteOffset, true);
      return normalized ? Math.max(v / 32767, -1) : v;
    }
    case GltfComponentType.UNSIGNED_SHORT: {
      const v = view.getUint16(byteOffset, true);
      return normalized ? v / 65535 : v;
    }
    case GltfComponentType.UNSIGNED_INT:
      return view.getUint32(byteOffset, true);
    case GltfComponentType.FLOAT:
      return view.getFloat32(byteOffset, true);
    default:
      throw new Error(`[vitrum/gltf-adapter] Unknown componentType ${String(ct)}`);
  }
}

/**
 * Unpack a glTF accessor into a Float32Array (all component types are converted
 * to float32, applying normalization as required).
 *
 * Returns a Float32Array of length `accessor.count * componentsPerElement`.
 */
export function unpackAccessorFloat(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  warnings: string[],
): Float32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);
  }

  const componentCount = TYPE_COMPONENT_COUNT[accessor.type];
  if (componentCount === undefined) {
    throw new Error(`[vitrum/gltf-adapter] Unknown accessor type "${accessor.type}"`);
  }
  const total = accessor.count * componentCount;
  const result = new Float32Array(total);

  if (accessor.bufferView !== undefined) {
    _readBufferViewIntoResult(gltf, buffers, accessor, componentCount, result);
  }
  // If bufferView is absent, result stays zero-initialized (valid per spec for sparse).

  if (accessor.sparse) {
    _applySparsePatch(gltf, buffers, accessor, componentCount, result, warnings);
  }

  return result;
}

function _readBufferViewIntoResult(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessor: GltfAccessor,
  componentCount: number,
  result: Float32Array,
): void {
  const bvIdx = accessor.bufferView!;
  const bv = gltf.bufferViews?.[bvIdx];
  if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`);

  const buf = _getBuffer(buffers, bv.buffer, gltf);
  const ct = accessor.componentType;
  const compSize = componentByteSize(ct);
  const normalized = accessor.normalized ?? false;

  const bvOffset = bv.byteOffset ?? 0;
  const accOffset = accessor.byteOffset ?? 0;
  const stride = bv.byteStride ?? compSize * componentCount;

  const dataView = new DataView(buf, bvOffset + accOffset);

  for (let i = 0; i < accessor.count; i++) {
    const elemOffset = i * stride;
    for (let c = 0; c < componentCount; c++) {
      result[i * componentCount + c] = readScalar(
        dataView,
        elemOffset + c * compSize,
        ct,
        normalized,
      );
    }
  }
}

interface SparseViews {
  idxView: DataView;
  idxCt: GltfComponentType;
  idxCompSize: number;
  valView: DataView;
  valCt: GltfComponentType;
  valCompSize: number;
  count: number;
}

/**
 * Resolve the DataViews and component metadata for an accessor's sparse patch.
 * Returns `null` (with a warning) if the required bufferViews or buffers are
 * unavailable, which allows callers to degrade gracefully.
 */
function _resolveSparseViews(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessor: GltfAccessor,
  warnings: string[] | null,
): SparseViews | null {
  const sparse = accessor.sparse!;

  const idxBv = gltf.bufferViews?.[sparse.indices.bufferView];
  if (!idxBv) {
    if (warnings) warnings.push('[vitrum/gltf-adapter] Sparse indices bufferView not found; patch skipped.');
    else throw new Error('[vitrum/gltf-adapter] Sparse indices bufferView not found');
    return null;
  }
  const idxBuf = buffers.get(idxBv.buffer);
  if (!idxBuf) {
    if (warnings) warnings.push(`[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable; patch skipped.`);
    else throw new Error(`[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable`);
    return null;
  }

  const valBv = gltf.bufferViews?.[sparse.values.bufferView];
  if (!valBv) {
    if (warnings) warnings.push('[vitrum/gltf-adapter] Sparse values bufferView not found; patch skipped.');
    else throw new Error('[vitrum/gltf-adapter] Sparse values bufferView not found');
    return null;
  }
  const valBuf = buffers.get(valBv.buffer);
  if (!valBuf) {
    if (warnings) warnings.push(`[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable; patch skipped.`);
    else throw new Error(`[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable`);
    return null;
  }

  const idxCt = sparse.indices.componentType;
  const idxCompSize = componentByteSize(idxCt);
  const valCt = accessor.componentType;
  const valCompSize = componentByteSize(valCt);

  return {
    idxView: new DataView(idxBuf, (idxBv.byteOffset ?? 0) + (sparse.indices.byteOffset ?? 0)),
    idxCt,
    idxCompSize,
    valView: new DataView(valBuf, (valBv.byteOffset ?? 0) + (sparse.values.byteOffset ?? 0)),
    valCt,
    valCompSize,
    count: sparse.count,
  };
}

function _applySparsePatch(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessor: GltfAccessor,
  componentCount: number,
  result: Float32Array,
  warnings: string[],
): void {
  warnings.push(
    `[vitrum/gltf-adapter] Accessor uses sparse storage (count=${accessor.sparse!.count}); applying patch.`,
  );

  const sv = _resolveSparseViews(gltf, buffers, accessor, warnings);
  if (!sv) return;

  const normalized = accessor.normalized ?? false;
  for (let s = 0; s < sv.count; s++) {
    const idx = Math.round(readScalar(sv.idxView, s * sv.idxCompSize, sv.idxCt, false));
    for (let c = 0; c < componentCount; c++) {
      result[idx * componentCount + c] = readScalar(
        sv.valView,
        (s * componentCount + c) * sv.valCompSize,
        sv.valCt,
        normalized,
      );
    }
  }
}

/**
 * Unpack a glTF accessor into a Uint32Array. Used for index buffers.
 * Accepts UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT.
 */
export function unpackAccessorUint32(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
): Uint32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);
  }
  if (accessor.type !== 'SCALAR') {
    throw new Error(
      `[vitrum/gltf-adapter] Index accessor must be SCALAR, got "${accessor.type}"`,
    );
  }

  const ct = accessor.componentType;
  if (
    ct !== GltfComponentType.UNSIGNED_BYTE &&
    ct !== GltfComponentType.UNSIGNED_SHORT &&
    ct !== GltfComponentType.UNSIGNED_INT
  ) {
    throw new Error(
      `[vitrum/gltf-adapter] Index accessor componentType ${ct} is not an unsigned integer type`,
    );
  }

  const result = new Uint32Array(accessor.count);

  if (accessor.bufferView === undefined) {
    // Zero-initialized — valid for pure-sparse, but unusual for indices.
    return result;
  }

  const bvIdx = accessor.bufferView;
  const bv = gltf.bufferViews?.[bvIdx];
  if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`);

  const buf = _getBuffer(buffers, bv.buffer, gltf);
  const compSize = componentByteSize(ct);
  const bvOffset = bv.byteOffset ?? 0;
  const accOffset = accessor.byteOffset ?? 0;
  const stride = bv.byteStride ?? compSize;
  const dataView = new DataView(buf, bvOffset + accOffset);

  for (let i = 0; i < accessor.count; i++) {
    result[i] = Math.round(readScalar(dataView, i * stride, ct, false));
  }

  if (accessor.sparse) {
    // Sparse integer index buffers are legal but extremely rare.
    // Uses null warnings → missing bufferViews silently skip (graceful degrade).
    const sv = _resolveSparseViews(gltf, buffers, accessor, null);
    if (sv) {
      for (let s = 0; s < sv.count; s++) {
        const idx = Math.round(readScalar(sv.idxView, s * sv.idxCompSize, sv.idxCt, false));
        result[idx] = Math.round(readScalar(sv.valView, s * sv.valCompSize, sv.valCt, false));
      }
    }
  }

  return result;
}

function _getBuffer(
  buffers: Map<number, ArrayBuffer>,
  bufferIndex: number,
  _gltf: GltfJson,
): ArrayBuffer {
  const buf = buffers.get(bufferIndex);
  if (buf === undefined) {
    throw new Error(
      `[vitrum/gltf-adapter] Buffer ${bufferIndex} is not available. ` +
        'When loading a .gltf file with external buffers, supply pre-fetched ArrayBuffers ' +
        'via opts.buffers. The adapter does not fetch URIs.',
    );
  }
  return buf;
}
