// accessors.ts — glTF accessor unpacking to typed arrays.
//
// Reads component-typed, optionally-normalized data from bufferViews.
// Sparse accessors are supported: indices + values patches are applied on top
// of the base (or zero) data. This is required for correctness since
// KHR_materials_* test assets use sparse normals.
//
// Reference: glTF 2.0 spec §3.6.2 (Accessors)

import type { GltfJson, GltfAccessor, GltfBufferView } from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';

export type GltfAccessorDiagnosticCode =
  | 'sparse-accessor-applied'
  | 'sparse-indices-buffer-view-not-found'
  | 'sparse-indices-buffer-unavailable'
  | 'sparse-values-buffer-view-not-found'
  | 'sparse-values-buffer-unavailable'
  | 'sparse-indices-buffer-view-truncated'
  | 'sparse-values-buffer-view-truncated'
  | 'invalid-sparse-indices-component-type'
  | 'sparse-index-out-of-range';

export interface GltfAccessorDiagnostic {
  readonly severity: 'warning';
  readonly code: GltfAccessorDiagnosticCode;
  readonly path: string;
  readonly message: string;
  readonly accessorIndex: number;
  readonly sparseEntryIndex?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly componentType?: number;
}

export type GltfAccessorDiagnosticSink = (diagnostic: GltfAccessorDiagnostic) => void;

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

export interface GltfAccessorBufferViewRange {
  readonly byteOffset: number;
  readonly byteStride: number;
  readonly elementByteLength: number;
  readonly requiredByteLength: number;
}

export function accessorBufferViewRange(
  accessor: GltfAccessor,
  bufferView?: Pick<GltfBufferView, 'byteStride'>,
  componentCount = typeComponentCount(accessor.type),
): GltfAccessorBufferViewRange {
  const compSize = componentByteSize(accessor.componentType);
  const elementByteLength = compSize * componentCount;
  const byteStride = bufferView?.byteStride ?? elementByteLength;
  const byteOffset = accessor.byteOffset ?? 0;
  const requiredByteLength = byteOffset + (accessor.count <= 0
    ? 0
    : (accessor.count - 1) * byteStride + elementByteLength);
  return {
    byteOffset,
    byteStride,
    elementByteLength,
    requiredByteLength,
  };
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
  onDiagnostic?: GltfAccessorDiagnosticSink,
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
    _applySparsePatch(gltf, buffers, accessorIndex, accessor, componentCount, result, warnings, onDiagnostic);
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
  const range = accessorBufferViewRange(accessor, bv, componentCount);
  validateBufferViewAccess(buf, bvIdx, bv, range.requiredByteLength, 'accessor');

  const dataView = new DataView(buf, bvOffset, bv.byteLength);

  for (let i = 0; i < accessor.count; i++) {
    const elemOffset = range.byteOffset + i * range.byteStride;
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
  idxByteOffset: number;
  idxCt: GltfComponentType;
  idxCompSize: number;
  valView: DataView;
  valByteOffset: number;
  valCt: GltfComponentType;
  valCompSize: number;
  count: number;
}

function _isSparseIndexComponentType(ct: GltfComponentType): boolean {
  return (
    ct === GltfComponentType.UNSIGNED_BYTE ||
    ct === GltfComponentType.UNSIGNED_SHORT ||
    ct === GltfComponentType.UNSIGNED_INT
  );
}

/**
 * Resolve the DataViews and component metadata for an accessor's sparse patch.
 * Returns `null` (with a warning) if the required bufferViews or buffers are
 * unavailable, which allows callers to degrade gracefully.
 */
function _resolveSparseViews(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  accessor: GltfAccessor,
  componentCount: number,
  warnings: string[] | null,
  onDiagnostic: GltfAccessorDiagnosticSink | undefined,
): SparseViews | null {
  const sparse = accessor.sparse!;

  const idxBv = gltf.bufferViews?.[sparse.indices.bufferView];
  if (!idxBv) {
    const message = '[vitrum/gltf-adapter] Sparse indices bufferView not found; patch skipped.';
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-indices-buffer-view-not-found',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
    });
    if (!warnings) throw new Error('[vitrum/gltf-adapter] Sparse indices bufferView not found');
    return null;
  }
  const idxBuf = buffers.get(idxBv.buffer);
  if (!idxBuf) {
    const message = `[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable; patch skipped.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-indices-buffer-unavailable',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
      bufferIndex: idxBv.buffer,
    });
    if (!warnings) throw new Error(`[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable`);
    return null;
  }

  const valBv = gltf.bufferViews?.[sparse.values.bufferView];
  if (!valBv) {
    const message = '[vitrum/gltf-adapter] Sparse values bufferView not found; patch skipped.';
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-values-buffer-view-not-found',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
    });
    if (!warnings) throw new Error('[vitrum/gltf-adapter] Sparse values bufferView not found');
    return null;
  }
  const valBuf = buffers.get(valBv.buffer);
  if (!valBuf) {
    const message = `[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable; patch skipped.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-values-buffer-unavailable',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
      bufferIndex: valBv.buffer,
    });
    if (!warnings) throw new Error(`[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable`);
    return null;
  }

  const idxCt = sparse.indices.componentType;
  if (!_isSparseIndexComponentType(idxCt)) {
    const message =
      `[vitrum/gltf-adapter] Sparse indices componentType ${idxCt} is invalid; ` +
      'expected UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT.';
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'invalid-sparse-indices-component-type',
      path: `accessors[${accessorIndex}].sparse.indices.componentType`,
      message: `${message} Patch skipped.`,
      accessorIndex,
      componentType: idxCt,
    });
    if (!warnings) throw new Error(message);
    return null;
  }
  const idxCompSize = componentByteSize(idxCt);
  const valCt = accessor.componentType;
  const valCompSize = componentByteSize(valCt);
  const idxByteOffset = sparse.indices.byteOffset ?? 0;
  const valByteOffset = sparse.values.byteOffset ?? 0;
  try {
    validateBufferViewAccess(
      idxBuf,
      sparse.indices.bufferView,
      idxBv,
      idxByteOffset + sparse.count * idxCompSize,
      'sparse indices',
    );
  } catch (error) {
    const message = `${String(error)} Patch skipped.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-indices-buffer-view-truncated',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
    });
    if (!warnings) throw error;
    return null;
  }
  try {
    validateBufferViewAccess(
      valBuf,
      sparse.values.bufferView,
      valBv,
      valByteOffset + sparse.count * componentCount * valCompSize,
      'sparse values',
    );
  } catch (error) {
    const message = `${String(error)} Patch skipped.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-values-buffer-view-truncated',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
    });
    if (!warnings) throw error;
    return null;
  }

  return {
    idxView: new DataView(idxBuf, idxBv.byteOffset ?? 0, idxBv.byteLength),
    idxByteOffset,
    idxCt,
    idxCompSize,
    valView: new DataView(valBuf, valBv.byteOffset ?? 0, valBv.byteLength),
    valByteOffset,
    valCt,
    valCompSize,
    count: sparse.count,
  };
}

function _applySparsePatch(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  accessor: GltfAccessor,
  componentCount: number,
  result: Float32Array,
  warnings: string[],
  onDiagnostic: GltfAccessorDiagnosticSink | undefined,
): void {
  emitAccessorDiagnostic(warnings, onDiagnostic, {
    severity: 'warning',
    code: 'sparse-accessor-applied',
    path: `accessors[${accessorIndex}].sparse`,
    message: `[vitrum/gltf-adapter] Accessor uses sparse storage (count=${accessor.sparse!.count}); applying patch.`,
    accessorIndex,
  });

  const sv = _resolveSparseViews(gltf, buffers, accessorIndex, accessor, componentCount, warnings, onDiagnostic);
  if (!sv) return;

  const normalized = accessor.normalized ?? false;
  for (let s = 0; s < sv.count; s++) {
    const idx = Math.round(readScalar(sv.idxView, sv.idxByteOffset + s * sv.idxCompSize, sv.idxCt, false));
    if (idx < 0 || idx >= accessor.count) {
      emitAccessorDiagnostic(warnings, onDiagnostic, {
        severity: 'warning',
        code: 'sparse-index-out-of-range',
        path: `accessors[${accessorIndex}].sparse.indices[${s}]`,
        message: `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}; patch entry skipped.`,
        accessorIndex,
        sparseEntryIndex: s,
      });
      continue;
    }
    for (let c = 0; c < componentCount; c++) {
      result[idx * componentCount + c] = readScalar(
        sv.valView,
        sv.valByteOffset + (s * componentCount + c) * sv.valCompSize,
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
  warnings: string[] | null = null,
  onDiagnostic?: GltfAccessorDiagnosticSink,
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

  if (accessor.bufferView !== undefined) {
    const bvIdx = accessor.bufferView;
    const bv = gltf.bufferViews?.[bvIdx];
    if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`);

    const buf = _getBuffer(buffers, bv.buffer, gltf);
    const compSize = componentByteSize(ct);
    const bvOffset = bv.byteOffset ?? 0;
    const range = accessorBufferViewRange(accessor, bv, 1);
    validateBufferViewAccess(buf, bvIdx, bv, range.requiredByteLength, 'index accessor');
    const dataView = new DataView(buf, bvOffset, bv.byteLength);

    for (let i = 0; i < accessor.count; i++) {
      result[i] = Math.round(readScalar(dataView, range.byteOffset + i * range.byteStride, ct, false));
    }
  }
  // If bufferView is absent, result stays zero-initialized before any sparse patch
  // is applied, which is valid for pure-sparse accessors.

  if (accessor.sparse) {
    // Sparse integer index buffers are legal but extremely rare.
    // Uses null warnings by default so malformed index-buffer sparse patches
    // preserve the historical hard-fail behavior.
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'warning',
      code: 'sparse-accessor-applied',
      path: `accessors[${accessorIndex}].sparse`,
      message: `[vitrum/gltf-adapter] Accessor uses sparse storage (count=${accessor.sparse.count}); applying patch.`,
      accessorIndex,
    });
    const sv = _resolveSparseViews(gltf, buffers, accessorIndex, accessor, 1, warnings, onDiagnostic);
    if (sv) {
      for (let s = 0; s < sv.count; s++) {
        const idx = Math.round(readScalar(sv.idxView, sv.idxByteOffset + s * sv.idxCompSize, sv.idxCt, false));
        if (idx < 0 || idx >= accessor.count) {
          emitAccessorDiagnostic(warnings, onDiagnostic, {
            severity: 'warning',
            code: 'sparse-index-out-of-range',
            path: `accessors[${accessorIndex}].sparse.indices[${s}]`,
            message: `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}`,
            accessorIndex,
            sparseEntryIndex: s,
          });
          throw new Error(
            `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}`,
          );
        }
        result[idx] = Math.round(readScalar(sv.valView, sv.valByteOffset + s * sv.valCompSize, sv.valCt, false));
      }
    }
  }

  return result;
}

function emitAccessorDiagnostic(
  warnings: string[] | null,
  onDiagnostic: GltfAccessorDiagnosticSink | undefined,
  diagnostic: GltfAccessorDiagnostic,
): void {
  if (onDiagnostic) {
    try {
      onDiagnostic(diagnostic);
      return;
    } catch {
      // Host diagnostic callbacks must not abort accessor decoding.
    }
  }
  if (warnings) warnings.push(diagnostic.message);
}

export function validateBufferViewAccess(
  buffer: ArrayBuffer,
  bufferViewIndex: number,
  bufferView: Pick<GltfBufferView, 'byteOffset' | 'byteLength'>,
  requiredByteLength: number,
  label: string,
): void {
  const byteOffset = bufferView.byteOffset ?? 0;
  const byteLength = bufferView.byteLength;
  if (byteOffset < 0 || byteLength < 0 || byteOffset + byteLength > buffer.byteLength) {
    throw new Error(
      `[vitrum/gltf-adapter] ${label} bufferView ${bufferViewIndex} declares byte range ` +
      `[${byteOffset}, ${byteOffset + byteLength}) outside buffer length ${buffer.byteLength}`,
    );
  }
  if (requiredByteLength > byteLength) {
    throw new Error(
      `[vitrum/gltf-adapter] ${label} requires ${requiredByteLength} bytes from bufferView ` +
      `${bufferViewIndex}, but it declares byteLength ${byteLength}`,
    );
  }
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
