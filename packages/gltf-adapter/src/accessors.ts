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
import {
  gltfArrayBufferByteLength,
  type ImportResourceLedger,
} from './importResourceBudget.js';

export type GltfAccessorDiagnosticCode =
  | 'sparse-accessor-applied'
  | 'sparse-indices-buffer-view-not-found'
  | 'sparse-indices-buffer-unavailable'
  | 'sparse-values-buffer-view-not-found'
  | 'sparse-values-buffer-unavailable'
  | 'sparse-indices-buffer-view-truncated'
  | 'sparse-values-buffer-view-truncated'
  | 'invalid-sparse-indices-component-type'
  | 'sparse-indices-not-strictly-increasing'
  | 'sparse-index-out-of-range';

export interface GltfAccessorDiagnostic {
  readonly severity: 'warning' | 'error';
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

export const GLTF_ACCESSOR_DECODE_BUDGET_BYTES = 512 * 1024 * 1024;

function assertSafeNonNegativeInteger(value: unknown, label: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`[vitrum/gltf-adapter] ${label} must be a non-negative safe integer.`);
  }
}

function checkedProduct(a: number, b: number, label: string): number {
  assertSafeNonNegativeInteger(a, `${label} left operand`);
  assertSafeNonNegativeInteger(b, `${label} right operand`);
  if (a !== 0 && b > Math.floor(Number.MAX_SAFE_INTEGER / a)) {
    throw new Error(`[vitrum/gltf-adapter] ${label} exceeds the safe integer range.`);
  }
  return a * b;
}

function checkedSum(a: number, b: number, label: string): number {
  assertSafeNonNegativeInteger(a, `${label} left operand`);
  assertSafeNonNegativeInteger(b, `${label} right operand`);
  if (b > Number.MAX_SAFE_INTEGER - a) {
    throw new Error(`[vitrum/gltf-adapter] ${label} exceeds the safe integer range.`);
  }
  return a + b;
}

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

/**
 * Number of rows per column for a matrix accessor `type` (the column length),
 * or `null` for non-matrix types. MAT2 → 2, MAT3 → 3, MAT4 → 4.
 */
function matrixColumnLength(type: string): number | null {
  switch (type) {
    case 'MAT2':
      return 2;
    case 'MAT3':
      return 3;
    case 'MAT4':
      return 4;
    default:
      return null;
  }
}

/**
 * Column-alignment layout for a matrix accessor whose component size is < 4
 * bytes (glTF 2.0 spec §3.6.2.4): each matrix column must start on a 4-byte
 * boundary, so a column of `colLen` components of `compSize` bytes is padded up
 * to `ceil(compSize*colLen/4)*4` bytes. Returns `null` when no padding applies
 * (non-matrix types, or matrix columns that are already 4-byte aligned — e.g.
 * MAT4/BYTE, MAT2/SHORT, or any FLOAT/UNSIGNED_INT matrix).
 */
function matrixColumnPadding(
  type: string,
  compSize: number,
): { readonly colLen: number; readonly colStride: number; readonly elementByteLength: number } | null {
  if (compSize >= 4) return null;
  const colLen = matrixColumnLength(type);
  if (colLen === null) return null;
  const unpadded = compSize * colLen;
  const colStride = Math.ceil(unpadded / 4) * 4;
  if (colStride === unpadded) return null;
  return { colLen, colStride, elementByteLength: colStride * colLen };
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
 * Validate the core glTF normalization contract and return its effective value.
 * Defaults apply only when the JSON property is absent.
 */
export function validateAccessorNormalization(
  accessor: Pick<GltfAccessor, 'componentType' | 'normalized'>,
  path: string,
): boolean {
  if (!Object.prototype.hasOwnProperty.call(accessor, 'normalized')) return false;
  if (typeof accessor.normalized !== 'boolean') {
    throw new TypeError(`[vitrum/gltf-adapter] ${path}.normalized must be a boolean.`);
  }
  if (
    accessor.normalized &&
    accessor.componentType !== GltfComponentType.BYTE &&
    accessor.componentType !== GltfComponentType.UNSIGNED_BYTE &&
    accessor.componentType !== GltfComponentType.SHORT &&
    accessor.componentType !== GltfComponentType.UNSIGNED_SHORT
  ) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.normalized may be true only for BYTE, ` +
        'UNSIGNED_BYTE, SHORT, or UNSIGNED_SHORT component types.',
    );
  }
  return accessor.normalized;
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
  assertSafeNonNegativeInteger(accessor.count, 'accessor.count');
  const compSize = componentByteSize(accessor.componentType);
  const padding = matrixColumnPadding(accessor.type, compSize);
  // glTF §3.6.2.4: matrix accessors with sub-4-byte components pad each column
  // up to a 4-byte boundary, so the element occupies more bytes than the tight
  // `compSize * componentCount`.
  const elementByteLength = padding !== null
    ? padding.elementByteLength
    : checkedProduct(compSize, componentCount, 'accessor element byte length');
  const byteStride = bufferView?.byteStride ?? elementByteLength;
  assertSafeNonNegativeInteger(byteStride, 'bufferView.byteStride');
  if (byteStride < elementByteLength || byteStride % compSize !== 0 || byteStride > 252) {
    throw new Error(
      `[vitrum/gltf-adapter] bufferView.byteStride ${byteStride} must be component-aligned, ` +
      `at least element byte length ${elementByteLength}, and no greater than 252.`,
    );
  }
  const byteOffset = accessor.byteOffset ?? 0;
  assertSafeNonNegativeInteger(byteOffset, 'accessor.byteOffset');
  if (byteOffset % compSize !== 0) {
    throw new Error(`[vitrum/gltf-adapter] accessor.byteOffset ${byteOffset} is not component-aligned.`);
  }
  const elementsSpan = accessor.count <= 0
    ? 0
    : checkedSum(
        checkedProduct(accessor.count - 1, byteStride, 'accessor byte span'),
        elementByteLength,
        'accessor byte span',
      );
  const requiredByteLength = checkedSum(
    byteOffset,
    elementsSpan,
    'accessor required byte length',
  );
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
  resourceLedger?: ImportResourceLedger,
): Float32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);
  }

  assertSafeNonNegativeInteger(accessor.count, `accessors[${accessorIndex}].count`);
  if (accessor.count === 0) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} count must be greater than zero.`);
  }
  validateAccessorNormalization(accessor, `accessors[${accessorIndex}]`);

  const componentCount = TYPE_COMPONENT_COUNT[accessor.type];
  if (componentCount === undefined) {
    throw new Error(`[vitrum/gltf-adapter] Unknown accessor type "${accessor.type}"`);
  }
  const total = checkedProduct(accessor.count, componentCount, `accessor ${accessorIndex} decoded element count`);
  const decodedBytes = checkedProduct(total, Float32Array.BYTES_PER_ELEMENT, `accessor ${accessorIndex} decoded byte length`);
  if (resourceLedger === undefined && decodedBytes > GLTF_ACCESSOR_DECODE_BUDGET_BYTES) {
    throw new Error(
      `[vitrum/gltf-adapter] Accessor ${accessorIndex} decoded byte length ${decodedBytes} exceeds ` +
      `${GLTF_ACCESSOR_DECODE_BUDGET_BYTES} byte budget.`,
    );
  }
  resourceLedger?.chargeDecodedGeometryBytes(decodedBytes, `accessors[${accessorIndex}]`);
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
  if (bvOffset % compSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] accessor bufferView ${bvIdx}.byteOffset ${bvOffset} ` +
        `is not aligned to component size ${compSize}.`,
    );
  }

  const dataView = new DataView(buf, bvOffset, bv.byteLength);

  // glTF §3.6.2.4: matrix accessors with sub-4-byte components store each
  // column on a 4-byte boundary, so component `c` (column-major) lives at
  // `column * colStride + row * compSize` within the element rather than the
  // tight `c * compSize`. `padding` is null for every other accessor.
  const padding = matrixColumnPadding(accessor.type, compSize);

  for (let i = 0; i < accessor.count; i++) {
    const elemOffset = range.byteOffset + i * range.byteStride;
    for (let c = 0; c < componentCount; c++) {
      const compByteOffset = padding !== null
        ? Math.floor(c / padding.colLen) * padding.colStride + (c % padding.colLen) * compSize
        : c * compSize;
      result[i * componentCount + c] = readScalar(
        dataView,
        elemOffset + compByteOffset,
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
  valElementByteLength: number;
  valPadding: ReturnType<typeof matrixColumnPadding>;
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
 * Malformed sparse storage is never recoverable: applying only part of the
 * authored patch would silently change geometry, animation, or skin data.
 */
function _resolveSparseViews(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  accessor: GltfAccessor,
  componentCount: number,
  warnings: string[] | null,
  onDiagnostic: GltfAccessorDiagnosticSink | undefined,
): SparseViews {
  const sparse = accessor.sparse!;
  assertSafeNonNegativeInteger(sparse.count, `accessors[${accessorIndex}].sparse.count`);
  if (sparse.count <= 0 || sparse.count > accessor.count) {
    throw new Error(
      `[vitrum/gltf-adapter] Accessor ${accessorIndex} sparse.count must be in [1, ${accessor.count}].`,
    );
  }

  const idxBv = gltf.bufferViews?.[sparse.indices.bufferView];
  if (!idxBv) {
    const message = '[vitrum/gltf-adapter] Sparse indices bufferView not found; accessor rejected.';
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'sparse-indices-buffer-view-not-found',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
    });
    throw new Error('[vitrum/gltf-adapter] Sparse indices bufferView not found');
  }
  const idxBuf = buffers.get(idxBv.buffer);
  if (!idxBuf) {
    const message = `[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable; accessor rejected.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'sparse-indices-buffer-unavailable',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
      bufferIndex: idxBv.buffer,
    });
    throw new Error(`[vitrum/gltf-adapter] Sparse indices buffer ${idxBv.buffer} unavailable`);
  }

  const valBv = gltf.bufferViews?.[sparse.values.bufferView];
  if (!valBv) {
    const message = '[vitrum/gltf-adapter] Sparse values bufferView not found; accessor rejected.';
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'sparse-values-buffer-view-not-found',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
    });
    throw new Error('[vitrum/gltf-adapter] Sparse values bufferView not found');
  }
  const valBuf = buffers.get(valBv.buffer);
  if (!valBuf) {
    const message = `[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable; accessor rejected.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'sparse-values-buffer-unavailable',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
      bufferIndex: valBv.buffer,
    });
    throw new Error(`[vitrum/gltf-adapter] Sparse values buffer ${valBv.buffer} unavailable`);
  }

  const idxCt = sparse.indices.componentType;
  if (!_isSparseIndexComponentType(idxCt)) {
    const message =
      `[vitrum/gltf-adapter] Sparse indices componentType ${idxCt} is invalid; ` +
      'expected UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT.';
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'invalid-sparse-indices-component-type',
      path: `accessors[${accessorIndex}].sparse.indices.componentType`,
      message: `${message} Accessor rejected.`,
      accessorIndex,
      componentType: idxCt,
    });
    throw new Error(message);
  }
  const idxCompSize = componentByteSize(idxCt);
  const valCt = accessor.componentType;
  const valCompSize = componentByteSize(valCt);
  const valPadding = matrixColumnPadding(accessor.type, valCompSize);
  const valElementByteLength =
    valPadding?.elementByteLength ??
    checkedProduct(componentCount, valCompSize, 'sparse value element byte length');
  const idxByteOffset = sparse.indices.byteOffset ?? 0;
  const valByteOffset = sparse.values.byteOffset ?? 0;
  assertSafeNonNegativeInteger(
    idxByteOffset,
    `accessors[${accessorIndex}].sparse.indices.byteOffset`,
  );
  assertSafeNonNegativeInteger(
    valByteOffset,
    `accessors[${accessorIndex}].sparse.values.byteOffset`,
  );
  if (idxByteOffset % idxCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] accessors[${accessorIndex}].sparse.indices.byteOffset ` +
        `${idxByteOffset} is not aligned to component size ${idxCompSize}.`,
    );
  }
  if (valByteOffset % valCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] accessors[${accessorIndex}].sparse.values.byteOffset ` +
        `${valByteOffset} is not aligned to component size ${valCompSize}.`,
    );
  }
  const requiredSparseIndexBytes = checkedSum(
    idxByteOffset,
    checkedProduct(
      sparse.count,
      idxCompSize,
      `accessors[${accessorIndex}] sparse indices byte length`,
    ),
    `accessors[${accessorIndex}] sparse indices required byte length`,
  );
  const requiredSparseValueBytes = checkedSum(
    valByteOffset,
    checkedProduct(
      sparse.count,
      valElementByteLength,
      `accessors[${accessorIndex}] sparse values byte length`,
    ),
    `accessors[${accessorIndex}] sparse values required byte length`,
  );
  try {
    validateBufferViewAccess(
      idxBuf,
      sparse.indices.bufferView,
      idxBv,
      requiredSparseIndexBytes,
      'sparse indices',
    );
  } catch (error) {
    const message = `${String(error)} Accessor rejected.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'sparse-indices-buffer-view-truncated',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.indices.bufferView,
    });
    throw error;
  }
  try {
    validateBufferViewAccess(
      valBuf,
      sparse.values.bufferView,
      valBv,
      requiredSparseValueBytes,
      'sparse values',
    );
  } catch (error) {
    const message = `${String(error)} Accessor rejected.`;
    emitAccessorDiagnostic(warnings, onDiagnostic, {
      severity: 'error',
      code: 'sparse-values-buffer-view-truncated',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      message,
      accessorIndex,
      bufferViewIndex: sparse.values.bufferView,
    });
    throw error;
  }
  const idxBufferViewOffset = idxBv.byteOffset ?? 0;
  const valBufferViewOffset = valBv.byteOffset ?? 0;
  if (idxBufferViewOffset % idxCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] sparse indices bufferView ${sparse.indices.bufferView}.byteOffset ` +
        `${idxBufferViewOffset} is not aligned to component size ${idxCompSize}.`,
    );
  }
  if (valBufferViewOffset % valCompSize !== 0) {
    throw new Error(
      `[vitrum/gltf-adapter] sparse values bufferView ${sparse.values.bufferView}.byteOffset ` +
        `${valBufferViewOffset} is not aligned to component size ${valCompSize}.`,
    );
  }

  return {
    idxView: new DataView(idxBuf, idxBufferViewOffset, idxBv.byteLength),
    idxByteOffset,
    idxCt,
    idxCompSize,
    valView: new DataView(valBuf, valBufferViewOffset, valBv.byteLength),
    valByteOffset,
    valCt,
    valCompSize,
    valElementByteLength,
    valPadding,
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

  const normalized = accessor.normalized ?? false;
  let previousIndex = -1;
  for (let s = 0; s < sv.count; s++) {
    const idx = Math.round(readScalar(sv.idxView, sv.idxByteOffset + s * sv.idxCompSize, sv.idxCt, false));
    if (idx < 0 || idx >= accessor.count || idx <= previousIndex) {
      const isOrderingViolation = idx <= previousIndex;
      emitAccessorDiagnostic(warnings, onDiagnostic, {
        severity: 'error',
        code: isOrderingViolation
          ? 'sparse-indices-not-strictly-increasing'
          : 'sparse-index-out-of-range',
        path: `accessors[${accessorIndex}].sparse.indices[${s}]`,
        message: isOrderingViolation
          ? `[vitrum/gltf-adapter] Sparse indices must be strictly increasing; ${idx} follows ${previousIndex}.`
          : `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}.`,
        accessorIndex,
        sparseEntryIndex: s,
      });
      throw new Error(
        isOrderingViolation
          ? `[vitrum/gltf-adapter] Sparse indices for accessor ${accessorIndex} are not strictly increasing.`
          : `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}.`,
      );
    }
    previousIndex = idx;
    for (let c = 0; c < componentCount; c++) {
      const componentByteOffset = sv.valPadding !== null
        ? Math.floor(c / sv.valPadding.colLen) * sv.valPadding.colStride +
          (c % sv.valPadding.colLen) * sv.valCompSize
        : c * sv.valCompSize;
      result[idx * componentCount + c] = readScalar(
        sv.valView,
        sv.valByteOffset + s * sv.valElementByteLength + componentByteOffset,
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
  resourceLedger?: ImportResourceLedger,
): Uint32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} not found`);
  }
  assertSafeNonNegativeInteger(accessor.count, `accessors[${accessorIndex}].count`);
  if (accessor.count === 0) {
    throw new Error(`[vitrum/gltf-adapter] Accessor ${accessorIndex} count must be greater than zero.`);
  }
  if (accessor.type !== 'SCALAR') {
    throw new Error(
      `[vitrum/gltf-adapter] Index accessor must be SCALAR, got "${accessor.type}"`,
    );
  }
  if (validateAccessorNormalization(accessor, `accessors[${accessorIndex}]`)) {
    throw new TypeError(
      `[vitrum/gltf-adapter] Index accessor ${accessorIndex} must not be normalized.`,
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

  const decodedBytes = checkedProduct(
    accessor.count,
    Uint32Array.BYTES_PER_ELEMENT,
    `accessor ${accessorIndex} decoded byte length`,
  );
  if (resourceLedger === undefined && decodedBytes > GLTF_ACCESSOR_DECODE_BUDGET_BYTES) {
    throw new Error(
      `[vitrum/gltf-adapter] Accessor ${accessorIndex} decoded byte length ${decodedBytes} exceeds ` +
      `${GLTF_ACCESSOR_DECODE_BUDGET_BYTES} byte budget.`,
    );
  }
  resourceLedger?.chargeDecodedGeometryBytes(decodedBytes, `accessors[${accessorIndex}]`);
  const result = new Uint32Array(accessor.count);

  if (accessor.bufferView !== undefined) {
    const bvIdx = accessor.bufferView;
    const bv = gltf.bufferViews?.[bvIdx];
    if (!bv) throw new Error(`[vitrum/gltf-adapter] BufferView ${bvIdx} not found`);

    const buf = _getBuffer(buffers, bv.buffer, gltf);
    const bvOffset = bv.byteOffset ?? 0;
    const range = accessorBufferViewRange(accessor, bv, 1);
    validateBufferViewAccess(buf, bvIdx, bv, range.requiredByteLength, 'index accessor');
    const compSize = componentByteSize(ct);
    if (bvOffset % compSize !== 0) {
      throw new Error(
        `[vitrum/gltf-adapter] index accessor bufferView ${bvIdx}.byteOffset ${bvOffset} ` +
          `is not aligned to component size ${compSize}.`,
      );
    }
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
    let previousIndex = -1;
    for (let s = 0; s < sv.count; s++) {
        const idx = Math.round(readScalar(sv.idxView, sv.idxByteOffset + s * sv.idxCompSize, sv.idxCt, false));
        if (idx < 0 || idx >= accessor.count || idx <= previousIndex) {
          const isOrderingViolation = idx <= previousIndex;
          emitAccessorDiagnostic(warnings, onDiagnostic, {
            severity: 'error',
            code: isOrderingViolation
              ? 'sparse-indices-not-strictly-increasing'
              : 'sparse-index-out-of-range',
            path: `accessors[${accessorIndex}].sparse.indices[${s}]`,
            message: isOrderingViolation
              ? `[vitrum/gltf-adapter] Sparse indices must be strictly increasing; ${idx} follows ${previousIndex}.`
              : `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}.`,
            accessorIndex,
            sparseEntryIndex: s,
          });
          throw new Error(
            isOrderingViolation
              ? `[vitrum/gltf-adapter] Sparse indices for accessor ${accessorIndex} are not strictly increasing.`
              : `[vitrum/gltf-adapter] Sparse index ${idx} is outside accessor count ${accessor.count}.`,
          );
        }
        previousIndex = idx;
        result[idx] = Math.round(readScalar(sv.valView, sv.valByteOffset + s * sv.valCompSize, sv.valCt, false));
      }
  }

  const restartSentinel =
    ct === GltfComponentType.UNSIGNED_BYTE
      ? 0xff
      : ct === GltfComponentType.UNSIGNED_SHORT
        ? 0xffff
        : 0xffff_ffff;
  for (let index = 0; index < result.length; index += 1) {
    if (result[index] === restartSentinel) {
      throw new RangeError(
        `[vitrum/gltf-adapter] Index accessor ${accessorIndex} uses reserved ` +
          `primitive-restart value ${restartSentinel} at element ${index}.`,
      );
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
  assertSafeNonNegativeInteger(
    byteOffset,
    `${label} bufferView ${bufferViewIndex}.byteOffset`,
  );
  assertSafeNonNegativeInteger(
    byteLength,
    `${label} bufferView ${bufferViewIndex}.byteLength`,
  );
  assertSafeNonNegativeInteger(
    requiredByteLength,
    `${label} required byte length`,
  );
  const declaredEnd = checkedSum(
    byteOffset,
    byteLength,
    `${label} bufferView ${bufferViewIndex} declared end`,
  );
  const intrinsicBufferByteLength = gltfArrayBufferByteLength(buffer);
  if (intrinsicBufferByteLength === undefined) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${label} bufferView ${bufferViewIndex} does not reference a genuine ArrayBuffer.`,
    );
  }
  if (declaredEnd > intrinsicBufferByteLength) {
    throw new Error(
      `[vitrum/gltf-adapter] ${label} bufferView ${bufferViewIndex} declares byte range ` +
      `[${byteOffset}, ${declaredEnd}) outside buffer length ${intrinsicBufferByteLength}`,
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
