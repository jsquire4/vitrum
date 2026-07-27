import type { GltfAccessor, GltfJson, GltfPrimitive } from './gltfTypes.js';
import { GltfComponentType } from './gltfTypes.js';
import {
  componentByteSize,
  typeComponentCount,
  unpackAccessorFloat,
  unpackAccessorUint32,
  validateAccessorNormalization,
} from './accessors.js';
import { gltfPrimitiveKey, type GltfSceneReachability } from './sceneScope.js';
import { validateGltfPropertyExtensions } from './gltfPropertyValidation.js';
import { validateDeclaredBufferRange } from './bufferRangeValidation.js';
import {
  chargeCompressionHookOutput,
  checkedCompressionProduct,
  checkedCompressionSum,
  CompressionAllocationLedger,
  compressionTypedArrayInfo,
  validateCompressionInputBudget,
  type CompressionDecodeState,
  type CompressionTypedArrayInfo,
} from './compressionLimits.js';
import type {
  DracoDecodeFn,
  DracoTypedArray,
  DracoAccessorComponentType,
  GltfCompressionDiagnostic,
} from './compression.js';
import {
  gltfArrayBufferByteLength,
  GltfResourceLimitError,
  type ImportResourceLedger,
} from './importResourceBudget.js';

const DRACO_EXT = 'KHR_draco_mesh_compression' as const;
const GLTF_MODE_TRIANGLES = 4;
const GLTF_MODE_TRIANGLE_STRIP = 5;

interface ValidDracoExtension {
  readonly bufferView: number;
  readonly attributes: Readonly<Record<string, number>>;
}

interface DracoAttributeMapping {
  readonly semantic: string;
  readonly accessorIndex: number;
  readonly accessor: GltfAccessor;
}

interface DecodedAttributePlan extends DracoAttributeMapping {
  readonly decoded: DracoTypedArray;
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

function safeInteger(value: unknown, path: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} must be a safe integer >= ${minimum}; received ${String(value)}.`,
    );
  }
  return value as number;
}

function canonicalIndexedSemantic(
  semantic: string,
  path: string,
): {
  readonly prefix: 'TEXCOORD' | 'COLOR' | 'JOINTS' | 'WEIGHTS';
  readonly setIndex: number;
} | undefined {
  const prefix = (['TEXCOORD', 'COLOR', 'JOINTS', 'WEIGHTS'] as const).find(
    (candidate) => semantic === candidate || semantic.startsWith(`${candidate}_`),
  );
  if (prefix === undefined) return undefined;
  const match = new RegExp(`^${prefix}_([0-9]+)$`, 'u').exec(semantic);
  if (match == null) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path} must use the ` +
        `${prefix}_<non-negative canonical integer> form.`,
    );
  }
  const setIndex = Number(match[1]);
  if (!Number.isSafeInteger(setIndex)) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} exceeds the supported non-negative ` +
        'safe-integer semantic range.',
    );
  }
  const canonical = `${prefix}_${setIndex}`;
  if (semantic !== canonical) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path} is not canonical; use "${canonical}".`,
    );
  }
  return { prefix, setIndex };
}

function validateDracoAttributeSemantic(semantic: string, path: string): void {
  if (
    semantic === 'POSITION' ||
    semantic === 'NORMAL' ||
    semantic === 'TANGENT' ||
    semantic.startsWith('_') ||
    canonicalIndexedSemantic(semantic, path) !== undefined
  ) {
    return;
  }
  throw new TypeError(
    `[vitrum/gltf-adapter] ${path} uses unknown non-application Draco ` +
      `attribute semantic "${semantic}".`,
  );
}

function isCanonicalJointsSemantic(semantic: string): boolean {
  const match = /^JOINTS_([0-9]+)$/u.exec(semantic);
  if (match == null) return false;
  const setIndex = Number(match[1]);
  return Number.isSafeInteger(setIndex) && semantic === `JOINTS_${setIndex}`;
}

function formatUnknownError(error: unknown): string {
  try {
    return String(error);
  } catch {
    return 'unknown decoder failure';
  }
}

function checkedProduct(a: number, b: number, path: string): number {
  const result = a * b;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path} is not a safe integer.`);
  }
  return result;
}

function stripDracoExtension(primitive: GltfPrimitive): void {
  if (primitive.extensions == null) return;
  delete primitive.extensions[DRACO_EXT];
  if (Object.keys(primitive.extensions).length === 0) delete primitive.extensions;
}

function emitDiagnostic(
  warnings: string[],
  sink: ((diagnostic: GltfCompressionDiagnostic) => void) | undefined,
  diagnostic: GltfCompressionDiagnostic,
): void {
  warnings.push(diagnostic.message);
  try {
    sink?.(diagnostic);
  } catch {
    // Host diagnostics are observational only.
  }
}

function getOwnDracoExtension(primitive: GltfPrimitive): unknown {
  const extensions = hasOwn(primitive, 'extensions') ? primitive.extensions : undefined;
  return extensions !== undefined && hasOwn(extensions, DRACO_EXT)
    ? extensions[DRACO_EXT]
    : undefined;
}

function validateExtension(raw: unknown, path: string): ValidDracoExtension {
  if (!isRecord(raw)) {
    throw new TypeError(`[vitrum/gltf-adapter] ${path} must be an object.`);
  }
  assertExactEnumerableKeys(
    raw,
    new Set(['bufferView', 'attributes', 'extensions', 'extras']),
    path,
  );
  validateGltfPropertyExtensions(raw, path);
  if (!hasOwn(raw, 'bufferView') || !hasOwn(raw, 'attributes')) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.bufferView and ${path}.attributes must be own properties.`,
    );
  }
  const bufferView = safeInteger(raw.bufferView, `${path}.bufferView`);
  if (!isRecord(raw.attributes)) {
    throw new TypeError(`[vitrum/gltf-adapter] ${path}.attributes must be an object.`);
  }
  const attributes: Record<string, number> = Object.create(null) as Record<string, number>;
  for (const key of Reflect.ownKeys(raw.attributes)) {
    if (!Object.prototype.propertyIsEnumerable.call(raw.attributes, key)) continue;
    if (typeof key !== 'string' || key.length === 0) {
      throw new TypeError(
        `[vitrum/gltf-adapter] ${path}.attributes contains an invalid semantic key.`,
      );
    }
    validateDracoAttributeSemantic(key, `${path}.attributes.${key}`);
    const uniqueId = safeInteger(raw.attributes[key], `${path}.attributes.${key}`);
    if (uniqueId > 0xffff_ffff) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${path}.attributes.${key} must be <= 4294967295.`,
      );
    }
    attributes[key] = uniqueId;
  }
  if (Object.keys(attributes).length === 0) {
    throw new Error(`[vitrum/gltf-adapter] ${path}.attributes must not be empty.`);
  }
  return { bufferView, attributes };
}

function validateAccessorShape(accessor: GltfAccessor, path: string): number {
  const count = safeInteger(accessor.count, `${path}.count`, 1);
  validateAccessorNormalization(accessor, path);
  if (accessor.componentType === GltfComponentType.UNSIGNED_INT) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.componentType UNSIGNED_INT is reserved ` +
        'for primitive index accessors and is not valid for a Draco vertex attribute.',
    );
  }
  if (accessor.type === 'MAT2' || accessor.type === 'MAT3' || accessor.type === 'MAT4') {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path}.type must be SCALAR, VEC2, VEC3, or VEC4 for a Draco vertex attribute; received ${accessor.type}.`,
    );
  }
  const components = typeComponentCount(accessor.type);
  componentByteSize(accessor.componentType);
  return checkedProduct(count, components, `${path}.count * components`);
}

function collectMappings(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  extension: ValidDracoExtension,
  primitivePath: string,
): DracoAttributeMapping[] {
  if (!hasOwn(primitive, 'attributes') || !isRecord(primitive.attributes)) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${primitivePath}.attributes must be an own object property.`,
    );
  }
  const mappings: DracoAttributeMapping[] = [];
  for (const semantic of Object.keys(extension.attributes)) {
    if (!hasOwn(primitive.attributes, semantic)) {
      throw new Error(
        `[vitrum/gltf-adapter] ${primitivePath} Draco semantic "${semantic}" has no exact primitive accessor mapping.`,
      );
    }
    const accessorIndex = primitive.attributes[semantic];
    if (!Number.isSafeInteger(accessorIndex) || (accessorIndex as number) < 0) {
      throw new Error(
        `[vitrum/gltf-adapter] ${primitivePath} Draco semantic "${semantic}" has no exact primitive accessor mapping.`,
      );
    }
    const accessor = gltf.accessors?.[accessorIndex as number];
    if (accessor == null) {
      throw new Error(
        `[vitrum/gltf-adapter] ${primitivePath} Draco semantic "${semantic}" references missing accessor ${accessorIndex}.`,
      );
    }
    validateAccessorShape(accessor, `accessors[${accessorIndex}]`);
    if (isCanonicalJointsSemantic(semantic)) {
      if (
        accessor.type !== 'VEC4' ||
        (accessor.componentType !== GltfComponentType.UNSIGNED_BYTE &&
          accessor.componentType !== GltfComponentType.UNSIGNED_SHORT)
      ) {
        throw new TypeError(
          `[vitrum/gltf-adapter] Draco semantic "${semantic}" accessor ` +
            `${accessorIndex} must be VEC4 with UNSIGNED_BYTE or UNSIGNED_SHORT ` +
            'components.',
        );
      }
    }
    mappings.push({ semantic, accessorIndex: accessorIndex as number, accessor });
  }
  return mappings;
}

function validateDracoPointCount(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  mappings: readonly DracoAttributeMapping[],
  primitivePath: string,
): number {
  const pointCount = mappings[0]!.accessor.count;
  for (const mapping of mappings) {
    if (mapping.accessor.count !== pointCount) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${primitivePath} Draco semantic ` +
          `"${mapping.semantic}" accessor ${mapping.accessorIndex} count ` +
          `${mapping.accessor.count} does not match the common Draco point ` +
          `count ${pointCount}.`,
      );
    }
  }

  const rawPositionAccessorIndex = hasOwn(primitive.attributes, 'POSITION')
    ? primitive.attributes.POSITION
    : undefined;
  if (rawPositionAccessorIndex !== undefined) {
    const positionAccessorIndex = safeInteger(
      rawPositionAccessorIndex,
      `${primitivePath}.attributes.POSITION`,
    );
    const positionAccessor = gltf.accessors?.[positionAccessorIndex];
    if (positionAccessor == null) {
      throw new Error(
        `[vitrum/gltf-adapter] ${primitivePath} POSITION references missing ` +
          `accessor ${positionAccessorIndex}.`,
      );
    }
    validateAccessorShape(positionAccessor, `accessors[${positionAccessorIndex}]`);
    if (positionAccessor.count !== pointCount) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${primitivePath} POSITION accessor ` +
          `${positionAccessorIndex} count ${positionAccessor.count} does not ` +
          `match the common Draco point count ${pointCount}.`,
      );
    }
  }
  return pointCount;
}

function validateCompressedDeclaration(
  gltf: GltfJson,
  extension: ValidDracoExtension,
  path: string,
): {
  readonly bufferIndex: number;
  readonly byteOffset: number;
  readonly byteLength: number;
} {
  const bufferView = gltf.bufferViews?.[extension.bufferView];
  if (bufferView == null) {
    throw new Error(`[vitrum/gltf-adapter] ${path}.bufferView ${extension.bufferView} is missing.`);
  }
  if (!hasOwn(bufferView, 'buffer') || !hasOwn(bufferView, 'byteLength')) {
    throw new TypeError(
      `[vitrum/gltf-adapter] bufferViews[${extension.bufferView}].buffer and ` +
        `bufferViews[${extension.bufferView}].byteLength must be own properties.`,
    );
  }
  const bufferIndex = safeInteger(bufferView.buffer, `bufferViews[${extension.bufferView}].buffer`);
  const byteOffset =
    !hasOwn(bufferView, 'byteOffset') || bufferView.byteOffset === undefined
      ? 0
      : safeInteger(bufferView.byteOffset, `bufferViews[${extension.bufferView}].byteOffset`);
  const byteLength = safeInteger(
    bufferView.byteLength,
    `bufferViews[${extension.bufferView}].byteLength`,
    1,
  );
  validateDeclaredBufferRange(
    gltf,
    bufferIndex,
    byteOffset,
    byteLength,
    `bufferViews[${extension.bufferView}]`,
  );
  return { bufferIndex, byteOffset, byteLength };
}

function validateCompressedBytes(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  extension: ValidDracoExtension,
  path: string,
  compressionInputBudgetByteLength?: number,
): Uint8Array {
  const { bufferIndex, byteOffset, byteLength } = validateCompressedDeclaration(
    gltf,
    extension,
    path,
  );
  validateCompressionInputBudget(
    byteLength,
    `${path} compressed input`,
    compressionInputBudgetByteLength,
  );
  const source = buffers.get(bufferIndex);
  if (source == null) {
    throw new Error(`[vitrum/gltf-adapter] compressed buffer ${bufferIndex} is unavailable.`);
  }
  const end = byteOffset + byteLength;
  const intrinsicByteLength = gltfArrayBufferByteLength(source);
  if (
    intrinsicByteLength === undefined ||
    !Number.isSafeInteger(end) ||
    end > intrinsicByteLength
  ) {
    throw new RangeError(
      `[vitrum/gltf-adapter] compressed bufferView ${extension.bufferView} range ` +
        `[${byteOffset}, ${String(end)}) exceeds buffer length ${String(intrinsicByteLength)}.`,
    );
  }
  return new Uint8Array(source, byteOffset, byteLength);
}

function validateExactAttributeFallback(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  mapping: DracoAttributeMapping,
  allocationLedger: CompressionAllocationLedger,
  allocationPath: string,
  resourceLedger: ImportResourceLedger | undefined,
): void {
  if (mapping.accessor.bufferView === undefined && mapping.accessor.sparse === undefined) {
    throw new Error(
      `accessor ${mapping.accessorIndex} has neither a fallback bufferView nor sparse data`,
    );
  }
  const expected = validateAccessorShape(mapping.accessor, `accessors[${mapping.accessorIndex}]`);
  allocationLedger.charge(
    checkedCompressionProduct(
      expected,
      Float32Array.BYTES_PER_ELEMENT,
      `${allocationPath} unpacked attribute`,
    ),
    `${allocationPath} unpacked attribute`,
  );
  const diagnostics: string[] = [];
  const unpacked = unpackAccessorFloat(
    gltf,
    buffers,
    mapping.accessorIndex,
    [],
    (diagnostic) => {
      if (diagnostic.code !== 'sparse-accessor-applied') diagnostics.push(diagnostic.message);
    },
    resourceLedger,
  );
  if (diagnostics.length > 0) throw new Error(diagnostics.join('; '));
  if (unpacked.length !== expected) {
    throw new RangeError(`accessor ${mapping.accessorIndex} fallback length is not exact`);
  }
  for (let index = 0; index < unpacked.length; index += 1) {
    if (!Number.isFinite(unpacked[index])) {
      throw new RangeError(`accessor ${mapping.accessorIndex} fallback[${index}] is non-finite`);
    }
  }
}

function indexComponentMaximum(componentType: GltfComponentType): number {
  switch (componentType) {
    case GltfComponentType.UNSIGNED_BYTE:
      return 0xff;
    case GltfComponentType.UNSIGNED_SHORT:
      return 0xffff;
    case GltfComponentType.UNSIGNED_INT:
      return 0xffff_ffff;
    default:
      throw new TypeError(
        `[vitrum/gltf-adapter] Draco index accessor componentType ${componentType} is not unsigned integer.`,
      );
  }
}

function validateIndexAccessor(accessor: GltfAccessor, accessorIndex: number): void {
  if (accessor.type !== 'SCALAR') {
    throw new TypeError(
      `[vitrum/gltf-adapter] Draco index accessor ${accessorIndex} must be SCALAR; received ${accessor.type}.`,
    );
  }
  safeInteger(accessor.count, `accessors[${accessorIndex}].count`, 1);
  if (validateAccessorNormalization(accessor, `accessors[${accessorIndex}]`)) {
    throw new TypeError(
      `[vitrum/gltf-adapter] Draco index accessor ${accessorIndex} must not be normalized.`,
    );
  }
  indexComponentMaximum(accessor.componentType);
}

function validateTriangleListCount(count: number, path: string): void {
  if (count === 0 || count % 3 !== 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} must be a non-empty triangle-list ` +
        `index count divisible by 3; received ${count}.`,
    );
  }
}

function validatePrimitiveMode(primitive: GltfPrimitive, primitivePath: string): number {
  const mode = hasOwn(primitive, 'mode') ? primitive.mode : GLTF_MODE_TRIANGLES;
  if (mode !== GLTF_MODE_TRIANGLES && mode !== GLTF_MODE_TRIANGLE_STRIP) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${primitivePath}.mode must be TRIANGLES (4) or ` +
        `TRIANGLE_STRIP (5) when ${DRACO_EXT} is present; received ${String(mode)}.`,
    );
  }
  return mode;
}

function validateIndexValues(
  indices: Uint8Array | Uint16Array | Uint32Array,
  accessor: GltfAccessor,
  accessorIndex: number,
  vertexCount: number | undefined,
  allowTriangleListCountChange = false,
): void {
  if (!allowTriangleListCountChange && indices.length !== accessor.count) {
    throw new RangeError(
      `[vitrum/gltf-adapter] Draco indices length ${indices.length} does not equal accessor ${accessorIndex} count ${accessor.count}.`,
    );
  }
  if (allowTriangleListCountChange && (indices.length === 0 || indices.length % 3 !== 0)) {
    throw new RangeError(
      `[vitrum/gltf-adapter] Draco TRIANGLE_STRIP conversion produced ${indices.length} face-list indices; ` +
        'a non-empty length divisible by 3 is required.',
    );
  }
  const componentMaximum = indexComponentMaximum(accessor.componentType);
  for (let index = 0; index < indices.length; index += 1) {
    const value = indices[index]!;
    if (value >= componentMaximum) {
      throw new RangeError(
        `[vitrum/gltf-adapter] Draco index ${value} at ${index} reaches or exceeds ` +
          `the reserved componentType maximum ${componentMaximum}.`,
      );
    }
    if (vertexCount !== undefined && value >= vertexCount) {
      throw new RangeError(
        `[vitrum/gltf-adapter] Draco index ${value} at ${index} exceeds vertex count ${vertexCount}.`,
      );
    }
  }
}

function validateExactIndexFallback(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  accessorIndex: number,
  accessor: GltfAccessor,
  vertexCount: number | undefined,
  allocationLedger: CompressionAllocationLedger,
  allocationPath: string,
  resourceLedger: ImportResourceLedger | undefined,
): void {
  validateIndexAccessor(accessor, accessorIndex);
  if (accessor.bufferView === undefined && accessor.sparse === undefined) {
    throw new Error(
      `index accessor ${accessorIndex} has neither a fallback bufferView nor sparse data`,
    );
  }
  allocationLedger.charge(
    checkedCompressionProduct(
      accessor.count,
      Uint32Array.BYTES_PER_ELEMENT,
      `${allocationPath} unpacked indices`,
    ),
    `${allocationPath} unpacked indices`,
  );
  const diagnostics: string[] = [];
  const indices = unpackAccessorUint32(
    gltf,
    buffers,
    accessorIndex,
    [],
    (diagnostic) => {
      if (diagnostic.code !== 'sparse-accessor-applied') diagnostics.push(diagnostic.message);
    },
    resourceLedger,
  );
  if (diagnostics.length > 0) throw new Error(diagnostics.join('; '));
  validateIndexValues(indices, accessor, accessorIndex, vertexCount);
}

function isDracoAttributeInfo(
  info: CompressionTypedArrayInfo | undefined,
): info is CompressionTypedArrayInfo {
  return (
    info !== undefined &&
    !info.shared &&
    (info.kind === 'Int8Array' ||
      info.kind === 'Uint8Array' ||
      info.kind === 'Int16Array' ||
      info.kind === 'Uint16Array' ||
      info.kind === 'Float32Array')
  );
}

function isDracoIndexInfo(
  info: CompressionTypedArrayInfo | undefined,
): info is CompressionTypedArrayInfo {
  return (
    info !== undefined &&
    !info.shared &&
    (info.kind === 'Uint8Array' || info.kind === 'Uint16Array' || info.kind === 'Uint32Array')
  );
}

function copyTypedArrayBytes(info: CompressionTypedArrayInfo): ArrayBuffer {
  return new Uint8Array(new Uint8Array(info.buffer, info.byteOffset, info.byteLength)).buffer;
}

function validateDecodedAttribute(
  info: CompressionTypedArrayInfo,
  mapping: DracoAttributeMapping,
): DracoTypedArray {
  if (!isDracoAttributeInfo(info)) {
    throw new TypeError(
      `[vitrum/gltf-adapter] Draco semantic "${mapping.semantic}" did not decode to a supported typed array.`,
    );
  }
  const copied = copyTypedArrayBytes(info);
  const snapshot: DracoTypedArray =
    info.kind === 'Int8Array'
      ? new Int8Array(copied)
      : info.kind === 'Uint8Array'
        ? new Uint8Array(copied)
        : info.kind === 'Int16Array'
          ? new Int16Array(copied)
          : info.kind === 'Uint16Array'
            ? new Uint16Array(copied)
            : new Float32Array(copied);
  const expected = validateAccessorShape(mapping.accessor, `accessors[${mapping.accessorIndex}]`);
  if (info.length !== expected) {
    throw new RangeError(
      `[vitrum/gltf-adapter] Draco semantic "${mapping.semantic}" decoded ${info.length} components; expected exactly ${expected}.`,
    );
  }
  if (info.kind === 'Float32Array' && isCanonicalJointsSemantic(mapping.semantic)) {
    throw new TypeError(
      `[vitrum/gltf-adapter] Draco semantic "${mapping.semantic}" must ` +
        'preserve its declared unsigned integer joint-index component type.',
    );
  }
  const decodedComponentType =
    info.kind === 'Int8Array'
      ? GltfComponentType.BYTE
      : info.kind === 'Uint8Array'
        ? GltfComponentType.UNSIGNED_BYTE
        : info.kind === 'Int16Array'
          ? GltfComponentType.SHORT
          : info.kind === 'Uint16Array'
            ? GltfComponentType.UNSIGNED_SHORT
            : GltfComponentType.FLOAT;
  if (decodedComponentType !== mapping.accessor.componentType && info.kind !== 'Float32Array') {
    throw new TypeError(
      `[vitrum/gltf-adapter] Draco semantic "${mapping.semantic}" component type does not match accessor ${mapping.accessorIndex}.`,
    );
  }
  if (info.kind === 'Float32Array') {
    for (let index = 0; index < snapshot.length; index += 1) {
      if (!Number.isFinite(snapshot[index])) {
        throw new RangeError(
          `[vitrum/gltf-adapter] Draco semantic "${mapping.semantic}" contains non-finite Float32 value at ${index}.`,
        );
      }
    }
  }
  return snapshot;
}

function snapshotDecodedIndices(
  info: CompressionTypedArrayInfo,
  missingMessage: string,
): Uint8Array | Uint16Array | Uint32Array {
  if (!isDracoIndexInfo(info)) {
    throw new TypeError(missingMessage);
  }
  const copied = copyTypedArrayBytes(info);
  return info.kind === 'Uint8Array'
    ? new Uint8Array(copied)
    : info.kind === 'Uint16Array'
      ? new Uint16Array(copied)
      : new Uint32Array(copied);
}

function nextBufferIndex(gltf: GltfJson, buffers: Map<number, ArrayBuffer>): number {
  let next = gltf.buffers?.length ?? 0;
  // Ignore unrelated out-of-schema Map keys; only exact collisions can move a
  // synthetic index. This prevents a single huge caller key from causing the
  // glTF descriptor array to be padded up to that key.
  while (buffers.has(next)) {
    if (next >= Number.MAX_SAFE_INTEGER) {
      throw new RangeError('[vitrum/gltf-adapter] No safe synthetic buffer index remains.');
    }
    next += 1;
  }
  return next;
}

function addSyntheticBufferView(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  value: ArrayBufferView,
  allocationLedger: CompressionAllocationLedger,
  allocationPath: string,
  resourceLedger: ImportResourceLedger | undefined,
): number {
  const info = compressionTypedArrayInfo(value);
  if (info === undefined || info.shared) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${allocationPath} must be a non-shared typed array.`,
    );
  }
  allocationLedger.charge(info.byteLength, `${allocationPath} retained copy`);
  resourceLedger?.chargeDecodedGeometryBytes(info.byteLength, `${allocationPath} retained copy`);
  const bytes = new Uint8Array(info.buffer, info.byteOffset, info.byteLength);
  const copy = new Uint8Array(bytes);
  const bufferIndex = nextBufferIndex(gltf, buffers);
  buffers.set(bufferIndex, copy.buffer);
  const bufferList = (gltf.buffers ??= []);
  while (bufferList.length < bufferIndex) bufferList.push({ byteLength: 0 });
  bufferList.push({ byteLength: copy.byteLength });
  const views = (gltf.bufferViews ??= []);
  views.push({ buffer: bufferIndex, byteOffset: 0, byteLength: copy.byteLength });
  return views.length - 1;
}

function appendDecodedAccessor(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  source: GltfAccessor,
  decoded: DracoTypedArray | Uint32Array,
  dequantizedFloat: boolean,
  allocationLedger: CompressionAllocationLedger,
  allocationPath: string,
  resourceLedger: ImportResourceLedger | undefined,
): number {
  const bufferView = addSyntheticBufferView(
    gltf,
    buffers,
    decoded,
    allocationLedger,
    allocationPath,
    resourceLedger,
  );
  const next: GltfAccessor = {
    ...source,
    bufferView,
    byteOffset: 0,
    ...(dequantizedFloat
      ? {
          componentType: GltfComponentType.FLOAT,
          normalized: false,
        }
      : {}),
  };
  if (source.min !== undefined || source.max !== undefined) {
    const components = typeComponentCount(source.type);
    const min = new Array<number>(components).fill(Number.POSITIVE_INFINITY);
    const max = new Array<number>(components).fill(Number.NEGATIVE_INFINITY);
    for (let index = 0; index < decoded.length; index += 1) {
      const component = index % components;
      min[component] = Math.min(min[component]!, decoded[index]!);
      max[component] = Math.max(max[component]!, decoded[index]!);
    }
    next.min = min;
    next.max = max;
  }
  delete next.sparse;
  const accessors = (gltf.accessors ??= []);
  accessors.push(next);
  return accessors.length - 1;
}

function encodeIndicesForAccessor(
  indices: Uint8Array | Uint16Array | Uint32Array,
  componentType: GltfComponentType,
): Uint8Array | Uint16Array | Uint32Array {
  switch (componentType) {
    case GltfComponentType.UNSIGNED_BYTE:
      return new Uint8Array(indices);
    case GltfComponentType.UNSIGNED_SHORT:
      return new Uint16Array(indices);
    case GltfComponentType.UNSIGNED_INT:
      return new Uint32Array(indices);
    default:
      throw new TypeError(
        `[vitrum/gltf-adapter] Draco index componentType ${componentType} is not unsigned integer.`,
      );
  }
}

function decodedIndexComponentType(
  indices: Uint8Array | Uint16Array | Uint32Array,
): GltfComponentType {
  return indices instanceof Uint8Array
    ? GltfComponentType.UNSIGNED_BYTE
    : indices instanceof Uint16Array
      ? GltfComponentType.UNSIGNED_SHORT
      : GltfComponentType.UNSIGNED_INT;
}

function validateSyntheticIndexValues(
  indices: Uint8Array | Uint16Array | Uint32Array,
  vertexCount: number | undefined,
  primitivePath: string,
): void {
  if (indices.length === 0 || indices.length % 3 !== 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${primitivePath} decoded ${indices.length} Draco face indices; ` +
        'a non-empty triangle-list length divisible by 3 is required.',
    );
  }
  const componentMaximum =
    indices instanceof Uint8Array ? 0xff : indices instanceof Uint16Array ? 0xffff : 0xffff_ffff;
  for (let index = 0; index < indices.length; index += 1) {
    const value = indices[index]!;
    if (value >= componentMaximum) {
      throw new RangeError(
        `[vitrum/gltf-adapter] Draco index ${value} at ${index} reaches the ` +
          `reserved componentType maximum ${componentMaximum}.`,
      );
    }
    if (vertexCount !== undefined && value >= vertexCount) {
      throw new RangeError(
        `[vitrum/gltf-adapter] Draco index ${value} at ${index} exceeds vertex count ${vertexCount}.`,
      );
    }
  }
}

function appendSyntheticIndexAccessor(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  indices: Uint8Array | Uint16Array | Uint32Array,
  allocationLedger: CompressionAllocationLedger,
  allocationPath: string,
  resourceLedger: ImportResourceLedger | undefined,
): number {
  const bufferView = addSyntheticBufferView(
    gltf,
    buffers,
    indices,
    allocationLedger,
    allocationPath,
    resourceLedger,
  );
  const accessors = (gltf.accessors ??= []);
  accessors.push({
    bufferView,
    byteOffset: 0,
    componentType: decodedIndexComponentType(indices),
    count: indices.length,
    type: 'SCALAR',
  });
  return accessors.length - 1;
}

function validateWholeFallback(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  primitive: GltfPrimitive,
  allocationLedger: CompressionAllocationLedger,
  primitivePath: string,
  resourceLedger: ImportResourceLedger | undefined,
): void {
  let positionCount: number | undefined;
  for (const [semantic, rawAccessorIndex] of Object.entries(primitive.attributes)) {
    const accessorIndex = safeInteger(rawAccessorIndex, `primitive.attributes.${semantic}`);
    const accessor = gltf.accessors?.[accessorIndex];
    if (accessor == null) {
      throw new Error(`attribute accessor ${accessorIndex} (${semantic}) is missing`);
    }
    validateExactAttributeFallback(
      gltf,
      buffers,
      { semantic, accessorIndex, accessor },
      allocationLedger,
      `${primitivePath} fallback ${semantic}`,
      resourceLedger,
    );
    if (semantic === 'POSITION') positionCount = accessor.count;
  }
  const ownsIndices = hasOwn(primitive, 'indices');
  const rawIndices = ownsIndices ? primitive.indices : undefined;
  if (ownsIndices) {
    const accessorIndex = safeInteger(rawIndices, 'primitive.indices');
    const accessor = gltf.accessors?.[accessorIndex];
    if (accessor == null) throw new Error(`index accessor ${accessorIndex} is missing`);
    validateExactIndexFallback(
      gltf,
      buffers,
      accessorIndex,
      accessor,
      positionCount,
      allocationLedger,
      `${primitivePath} fallback indices`,
      resourceLedger,
    );
  }
}

/**
 * Minimum decoded-geometry allocation implied by a structurally valid Draco
 * declaration. Integer attributes may legally arrive dequantized as Float32,
 * and strip/no-accessor face counts are decoder-defined, so those paths are
 * tightened again against the exact hook result before adapter copies.
 */
function declaredDracoGeometryMinimum(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  primitiveMode: number,
  mappings: readonly DracoAttributeMapping[],
  primitivePath: string,
  extensionPath: string,
): number {
  const parts: number[] = [];
  for (const mapping of mappings) {
    const componentCount = validateAccessorShape(
      mapping.accessor,
      `accessors[${mapping.accessorIndex}]`,
    );
    const byteLength = checkedCompressionProduct(
      componentCount,
      componentByteSize(mapping.accessor.componentType),
      `${extensionPath} ${mapping.semantic} declared decoded output`,
    );
    // Hook output + validation snapshot + retained synthetic-buffer copy.
    parts.push(byteLength, byteLength, byteLength);
  }

  const ownsIndices = hasOwn(primitive, 'indices');
  if (ownsIndices) {
    const accessorIndex = safeInteger(primitive.indices, `${primitivePath}.indices`);
    const accessor = gltf.accessors?.[accessorIndex];
    if (accessor == null) {
      throw new Error(
        `[vitrum/gltf-adapter] ${primitivePath} references missing index accessor ${accessorIndex}.`,
      );
    }
    validateIndexAccessor(accessor, accessorIndex);
    const decodedCount = primitiveMode === GLTF_MODE_TRIANGLES ? accessor.count : 3;
    const hookAndSnapshotMinimum = checkedCompressionProduct(
      decodedCount,
      Uint8Array.BYTES_PER_ELEMENT,
      `${extensionPath} declared index hook output`,
    );
    const encodedByteLength = checkedCompressionProduct(
      decodedCount,
      componentByteSize(accessor.componentType),
      `${extensionPath} declared re-encoded index output`,
    );
    // Hook output + validation snapshot + re-encoded output + retained copy.
    parts.push(
      hookAndSnapshotMinimum,
      hookAndSnapshotMinimum,
      encodedByteLength,
      encodedByteLength,
    );
  } else {
    // A valid synthetic Draco face list is non-empty and divisible by three.
    const minimumFaceListBytes = 3 * Uint8Array.BYTES_PER_ELEMENT;
    // Hook output + validation snapshot + retained synthetic-buffer copy.
    parts.push(minimumFaceListBytes, minimumFaceListBytes, minimumFaceListBytes);
  }
  return checkedCompressionSum(parts, `${extensionPath} declared decoded geometry`);
}

/**
 * Validate caller-owned Draco declarations before structuredClone strips
 * symbol-keyed JSON impostors. The decode pass repeats these checks against its
 * private clone so this preflight is observational and cannot mutate input.
 */
export function preflightDracoCompressionDeclarations(
  gltf: GltfJson,
  sceneReachability: GltfSceneReachability | undefined,
): void {
  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (
        sceneReachability !== undefined &&
        !sceneReachability.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))
      )
        continue;
      const rawExtension = getOwnDracoExtension(primitive);
      if (rawExtension === undefined) continue;
      const primitivePath = `meshes[${meshIndex}].primitives[${primitiveIndex}]`;
      const extension = validateExtension(rawExtension, `${primitivePath}.extensions.${DRACO_EXT}`);
      validateCompressedDeclaration(gltf, extension, primitivePath);
      const primitiveMode = validatePrimitiveMode(primitive, primitivePath);
      const mappings = collectMappings(gltf, primitive, extension, primitivePath);
      validateDracoPointCount(gltf, primitive, mappings, primitivePath);
      const ownsIndices = hasOwn(primitive, 'indices');
      const rawIndices = ownsIndices ? primitive.indices : undefined;
      if (ownsIndices) {
        const accessorIndex = safeInteger(rawIndices, `${primitivePath}.indices`);
        const accessor = gltf.accessors?.[accessorIndex];
        if (accessor == null) {
          throw new Error(
            `[vitrum/gltf-adapter] ${primitivePath} references missing index accessor ${accessorIndex}.`,
          );
        }
        validateIndexAccessor(accessor, accessorIndex);
        if (primitiveMode === GLTF_MODE_TRIANGLES) {
          validateTriangleListCount(
            accessor.count,
            `accessors[${accessorIndex}].count for ${primitivePath}`,
          );
        }
      }
    }
  }
}

export async function resolveDracoStrict(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  decode: DracoDecodeFn | undefined,
  required: boolean,
  warnings: string[],
  onDiagnostic: ((diagnostic: GltfCompressionDiagnostic) => void) | undefined,
  sceneReachability: GltfSceneReachability | undefined,
  allocationLedger: CompressionAllocationLedger,
  decodeState: CompressionDecodeState,
  resourceLedger: ImportResourceLedger | undefined,
  hookOutputPrecharged: boolean,
): Promise<void> {
  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (
        sceneReachability !== undefined &&
        !sceneReachability.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))
      )
        continue;
      const rawExtension = getOwnDracoExtension(primitive);
      if (rawExtension === undefined) continue;
      const primitivePath = `meshes[${meshIndex}].primitives[${primitiveIndex}]`;
      const extensionPath = `${primitivePath}.extensions.${DRACO_EXT}`;
      const extension = validateExtension(rawExtension, extensionPath);
      const primitiveMode = validatePrimitiveMode(primitive, primitivePath);
      const mappings = collectMappings(gltf, primitive, extension, primitivePath);
      const dracoPointCount = validateDracoPointCount(gltf, primitive, mappings, primitivePath);
      const label = mesh.name ?? meshIndex;

      const useWholeFallback = (reason: string): void => {
        if (required) {
          throw new Error(
            `[vitrum/gltf-adapter] ${DRACO_EXT} is listed in extensionsRequired on mesh ` +
              `"${label}"; required Draco assets must decode the required extension: ${reason}`,
          );
        }
        try {
          validateWholeFallback(
            gltf,
            buffers,
            primitive,
            allocationLedger,
            primitivePath,
            resourceLedger,
          );
        } catch (fallbackError) {
          if (fallbackError instanceof GltfResourceLimitError) throw fallbackError;
          throw new Error(
            `[vitrum/gltf-adapter] Draco mesh "${label}" could not decode (${reason}) and has no fully valid accessor fallback: ${formatUnknownError(fallbackError)}`,
          );
        }
        emitDiagnostic(warnings, onDiagnostic, {
          severity: 'warning',
          code: 'draco-fallback-accessors-used',
          path: extensionPath,
          extension: DRACO_EXT,
          meshIndex,
          primitiveIndex,
          message:
            `[vitrum/gltf-adapter] Draco mesh "${label}" could not decode (${reason}). ` +
            'Using its fully validated uncompressed fallback accessors.',
        });
        stripDracoExtension(primitive);
      };

      if (decode === undefined || decodeState.attemptsDisabled) {
        useWholeFallback(
          decode === undefined
            ? 'no dracoDecode hook was supplied'
            : 'compressed decode attempts were disabled after an earlier allocation-budget failure',
        );
        continue;
      }

      let compressed: Uint8Array;
      try {
        compressed = validateCompressedBytes(
          gltf,
          buffers,
          extension,
          extensionPath,
          // The import ledger has already governed the backing resource. A zero
          // here removes only the duplicate fixed compression-input ceiling;
          // intrinsic range and safe-integer validation remain active.
          resourceLedger === undefined ? undefined : 0,
        );
      } catch (error) {
        if (error instanceof GltfResourceLimitError) throw error;
        useWholeFallback(formatUnknownError(error));
        continue;
      }

      resourceLedger?.ensureDecodedGeometryBytes(
        declaredDracoGeometryMinimum(
          gltf,
          primitive,
          primitiveMode,
          mappings,
          primitivePath,
          extensionPath,
        ),
        `${extensionPath} declared decoded output and adapter copies`,
      );

      let resultAttributes: Record<string, unknown>;
      let resultIndices: unknown;
      try {
        allocationLedger.charge(compressed.byteLength, `${extensionPath} compressed input copy`);
        const candidate = await decode(new Uint8Array(compressed), extension.attributes, {
          attributes: Object.fromEntries(
            mappings.map(({ semantic, accessor }) => [
              semantic,
              {
                componentType: accessor.componentType as DracoAccessorComponentType,
                normalized: accessor.normalized === true,
                count: accessor.count,
                type: accessor.type as 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4',
              },
            ]),
          ),
        });
        if (!isRecord(candidate)) throw new TypeError('dracoDecode result must be an object');
        if (!hasOwn(candidate, 'attributes')) {
          throw new TypeError('dracoDecode result must own its attributes field');
        }
        if (!hasOwn(candidate, 'indices')) {
          throw new TypeError('dracoDecode result must own its indices field');
        }
        const candidateAttributes = candidate.attributes;
        const candidateIndices = candidate.indices;
        assertExactEnumerableKeys(
          candidate,
          new Set(['attributes', 'indices']),
          'dracoDecode result',
        );
        if (!isRecord(candidateAttributes)) {
          throw new TypeError('dracoDecode result.attributes must be an object');
        }
        assertExactEnumerableKeys(
          candidateAttributes,
          new Set(mappings.map(({ semantic }) => semantic)),
          'dracoDecode result.attributes',
        );
        resultAttributes = candidateAttributes;
        resultIndices = candidateIndices;
      } catch (error) {
        if (error instanceof GltfResourceLimitError) throw error;
        useWholeFallback(formatUnknownError(error));
        continue;
      }

      const attributePlans: DecodedAttributePlan[] = [];
      let decodedIndices: Uint8Array | Uint16Array | Uint32Array | undefined;
      let indexAccessor: GltfAccessor | undefined;
      let indexAccessorIndex: number | undefined;
      try {
        const rawAttributes = new Map<string, CompressionTypedArrayInfo>();
        let attributeByteLength = 0;
        let classificationError: Error | undefined;
        for (const mapping of mappings) {
          let candidate: unknown;
          try {
            if (!hasOwn(resultAttributes, mapping.semantic)) {
              throw new Error(`dracoDecode omitted semantic "${mapping.semantic}"`);
            }
            candidate = resultAttributes[mapping.semantic];
          } catch (error) {
            classificationError ??=
              error instanceof Error ? error : new Error(formatUnknownError(error));
            continue;
          }
          const info = compressionTypedArrayInfo(candidate);
          if (info !== undefined) {
            attributeByteLength = checkedCompressionSum(
              [attributeByteLength, info.byteLength],
              `${extensionPath} decoded attribute output`,
            );
          }
          if (!isDracoAttributeInfo(info)) {
            if (candidate === undefined) {
              classificationError ??= new Error(
                `dracoDecode omitted semantic "${mapping.semantic}"`,
              );
            } else {
              classificationError ??= new TypeError(
                `[vitrum/gltf-adapter] Draco semantic "${mapping.semantic}" did not decode to a supported non-shared typed array.`,
              );
            }
            continue;
          }
          rawAttributes.set(mapping.semantic, info);
        }

        const indexInfo = compressionTypedArrayInfo(resultIndices);
        const rawIndexByteLength = indexInfo?.byteLength ?? 0;
        if (!isDracoIndexInfo(indexInfo)) {
          classificationError ??= new TypeError(
            'dracoDecode omitted exact non-shared unsigned integer indices',
          );
        }

        const hookOutputByteLength = checkedCompressionSum(
          [attributeByteLength, rawIndexByteLength],
          `${extensionPath} hook output allocation`,
        );
        if (!hookOutputPrecharged) {
          resourceLedger?.chargeDecodedGeometryBytes(
            hookOutputByteLength,
            `${extensionPath} hook output allocation`,
          );
        }
        chargeCompressionHookOutput(
          allocationLedger,
          decodeState,
          hookOutputByteLength,
          `${extensionPath} hook output allocation`,
        );
        if (classificationError !== undefined) throw classificationError;
        if (!isDracoIndexInfo(indexInfo)) {
          throw new TypeError('dracoDecode omitted exact non-shared unsigned integer indices');
        }

        const ownsIndices = hasOwn(primitive, 'indices');
        const rawIndices = ownsIndices ? primitive.indices : undefined;
        let encodedIndexByteLength = 0;
        if (ownsIndices) {
          indexAccessorIndex = safeInteger(rawIndices, `${primitivePath}.indices`);
          indexAccessor = gltf.accessors?.[indexAccessorIndex];
          if (indexAccessor == null) {
            throw new Error(
              `[vitrum/gltf-adapter] ${primitivePath} references missing index accessor ${indexAccessorIndex}.`,
            );
          }
          validateIndexAccessor(indexAccessor, indexAccessorIndex);
          if (primitiveMode === GLTF_MODE_TRIANGLES) {
            validateTriangleListCount(
              indexAccessor.count,
              `accessors[${indexAccessorIndex}].count for ${primitivePath}`,
            );
          }
          encodedIndexByteLength = checkedCompressionProduct(
            indexInfo.length,
            componentByteSize(indexAccessor.componentType),
            `${extensionPath} re-encoded index output`,
          );
        }

        const futureAllocation = ownsIndices
          ? checkedCompressionSum(
              [
                attributeByteLength,
                attributeByteLength,
                rawIndexByteLength,
                encodedIndexByteLength,
                encodedIndexByteLength,
              ],
              `${extensionPath} validation and retained allocation`,
            )
          : checkedCompressionSum(
              [attributeByteLength, attributeByteLength, rawIndexByteLength, rawIndexByteLength],
              `${extensionPath} validation and retained allocation`,
            );
        try {
          resourceLedger?.ensureDecodedGeometryBytes(
            futureAllocation,
            `${extensionPath} validation and retained allocation`,
          );
          allocationLedger.ensureAvailable(futureAllocation, `${extensionPath} post-hook decode`);
        } catch (error) {
          decodeState.attemptsDisabled = true;
          throw error;
        }

        for (const mapping of mappings) {
          const info = rawAttributes.get(mapping.semantic);
          if (info === undefined) {
            throw new Error(`dracoDecode omitted semantic "${mapping.semantic}"`);
          }
          allocationLedger.charge(
            info.byteLength,
            `${extensionPath} ${mapping.semantic} validation snapshot`,
          );
          resourceLedger?.chargeDecodedGeometryBytes(
            info.byteLength,
            `${extensionPath} ${mapping.semantic} validation snapshot`,
          );
          attributePlans.push({
            ...mapping,
            decoded: validateDecodedAttribute(info, mapping),
          });
        }

        allocationLedger.charge(rawIndexByteLength, `${extensionPath} index validation snapshot`);
        resourceLedger?.chargeDecodedGeometryBytes(
          rawIndexByteLength,
          `${extensionPath} index validation snapshot`,
        );
        const indexSnapshot = snapshotDecodedIndices(
          indexInfo,
          ownsIndices
            ? 'dracoDecode omitted exact unsigned integer indices'
            : 'dracoDecode omitted triangle-list indices',
        );
        if (ownsIndices && indexAccessor !== undefined && indexAccessorIndex !== undefined) {
          validateIndexValues(
            indexSnapshot,
            indexAccessor,
            indexAccessorIndex,
            dracoPointCount,
            primitiveMode === GLTF_MODE_TRIANGLE_STRIP,
          );
        } else {
          validateSyntheticIndexValues(indexSnapshot, dracoPointCount, primitivePath);
        }
        decodedIndices = indexSnapshot;
      } catch (error) {
        if (error instanceof GltfResourceLimitError) throw error;
        // Draco may reorder vertices and faces. Mixing one decoded stream with
        // an uncompressed fallback stream can therefore corrupt topology even
        // when every individual accessor is shape-valid. Fall back atomically.
        useWholeFallback(formatUnknownError(error));
        continue;
      }

      // Publish only after every Draco-owned attribute and the face-list index
      // stream have been decoded and validated as one atomic domain.
      for (const plan of attributePlans) {
        const decodedAccessorIndex = appendDecodedAccessor(
          gltf,
          buffers,
          plan.accessor,
          plan.decoded,
          plan.decoded instanceof Float32Array &&
            plan.accessor.componentType !== GltfComponentType.FLOAT,
          allocationLedger,
          `${extensionPath} ${plan.semantic} decoded accessor`,
          resourceLedger,
        );
        // Define an own data property so a valid custom semantic such as
        // "__proto__" cannot invoke Object.prototype's legacy setter.
        Object.defineProperty(primitive.attributes, plan.semantic, {
          configurable: true,
          enumerable: true,
          writable: true,
          value: decodedAccessorIndex,
        });
      }
      if (decodedIndices !== undefined) {
        const decodedIndexAccessor =
          indexAccessor !== undefined && indexAccessorIndex !== undefined
            ? (() => {
                const encodedByteLength = checkedCompressionProduct(
                  decodedIndices.length,
                  componentByteSize(indexAccessor.componentType),
                  `${extensionPath} re-encoded index output`,
                );
                allocationLedger.charge(
                  encodedByteLength,
                  `${extensionPath} re-encoded index output`,
                );
                resourceLedger?.chargeDecodedGeometryBytes(
                  encodedByteLength,
                  `${extensionPath} re-encoded index output`,
                );
                return appendDecodedAccessor(
                  gltf,
                  buffers,
                  primitiveMode === GLTF_MODE_TRIANGLE_STRIP
                    ? { ...indexAccessor, count: decodedIndices.length }
                    : indexAccessor,
                  encodeIndicesForAccessor(decodedIndices, indexAccessor.componentType),
                  false,
                  allocationLedger,
                  `${extensionPath} decoded index accessor`,
                  resourceLedger,
                );
              })()
            : appendSyntheticIndexAccessor(
                gltf,
                buffers,
                decodedIndices,
                allocationLedger,
                `${extensionPath} decoded index accessor`,
                resourceLedger,
              );
        Object.defineProperty(primitive, 'indices', {
          configurable: true,
          enumerable: true,
          writable: true,
          value: decodedIndexAccessor,
        });
        // Draco's mesh API exposes connectivity as face lists. glTF permits a
        // compressed TRIANGLE_STRIP, but Draco does not retain the authored
        // strip sequence. Publish the equivalent face list through a fresh
        // count-correct accessor and normalize the cloned primitive to TRIANGLES.
        if (primitiveMode === GLTF_MODE_TRIANGLE_STRIP) {
          Object.defineProperty(primitive, 'mode', {
            configurable: true,
            enumerable: true,
            writable: true,
            value: GLTF_MODE_TRIANGLES,
          });
        }
      }
      stripDracoExtension(primitive);
    }
  }
}
