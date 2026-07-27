// Built-in glTF geometry compression decoders.
//
// Keep the decoder modules lazy: uncompressed assets must not pay the WASM
// initialization cost. Host hooks remain supported and take precedence.
//
// Draco decoder wrapper + WASM: Google Draco 1.5.7, Apache-2.0.
// https://github.com/google/draco

import type {
  DracoDecodeFn,
  DracoDecodeResult,
  DracoTypedArray,
  MeshoptDecodeFn,
} from './compression.js';
import { GltfComponentType } from './gltfTypes.js';
import {
  COMPRESSION_DECODE_BUDGET_BYTES,
  validateCompressionInputBudget,
} from './compressionLimits.js';
import type { ImportResourceLedger } from './importResourceBudget.js';

interface DracoNumericArrayLike {
  size(): number;
  GetValue(index: number): number;
}

type DracoFloat32ArrayLike = DracoNumericArrayLike;
type DracoInt32ArrayLike = DracoNumericArrayLike;

interface DracoStatusLike {
  ok(): boolean;
  error_msg(): string;
}

interface DracoAttributeLike {
  readonly ptr?: number;
  num_components(): number;
  data_type(): number;
  normalized(): boolean;
}

interface DracoDecoderLike {
  DecodeBufferToMesh(buffer: DracoDecoderBufferLike, mesh: DracoMeshLike): DracoStatusLike;
  GetAttributeByUniqueId(mesh: DracoMeshLike, uniqueId: number): DracoAttributeLike | null;
  GetAttributeFloatForAllPoints(
    mesh: DracoMeshLike,
    attribute: DracoAttributeLike,
    out: DracoFloat32ArrayLike,
  ): boolean;
  GetAttributeInt8ForAllPoints(
    mesh: DracoMeshLike,
    attribute: DracoAttributeLike,
    out: DracoNumericArrayLike,
  ): boolean;
  GetAttributeUInt8ForAllPoints(
    mesh: DracoMeshLike,
    attribute: DracoAttributeLike,
    out: DracoNumericArrayLike,
  ): boolean;
  GetAttributeInt16ForAllPoints(
    mesh: DracoMeshLike,
    attribute: DracoAttributeLike,
    out: DracoNumericArrayLike,
  ): boolean;
  GetAttributeUInt16ForAllPoints(
    mesh: DracoMeshLike,
    attribute: DracoAttributeLike,
    out: DracoNumericArrayLike,
  ): boolean;
  GetFaceFromMesh(mesh: DracoMeshLike, faceIndex: number, out: DracoInt32ArrayLike): boolean;
}

interface DracoDecoderBufferLike {
  Init(data: Int8Array, byteLength: number): void;
}

interface DracoMeshLike {
  num_faces(): number;
  num_points(): number;
}

interface DracoDecoderModuleLike {
  readonly DT_INT8: number;
  readonly DT_UINT8: number;
  readonly DT_INT16: number;
  readonly DT_UINT16: number;
  readonly DT_UINT32: number;
  readonly DT_FLOAT32: number;
  Decoder: new () => DracoDecoderLike;
  DecoderBuffer: new () => DracoDecoderBufferLike;
  Mesh: new () => DracoMeshLike;
  DracoFloat32Array: new () => DracoFloat32ArrayLike;
  DracoInt8Array: new () => DracoNumericArrayLike;
  DracoUInt8Array: new () => DracoNumericArrayLike;
  DracoInt16Array: new () => DracoNumericArrayLike;
  DracoUInt16Array: new () => DracoNumericArrayLike;
  DracoInt32Array: new () => DracoInt32ArrayLike;
  destroy(value: unknown): void;
}

interface MeshoptDecoderLike {
  readonly supported: boolean;
  readonly ready: Promise<void>;
  decodeGltfBuffer(
    target: Uint8Array,
    count: number,
    byteStride: number,
    source: Uint8Array,
    mode: string,
    filter?: string,
  ): void;
}

/** @internal Backward-compatible test-facing name for the shared ceiling. */
export const BUILTIN_COMPRESSION_DECODE_BUDGET_BYTES = COMPRESSION_DECODE_BUDGET_BYTES;

/** @internal Per-import limits for adapter-owned built-in codec allocations. */
export interface BuiltinCompressionDecoderLimits {
  /** Zero disables this ceiling while retaining safe-integer validation. */
  readonly maxCompressedInputBytes?: number;
  /** Zero disables this ceiling while retaining safe-integer validation. */
  readonly maxDecodedOutputBytes?: number;
  /**
   * Called with the cumulative JS output planned by one decode before the next
   * adapter-owned typed array is allocated. Import paths use this to preserve
   * typed public resource-limit errors.
   */
  readonly beforeDecodedAllocation?: (
    plannedByteLength: number,
    path: string,
    additionalByteLength: number,
  ) => void;
}

interface NormalizedBuiltinCompressionDecoderLimits {
  readonly maxCompressedInputBytes: number;
  readonly maxDecodedOutputBytes: number;
  readonly beforeDecodedAllocation:
    | ((
        plannedByteLength: number,
        path: string,
        additionalByteLength: number,
      ) => void)
    | undefined;
}

function safeInteger(value: unknown, path: string, minimum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new RangeError(`${path} must be a safe integer >= ${minimum}; received ${String(value)}`);
  }
  return value as number;
}

function normalizeBuiltinCompressionDecoderLimits(
  limits: BuiltinCompressionDecoderLimits,
): NormalizedBuiltinCompressionDecoderLimits {
  const maxCompressedInputBytes = safeInteger(
    limits.maxCompressedInputBytes ?? COMPRESSION_DECODE_BUDGET_BYTES,
    'built-in compression maxCompressedInputBytes',
    0,
  );
  const maxDecodedOutputBytes = safeInteger(
    limits.maxDecodedOutputBytes ?? COMPRESSION_DECODE_BUDGET_BYTES,
    'built-in compression maxDecodedOutputBytes',
    0,
  );
  if (
    limits.beforeDecodedAllocation !== undefined &&
    typeof limits.beforeDecodedAllocation !== 'function'
  ) {
    throw new TypeError('built-in compression beforeDecodedAllocation must be a function');
  }
  return {
    maxCompressedInputBytes,
    maxDecodedOutputBytes,
    beforeDecodedAllocation: limits.beforeDecodedAllocation,
  };
}

function checkedProduct(left: number, right: number, path: string): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`${path} is not a safe integer`);
  }
  return result;
}

function validatedDecodedByteTotal(
  current: number,
  additional: number,
  path: string,
  limits: NormalizedBuiltinCompressionDecoderLimits,
): number {
  const validatedCurrent = safeInteger(current, `${path} current decoded bytes`, 0);
  const validatedAdditional = safeInteger(additional, `${path} additional decoded bytes`, 0);
  if (validatedAdditional > Number.MAX_SAFE_INTEGER - validatedCurrent) {
    throw new RangeError(`${path} exceeds the safe integer range`);
  }
  const next = validatedCurrent + validatedAdditional;
  if (limits.maxDecodedOutputBytes !== 0 && next > limits.maxDecodedOutputBytes) {
    throw new RangeError(
      `${path} exceeds the built-in decode memory budget of ` +
      `${limits.maxDecodedOutputBytes} bytes`,
    );
  }
  return next;
}

function reserveDecodedBytes(
  current: number,
  additional: number,
  path: string,
  limits: NormalizedBuiltinCompressionDecoderLimits,
): number {
  const next = validatedDecodedByteTotal(current, additional, path, limits);
  const validatedAdditional = safeInteger(additional, `${path} additional decoded bytes`, 0);
  limits.beforeDecodedAllocation?.(next, path, validatedAdditional);
  return next;
}

function accessorComponentCount(type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4'): number {
  switch (type) {
    case 'SCALAR':
      return 1;
    case 'VEC2':
      return 2;
    case 'VEC3':
      return 3;
    case 'VEC4':
      return 4;
  }
}

function dracoDataTypeForComponentType(
  module: DracoDecoderModuleLike,
  componentType: GltfComponentType,
): number {
  switch (componentType) {
    case GltfComponentType.BYTE:
      return module.DT_INT8;
    case GltfComponentType.UNSIGNED_BYTE:
      return module.DT_UINT8;
    case GltfComponentType.SHORT:
      return module.DT_INT16;
    case GltfComponentType.UNSIGNED_SHORT:
      return module.DT_UINT16;
    case GltfComponentType.FLOAT:
      return module.DT_FLOAT32;
    default:
      throw new TypeError(`Unsupported Draco accessor componentType ${String(componentType)}`);
  }
}

let dracoModulePromise: Promise<DracoDecoderModuleLike> | undefined;
let meshoptDecoderPromise: Promise<MeshoptDecoderLike> | undefined;

const DRACO_DECODER_WASM_URL = new URL('./assets/draco_decoder.wasm', import.meta.url);

async function loadDracoWasmBinary(): Promise<ArrayBuffer> {
  if (DRACO_DECODER_WASM_URL.protocol === 'file:') {
    // Keep the Node fallback out of browser bundles. The statically constructed
    // URL above remains visible to Vite/Rollup so the package-owned WASM is
    // emitted as an asset in browser builds.
    const nodeFsSpecifier = `node:${'fs/promises'}`;
    const nodeFs = (await import(/* @vite-ignore */ nodeFsSpecifier)) as {
      readonly readFile: (url: URL) => Promise<Uint8Array>;
    };
    const bytes = await nodeFs.readFile(DRACO_DECODER_WASM_URL);
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return copy.buffer;
  }

  const response = await fetch(DRACO_DECODER_WASM_URL);
  if (!response.ok) {
    throw new Error(
      `Failed to load the built-in Draco decoder WASM ` +
        `(${response.status} ${response.statusText}) from ${DRACO_DECODER_WASM_URL.href}`,
    );
  }
  return response.arrayBuffer();
}

function loadDracoModule(): Promise<DracoDecoderModuleLike> {
  if (dracoModulePromise === undefined) {
    const pending = Promise.all([
      import('./vendor/draco_decoder_browser.js'),
      loadDracoWasmBinary(),
    ]).then(async ([{ default: createDecoderModule }, wasmBinary]) => {
      if (typeof createDecoderModule !== 'function') {
        throw new TypeError('Draco decoder module did not export a factory function');
      }
      return createDecoderModule({
        wasmBinary,
        locateFile: (path: string) => (path.endsWith('.wasm') ? DRACO_DECODER_WASM_URL.href : path),
      }) as Promise<DracoDecoderModuleLike>;
    });
    dracoModulePromise = pending.catch((error: unknown) => {
      dracoModulePromise = undefined;
      throw error;
    });
  }
  return dracoModulePromise;
}

async function loadMeshoptDecoder(): Promise<MeshoptDecoderLike> {
  if (meshoptDecoderPromise === undefined) {
    const pending = import('meshoptimizer/decoder').then(async ({ MeshoptDecoder }) => {
      await MeshoptDecoder.ready;
      if (!MeshoptDecoder.supported) {
        throw new Error('meshoptimizer decoder is not supported in this runtime');
      }
      return MeshoptDecoder;
    });
    meshoptDecoderPromise = pending.catch((error: unknown) => {
      meshoptDecoderPromise = undefined;
      throw error;
    });
  }
  return meshoptDecoderPromise;
}

function safelyDestroy(module: DracoDecoderModuleLike, value: unknown): void {
  if (value == null) return;
  try {
    module.destroy(value);
  } catch {
    // Preserve the decode result/error if a third-party destructor rejects an
    // already-invalid object.
  }
}

/** Create a lazy KHR_draco_mesh_compression decoder for one import policy. */
export function createBuiltinDracoDecode(
  limits: BuiltinCompressionDecoderLimits = {},
): DracoDecodeFn {
  const normalizedLimits = normalizeBuiltinCompressionDecoderLimits(limits);
  return async (compressed, attributeIds, context) => {
    validateCompressionInputBudget(
      compressed.byteLength,
      'Draco compressed input',
      normalizedLimits.maxCompressedInputBytes,
    );
    let reservedDecodedBytes = 0;
    const module = await loadDracoModule();
    let buffer: DracoDecoderBufferLike | undefined;
    let decoder: DracoDecoderLike | undefined;
    let mesh: DracoMeshLike | undefined;
    let status: DracoStatusLike | undefined;

    try {
      buffer = new module.DecoderBuffer();
      decoder = new module.Decoder();
      mesh = new module.Mesh();
      const bytes = new Int8Array(compressed.buffer, compressed.byteOffset, compressed.byteLength);
      buffer.Init(bytes, bytes.byteLength);
      status = decoder.DecodeBufferToMesh(buffer, mesh);
      if (!status.ok()) {
        throw new Error(
          `Draco mesh decode failed: ${status.error_msg() || 'unknown decoder error'}`,
        );
      }

      const pointCount = safeInteger(mesh.num_points(), 'Draco mesh point count', 0);

      const attributes: Record<string, DracoTypedArray> = Object.create(null) as Record<
        string,
        DracoTypedArray
      >;
      for (const [semantic, uniqueId] of Object.entries(attributeIds)) {
        if (!Number.isSafeInteger(uniqueId) || uniqueId < 0 || uniqueId > 0xffff_ffff) {
          throw new Error(
            `Draco attribute ${semantic} unique id ${uniqueId} is outside the ` +
              'decoder uint32 domain 0..4294967295',
          );
        }
        const attribute = decoder.GetAttributeByUniqueId(mesh, uniqueId);
        if (attribute == null || attribute.ptr === 0) {
          throw new Error(`Draco attribute ${semantic} (unique id ${uniqueId}) was not found`);
        }
        const schema = context?.attributes[semantic];
        if (schema === undefined) {
          throw new Error(`Draco attribute ${semantic} is missing its declared accessor schema`);
        }
        const declaredPointCount = safeInteger(
          schema.count,
          `Draco attribute ${semantic} declared point count`,
          1,
        );
        if (pointCount !== declaredPointCount) {
          throw new RangeError(
            `Draco attribute ${semantic} mesh has ${pointCount} points; ` +
              `accessor declares ${declaredPointCount}`,
          );
        }
        const componentCount = safeInteger(
          attribute.num_components(),
          `Draco attribute ${semantic} component count`,
          1,
        );
        const declaredComponentCount = accessorComponentCount(schema.type);
        if (componentCount !== declaredComponentCount) {
          throw new RangeError(
            `Draco attribute ${semantic} has ${componentCount} components; ` +
              `accessor ${schema.type} declares ${declaredComponentCount}`,
          );
        }
        const componentType: GltfComponentType = schema.componentType;
        const dataType = safeInteger(
          attribute.data_type(),
          `Draco attribute ${semantic} data type`,
          0,
        );
        const expectedDataType = dracoDataTypeForComponentType(module, componentType);
        if (dataType !== expectedDataType) {
          throw new TypeError(
            `Draco attribute ${semantic} data type ${dataType} does not match ` +
              `accessor componentType ${componentType} (Draco type ${expectedDataType})`,
          );
        }
        const normalized = attribute.normalized();
        if (typeof normalized !== 'boolean' || normalized !== schema.normalized) {
          throw new TypeError(
            `Draco attribute ${semantic} normalized=${String(normalized)} does not match ` +
              `accessor normalized=${String(schema.normalized)}`,
          );
        }
        const expectedCount = checkedProduct(
          pointCount,
          componentCount,
          `Draco attribute ${semantic} point/component product`,
        );
        const bytesPerElement = (() => {
          switch (componentType) {
            case GltfComponentType.BYTE:
            case GltfComponentType.UNSIGNED_BYTE:
              return 1;
            case GltfComponentType.SHORT:
            case GltfComponentType.UNSIGNED_SHORT:
              return 2;
            case GltfComponentType.FLOAT:
              return 4;
            default:
              throw new TypeError(
                `Draco attribute ${semantic} has unsupported componentType ${String(componentType)}`,
              );
          }
        })();
        const byteLength = checkedProduct(
          expectedCount,
          bytesPerElement,
          `Draco attribute ${semantic} decoded byte length`,
        );
        // Preserve the standalone built-in ceiling before asking the codec to
        // populate opaque temporary/WASM storage. Commit the public geometry
        // charge only after that succeeds, immediately before the adapter
        // allocates its retained JS typed array. A later validation or attribute
        // failure must not release this charge.
        validatedDecodedByteTotal(
          reservedDecodedBytes,
          byteLength,
          `Draco attribute ${semantic}`,
          normalizedLimits,
        );
        const requireDecodedAttribute = (ok: boolean): void => {
          if (!ok) {
            throw new Error(`Draco attribute ${semantic} could not be decoded`);
          }
        };
        let data: DracoNumericArrayLike | undefined;
        try {
          let ok: boolean;
          let output: DracoTypedArray;
          switch (componentType) {
            case GltfComponentType.BYTE:
              data = new module.DracoInt8Array();
              ok = decoder.GetAttributeInt8ForAllPoints(mesh, attribute, data);
              requireDecodedAttribute(ok);
              reservedDecodedBytes = reserveDecodedBytes(
                reservedDecodedBytes,
                byteLength,
                `Draco attribute ${semantic}`,
                normalizedLimits,
              );
              output = new Int8Array(expectedCount);
              break;
            case GltfComponentType.UNSIGNED_BYTE:
              data = new module.DracoUInt8Array();
              ok = decoder.GetAttributeUInt8ForAllPoints(mesh, attribute, data);
              requireDecodedAttribute(ok);
              reservedDecodedBytes = reserveDecodedBytes(
                reservedDecodedBytes,
                byteLength,
                `Draco attribute ${semantic}`,
                normalizedLimits,
              );
              output = new Uint8Array(expectedCount);
              break;
            case GltfComponentType.SHORT:
              data = new module.DracoInt16Array();
              ok = decoder.GetAttributeInt16ForAllPoints(mesh, attribute, data);
              requireDecodedAttribute(ok);
              reservedDecodedBytes = reserveDecodedBytes(
                reservedDecodedBytes,
                byteLength,
                `Draco attribute ${semantic}`,
                normalizedLimits,
              );
              output = new Int16Array(expectedCount);
              break;
            case GltfComponentType.UNSIGNED_SHORT:
              data = new module.DracoUInt16Array();
              ok = decoder.GetAttributeUInt16ForAllPoints(mesh, attribute, data);
              requireDecodedAttribute(ok);
              reservedDecodedBytes = reserveDecodedBytes(
                reservedDecodedBytes,
                byteLength,
                `Draco attribute ${semantic}`,
                normalizedLimits,
              );
              output = new Uint16Array(expectedCount);
              break;
            case GltfComponentType.FLOAT:
              data = new module.DracoFloat32Array();
              ok = decoder.GetAttributeFloatForAllPoints(mesh, attribute, data);
              requireDecodedAttribute(ok);
              reservedDecodedBytes = reserveDecodedBytes(
                reservedDecodedBytes,
                byteLength,
                `Draco attribute ${semantic}`,
                normalizedLimits,
              );
              output = new Float32Array(expectedCount);
              break;
            default:
              throw new TypeError(
                `Draco attribute ${semantic} has unsupported componentType ${String(componentType)}`,
              );
          }
          const count = data.size();
          if (!Number.isSafeInteger(count) || count < 0) {
            throw new Error(`Draco attribute ${semantic} reported invalid length ${count}`);
          }
          if (count !== expectedCount) {
            throw new RangeError(
              `Draco attribute ${semantic} reported ${count} values; expected ` +
                `${pointCount} points × ${componentCount} components = ${expectedCount}`,
            );
          }
          if (output.byteLength !== byteLength) {
            throw new Error(
              `Draco attribute ${semantic} allocated ${output.byteLength} bytes; expected ${byteLength}`,
            );
          }
          for (let index = 0; index < count; index += 1) {
            const value = data.GetValue(index);
            if (!Number.isFinite(value)) {
              throw new RangeError(
                `Draco attribute ${semantic} contains a non-finite value at ${index}`,
              );
            }
            output[index] = value;
            const stored = output[index]!;
            if (
              !Number.isFinite(stored) ||
              (!(output instanceof Float32Array) && stored !== value)
            ) {
              throw new RangeError(
                `Draco attribute ${semantic} value ${String(value)} at ${index} cannot be represented ` +
                  `exactly by accessor componentType ${componentType}`,
              );
            }
          }
          attributes[semantic] = output;
        } finally {
          safelyDestroy(module, data);
        }
      }

      const faceCount = mesh.num_faces();
      if (!Number.isSafeInteger(faceCount) || faceCount < 0) {
        throw new Error(`Draco mesh reported invalid face count ${faceCount}`);
      }
      const indexCount = checkedProduct(faceCount, 3, 'Draco mesh face/index product');
      const indexByteLength = checkedProduct(
        indexCount,
        Uint32Array.BYTES_PER_ELEMENT,
        'Draco index decoded byte length',
      );
      reservedDecodedBytes = reserveDecodedBytes(
        reservedDecodedBytes,
        indexByteLength,
        'Draco indices',
        normalizedLimits,
      );
      const indices = new Uint32Array(indexCount);
      const face = new module.DracoInt32Array();
      try {
        for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
          if (decoder.GetFaceFromMesh(mesh, faceIndex, face) === false) {
            throw new Error(`Draco face ${faceIndex} could not be decoded`);
          }
          const offset = faceIndex * 3;
          for (let corner = 0; corner < 3; corner += 1) {
            const value = face.GetValue(corner);
            if (
              !Number.isSafeInteger(value) ||
              value < 0 ||
              value > 0xffff_ffff ||
              value >= pointCount
            ) {
              throw new RangeError(
                `Draco face ${faceIndex} corner ${corner} has invalid point index ${String(value)} ` +
                  `for ${pointCount} decoded points`,
              );
            }
            indices[offset + corner] = value;
          }
        }
      } finally {
        safelyDestroy(module, face);
      }

      const result: DracoDecodeResult = { attributes, indices };
      return result;
    } finally {
      safelyDestroy(module, status);
      safelyDestroy(module, mesh);
      safelyDestroy(module, decoder);
      safelyDestroy(module, buffer);
    }
  };
}

/** Built-in KHR_draco_mesh_compression decoder with conservative standalone defaults. */
export const builtinDracoDecode: DracoDecodeFn = createBuiltinDracoDecode();

/**
 * Built-in EXT/KHR_meshopt_compression decoder, loaded on first use.
 * meshoptimizer 1.1.1, MIT: https://github.com/zeux/meshoptimizer
 */
export function createBuiltinMeshoptDecode(
  limits: BuiltinCompressionDecoderLimits = {},
): MeshoptDecodeFn {
  const normalizedLimits = normalizeBuiltinCompressionDecoderLimits(limits);
  return async (compressed, count, byteStride, mode, filter) => {
    validateCompressionInputBudget(
      compressed.byteLength,
      'meshopt compressed input',
      normalizedLimits.maxCompressedInputBytes,
    );
    const validatedCount = safeInteger(count, 'meshopt count', 1);
    const validatedStride = safeInteger(byteStride, 'meshopt byteStride', 1);
    const targetByteLength = checkedProduct(
      validatedCount,
      validatedStride,
      'meshopt count × byteStride',
    );
    // Input bytes have their own independently configurable ceiling. The decoded
    // output ledger therefore starts at zero instead of combining both domains.
    validatedDecodedByteTotal(0, targetByteLength, 'meshopt decoded buffer', normalizedLimits);
    const decoder = await loadMeshoptDecoder();
    reserveDecodedBytes(0, targetByteLength, 'meshopt decoded buffer', normalizedLimits);
    const target = new Uint8Array(targetByteLength);
    decoder.decodeGltfBuffer(
      target,
      validatedCount,
      validatedStride,
      compressed,
      mode,
      filter === 'NONE' ? undefined : filter,
    );
    return target;
  };
}

/** Built-in meshopt decoder with conservative standalone defaults. */
export const builtinMeshoptDecode: MeshoptDecodeFn = createBuiltinMeshoptDecode();

/**
 * Create both built-ins against the authoritative ledger for one public import.
 *
 * Acquired input is already charged by that ledger, so the duplicate codec
 * input/output ceilings are disabled. The callback commits each incremental
 * adapter-owned JS output immediately before allocation, preserving both
 * monotonic failure accounting and GltfResourceLimitError.
 */
export function createImportBuiltinCompressionDecoders(resourceLedger: ImportResourceLedger): {
  readonly dracoDecode: DracoDecodeFn;
  readonly meshoptDecode: MeshoptDecodeFn;
} {
  const limits: BuiltinCompressionDecoderLimits = {
    maxCompressedInputBytes: 0,
    maxDecodedOutputBytes: 0,
    beforeDecodedAllocation: (_plannedByteLength, path, additionalByteLength) => {
      resourceLedger.chargeDecodedGeometryBytes(additionalByteLength, `built-in ${path}`);
    },
  };
  return {
    dracoDecode: createBuiltinDracoDecode(limits),
    meshoptDecode: createBuiltinMeshoptDecode(limits),
  };
}
