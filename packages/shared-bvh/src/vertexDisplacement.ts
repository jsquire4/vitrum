import {
  getPrimitiveActiveColorSet,
  getPrimitiveUvSet,
  sparseArrayHasDefinedEntry,
  sparseArrayOwnIndices,
  type MaterialSpec,
  type PrimitiveUvSets,
  type PrimitiveColorSets,
  type TextureRef,
  type TextureWrapMode,
  type UvTransform,
} from '@vitrum/core';
import { halfToFloat } from './textureDecode.js';

interface RawHeightPixels {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly channels: 1 | 2 | 3 | 4;
  readonly decode: (value: number) => number;
  readonly filter: 'nearest' | 'linear';
}

type DisplacementWarningSink = (warning: string) => void;

const MAX_DISPLACEMENT_SUBDIVISIONS = 4;
const MAX_MICRODISPLACED_TRIANGLES = 262_144;
const MAX_DISPLACEMENT_CPU_BYTES = 512 * 1024 * 1024;
const FLOAT32_TAG = '[object Float32Array]';
const UINT8_TAG = '[object Uint8Array]';
const UINT8_CLAMPED_TAG = '[object Uint8ClampedArray]';
const UINT16_TAG = '[object Uint16Array]';
const UINT32_TAG = '[object Uint32Array]';

export interface MicrodisplacedMeshGeometry {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices: Uint32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;
  readonly subdivisions: number;
}

function textureHandleType(handle: unknown): string {
  return handle == null ? 'null' : Object.prototype.toString.call(handle);
}

function textureRefSourcePath(ref: TextureRef | undefined): string | undefined {
  if (ref == null) return undefined;
  for (const symbol of Object.getOwnPropertySymbols(ref)) {
    if (symbol.description !== 'vitrum.gltf.textureRefSource') continue;
    const value = (ref as unknown as Record<symbol, unknown>)[symbol];
    if (value == null || typeof value !== 'object') return undefined;
    const path = (value as { readonly path?: unknown }).path;
    return typeof path === 'string' ? path : undefined;
  }
  return undefined;
}

function warningPrefix(primitiveId: string, ref?: TextureRef): string {
  const sourcePath = textureRefSourcePath(ref);
  return `Primitive "${primitiveId}" displacementMap${sourcePath !== undefined ? ` at ${sourcePath}` : ''}`;
}

function displacementError(
  primitiveId: string,
  ref: TextureRef | undefined,
  reason: string,
): Error {
  return new Error(`${warningPrefix(primitiveId, ref)} ${reason}`);
}

function assertFiniteFloat32(
  value: unknown,
  primitiveId: string,
  ref: TextureRef | undefined,
  path: string,
): asserts value is number {
  if (
    typeof value !== 'number' ||
    !Number.isFinite(value) ||
    !Number.isFinite(Math.fround(value))
  ) {
    throw displacementError(
      primitiveId,
      ref,
      `${path} must be finite and representable as float32 (got ${String(value)}).`,
    );
  }
}

function assertAllocationBudget(
  byteCount: bigint,
  primitiveId: string,
  ref: TextureRef,
  label: string,
): void {
  if (byteCount > BigInt(MAX_DISPLACEMENT_CPU_BYTES)) {
    throw displacementError(
      primitiveId,
      ref,
      `${label} requires ${byteCount.toString()} CPU bytes, above the 512 MiB staging budget.`,
    );
  }
}

function handlePayload(handle: unknown): {
  readonly width: unknown;
  readonly height: unknown;
  readonly data: ArrayLike<number> | undefined;
  readonly hint?: { readonly channels?: number; readonly dataType?: string };
} | null {
  if (handle == null || typeof handle !== 'object') return null;
  const h = handle as {
    readonly width?: number;
    readonly height?: number;
    readonly data?: ArrayLike<number>;
    readonly image?: {
      readonly width?: number;
      readonly height?: number;
      readonly data?: ArrayLike<number>;
      readonly channels?: number;
      readonly dataType?: string;
    };
    readonly __vitrum_hint__?: { readonly channels?: number; readonly dataType?: string };
    readonly channels?: number;
    readonly dataType?: string;
  };
  // One coherent payload only: dimensions and data may not be borrowed across
  // a partial raw handle and a separate image object.
  const source = h.data != null ? h : h.image;
  if (source == null) return null;
  const hint = h.__vitrum_hint__ ?? (
    source.channels != null || source.dataType != null
      ? {
          ...(source.channels != null ? { channels: source.channels } : {}),
          ...(source.dataType != null ? { dataType: source.dataType } : {}),
        }
      : undefined
  );
  return {
    width: source.width,
    height: source.height,
    data: source.data,
    ...(hint != null ? { hint } : {}),
  };
}

function decoderFor(
  data: ArrayLike<number>,
  dataType: string | undefined,
  primitiveId: string,
  ref: TextureRef,
): (value: number) => number {
  const backing = Object.prototype.toString.call(data);
  if (!ArrayBuffer.isView(data)) {
    throw displacementError(primitiveId, ref, `pixel backing ${backing} is unsupported.`);
  }
  if (backing === UINT8_TAG || backing === UINT8_CLAMPED_TAG) {
    if (dataType != null && dataType !== 'uint8') {
      throw displacementError(
        primitiveId,
        ref,
        `dataType "${dataType}" does not match ${backing}; expected "uint8".`,
      );
    }
    return (value) => value / 255;
  }
  if (backing === UINT16_TAG) {
    if (dataType === 'uint16') return (value) => value / 65535;
    if (dataType === 'float16' || dataType === 'half-float') return halfToFloat;
    throw displacementError(
      primitiveId,
      ref,
      `Uint16Array displacement pixels require explicit dataType "uint16" or "float16" (got ${String(dataType)}).`,
    );
  }
  if (backing === FLOAT32_TAG) {
    if (dataType != null && dataType !== 'float32') {
      throw displacementError(
        primitiveId,
        ref,
        `dataType "${dataType}" does not match Float32Array; expected "float32".`,
      );
    }
    return (value) => value;
  }
  throw displacementError(
    primitiveId,
    ref,
    `pixel backing ${backing} is unsupported; use Uint8Array/Uint8ClampedArray, explicit Uint16Array, or Float32Array.`,
  );
}

function readHeightPixels(
  ref: TextureRef,
  primitiveId: string,
): RawHeightPixels {
  const payload = handlePayload(ref.handle);
  if (payload == null) {
    throw displacementError(
      primitiveId,
      ref,
      `handle is not CPU-readable (${textureHandleType(ref.handle)}).`,
    );
  }
  const { width, height, data, hint } = payload;
  if (
    typeof width !== 'number' ||
    typeof height !== 'number' ||
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width <= 0 ||
    height <= 0
  ) {
    throw displacementError(
      primitiveId,
      ref,
      `width and height must be positive safe integers (got ${String(width)}x${String(height)}).`,
    );
  }
  if (data == null || typeof data.length !== 'number') {
    throw displacementError(primitiveId, ref, 'handle must expose CPU pixel data.');
  }
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount)) {
    throw displacementError(primitiveId, ref, 'width*height exceeds safe integer arithmetic.');
  }
  if (!Number.isSafeInteger(data.length) || data.length < 0) {
    throw displacementError(
      primitiveId,
      ref,
      `data.length must be a non-negative safe integer (got ${String(data.length)}).`,
    );
  }
  const inferredChannels = data.length / pixelCount;
  const channels = hint?.channels ?? inferredChannels;
  if (!Number.isSafeInteger(channels) || channels < 1 || channels > 4) {
    throw displacementError(
      primitiveId,
      ref,
      `channels must be an exact integer in [1, 4] (got ${String(channels)}).`,
    );
  }
  const expectedValueCount = pixelCount * channels;
  if (!Number.isSafeInteger(expectedValueCount) || data.length !== expectedValueCount) {
    throw displacementError(
      primitiveId,
      ref,
      `data length ${data.length} must equal width*height*channels (${expectedValueCount}).`,
    );
  }
  const decode = decoderFor(data, hint?.dataType, primitiveId, ref);
  for (let index = 0; index < data.length; index += 1) {
    const decoded = decode(Number(data[index]));
    assertFiniteFloat32(decoded, primitiveId, ref, `decoded pixel[${index}]`);
  }
  const filter = displacementFilter(ref, primitiveId);
  return {
    width,
    height,
    data,
    channels: channels as 1 | 2 | 3 | 4,
    decode,
    filter,
  };
}

function displacementFilter(ref: TextureRef, primitiveId: string): 'nearest' | 'linear' {
  for (const [field, value] of [
    ['magFilter', ref.magFilter],
    ['minFilter', ref.minFilter],
  ] as const) {
    if (value != null && value !== 'nearest' && value !== 'linear') {
      throw displacementError(primitiveId, ref, `${field} must be "nearest" or "linear".`);
    }
  }
  if (
    ref.magFilter != null &&
    ref.minFilter != null &&
    ref.magFilter !== ref.minFilter
  ) {
    throw displacementError(
      primitiveId,
      ref,
      'CPU displacement requires matching magFilter/minFilter because no screen-space LOD exists.',
    );
  }
  if (ref.mipFilter != null && ref.mipFilter !== 'none') {
    throw displacementError(
      primitiveId,
      ref,
      `mipFilter "${String(ref.mipFilter)}" cannot be honored by the base-level CPU displacement sampler.`,
    );
  }
  return ref.magFilter ?? ref.minFilter ?? 'linear';
}

function wrapCoord(
  value: number,
  mode: TextureWrapMode | undefined,
  primitiveId: string,
  ref: TextureRef,
  path: string,
): number {
  assertFiniteFloat32(value, primitiveId, ref, path);
  switch (mode ?? 'repeat') {
    case 'clamp-to-edge':
      return Math.min(1, Math.max(0, value));
    case 'mirrored-repeat': {
      const period = Math.floor(value);
      const frac = value - period;
      return Math.abs(period % 2) === 1 ? 1 - frac : frac;
    }
    case 'repeat':
      return value - Math.floor(value);
    default:
      throw displacementError(
        primitiveId,
        ref,
        `${path} uses unsupported wrap mode ${String(mode)}.`,
      );
  }
}

function applyUvTransform(
  u: number,
  v: number,
  transform: UvTransform | undefined,
  primitiveId: string,
  ref: TextureRef,
): readonly [number, number] {
  const sx = transform?.scale?.[0] ?? 1;
  const sy = transform?.scale?.[1] ?? 1;
  const rotation = transform?.rotation ?? 0;
  assertFiniteFloat32(u, primitiveId, ref, 'UV u');
  assertFiniteFloat32(v, primitiveId, ref, 'UV v');
  assertFiniteFloat32(sx, primitiveId, ref, 'transform.scale[0]');
  assertFiniteFloat32(sy, primitiveId, ref, 'transform.scale[1]');
  assertFiniteFloat32(rotation, primitiveId, ref, 'transform.rotation');
  if (transform?.scale != null && transform.scale.length !== 2) {
    throw displacementError(primitiveId, ref, 'transform.scale must have length 2.');
  }
  if (transform?.offset != null && transform.offset.length !== 2) {
    throw displacementError(primitiveId, ref, 'transform.offset must have length 2.');
  }
  const ox = transform?.offset?.[0] ?? 0;
  const oy = transform?.offset?.[1] ?? 0;
  assertFiniteFloat32(ox, primitiveId, ref, 'transform.offset[0]');
  assertFiniteFloat32(oy, primitiveId, ref, 'transform.offset[1]');
  const x = u * sx;
  const y = v * sy;
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  const transformedU = x * c - y * s + ox;
  const transformedV = x * s + y * c + oy;
  assertFiniteFloat32(transformedU, primitiveId, ref, 'transformed UV u');
  assertFiniteFloat32(transformedV, primitiveId, ref, 'transformed UV v');
  return [transformedU, transformedV];
}

function redAt(pixels: RawHeightPixels, x: number, y: number): number {
  const clampedX = Math.min(pixels.width - 1, Math.max(0, x));
  const clampedY = Math.min(pixels.height - 1, Math.max(0, y));
  const offset = (clampedY * pixels.width + clampedX) * pixels.channels;
  return pixels.decode(Number(pixels.data[offset]));
}

function sampleHeight(
  pixels: RawHeightPixels,
  u: number,
  v: number,
  ref: TextureRef,
  primitiveId: string,
): number {
  const [tu, tv] = applyUvTransform(u, v, ref.transform, primitiveId, ref);
  const wu = wrapCoord(tu, ref.wrapS, primitiveId, ref, 'transformed UV u');
  const wv = wrapCoord(tv, ref.wrapT, primitiveId, ref, 'transformed UV v');
  if (pixels.width === 1 && pixels.height === 1) {
    return redAt(pixels, 0, 0);
  }
  const x = wu * Math.max(0, pixels.width - 1);
  const y = wv * Math.max(0, pixels.height - 1);
  if (pixels.filter === 'nearest') return redAt(pixels, Math.round(x), Math.round(y));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(pixels.width - 1, x0 + 1);
  const y1 = Math.min(pixels.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const h00 = redAt(pixels, x0, y0);
  const h10 = redAt(pixels, x1, y0);
  const h01 = redAt(pixels, x0, y1);
  const h11 = redAt(pixels, x1, y1);
  const hx0 = h00 * (1 - fx) + h10 * fx;
  const hx1 = h01 * (1 - fx) + h11 * fx;
  const sampled = hx0 * (1 - fy) + hx1 * fy;
  assertFiniteFloat32(sampled, primitiveId, ref, 'sampled height');
  return sampled;
}

function isFloat32Array(value: unknown): value is Float32Array {
  return ArrayBuffer.isView(value) && Object.prototype.toString.call(value) === FLOAT32_TAG;
}

function isRuntimeArray(value: unknown): boolean {
  return Array.isArray(value);
}

function assertFloat32Stream(
  value: unknown,
  expectedLength: number,
  primitiveId: string,
  ref: TextureRef,
  path: string,
): asserts value is Float32Array {
  if (!isFloat32Array(value)) {
    throw displacementError(primitiveId, ref, `${path} must be a Float32Array.`);
  }
  if (value.length !== expectedLength) {
    throw displacementError(
      primitiveId,
      ref,
      `${path}.length must be exactly ${expectedLength} (got ${value.length}).`,
    );
  }
  for (let index = 0; index < value.length; index += 1) {
    assertFiniteFloat32(value[index], primitiveId, ref, `${path}[${index}]`);
  }
}

function assertMatchingStreams(
  left: Float32Array,
  right: Float32Array,
  primitiveId: string,
  ref: TextureRef,
  path: string,
): void {
  if (left.length !== right.length) {
    throw displacementError(primitiveId, ref, `${path} aliases have different lengths.`);
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      throw displacementError(
        primitiveId,
        ref,
        `${path} aliases differ at component ${index}.`,
      );
    }
  }
}

function validateTextureContract(ref: TextureRef, primitiveId: string): number {
  const texCoord = ref.texCoord ?? 0;
  if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
    throw displacementError(
      primitiveId,
      ref,
      `texCoord must be a non-negative safe integer (got ${String(texCoord)}).`,
    );
  }
  for (const [path, mode] of [['wrapS', ref.wrapS], ['wrapT', ref.wrapT]] as const) {
    if (
      mode != null &&
      mode !== 'repeat' &&
      mode !== 'clamp-to-edge' &&
      mode !== 'mirrored-repeat'
    ) {
      throw displacementError(primitiveId, ref, `${path} has unsupported mode ${String(mode)}.`);
    }
  }
  displacementFilter(ref, primitiveId);
  const transform = ref.transform;
  if (transform?.scale != null && transform.scale.length !== 2) {
    throw displacementError(primitiveId, ref, 'transform.scale must have length 2.');
  }
  if (transform?.offset != null && transform.offset.length !== 2) {
    throw displacementError(primitiveId, ref, 'transform.offset must have length 2.');
  }
  assertFiniteFloat32(transform?.scale?.[0] ?? 1, primitiveId, ref, 'transform.scale[0]');
  assertFiniteFloat32(transform?.scale?.[1] ?? 1, primitiveId, ref, 'transform.scale[1]');
  assertFiniteFloat32(transform?.offset?.[0] ?? 0, primitiveId, ref, 'transform.offset[0]');
  assertFiniteFloat32(transform?.offset?.[1] ?? 0, primitiveId, ref, 'transform.offset[1]');
  assertFiniteFloat32(transform?.rotation ?? 0, primitiveId, ref, 'transform.rotation');
  return texCoord;
}

type DisplacementUvInput = {
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
};

function validateUvStreams(
  input: DisplacementUvInput,
  vertexCount: number,
  texCoord: number,
  primitiveId: string,
  ref: TextureRef,
): { readonly selected: Float32Array; readonly all: PrimitiveUvSets } {
  const expectedLength = vertexCount * 2;
  if (input.uvs != null) {
    assertFloat32Stream(input.uvs, expectedLength, primitiveId, ref, 'uvs');
  }
  if (input.uv1 != null) {
    assertFloat32Stream(input.uv1, expectedLength, primitiveId, ref, 'uv1');
  }
  if (input.uvSets != null && !isRuntimeArray(input.uvSets)) {
    throw displacementError(primitiveId, ref, 'uvSets must be an array.');
  }
  const all: Array<Float32Array | undefined> = [];
  if (input.uvSets != null) {
    for (const setIndex of sparseArrayOwnIndices(input.uvSets)) {
      const scalable = input.uvSets[setIndex];
      if (scalable == null) continue;
      assertFloat32Stream(
        scalable,
        expectedLength,
        primitiveId,
        ref,
        `uvSets[${setIndex}]`,
      );
      all[setIndex] = scalable;
    }
  }
  for (const [setIndex, legacy] of [[0, input.uvs], [1, input.uv1]] as const) {
    const scalable = input.uvSets?.[setIndex];
    if (scalable != null) {
      if (legacy != null) {
        assertMatchingStreams(scalable, legacy, primitiveId, ref, `uvSets[${setIndex}]`);
      }
    } else if (legacy != null) {
      all[setIndex] = legacy;
    }
  }
  const selected = all[texCoord];
  if (selected == null) {
    throw displacementError(
      primitiveId,
      ref,
      `requests TEXCOORD_${texCoord}, but that exact UV channel is absent.`,
    );
  }
  return { selected, all };
}

function validateBaseGeometry(
  input: DisplacementUvInput & {
    readonly material: MaterialSpec;
    readonly positions: Float32Array;
    readonly normals: Float32Array;
  },
  primitiveId: string,
  ref: TextureRef,
): {
  readonly vertexCount: number;
  readonly uvSource: Float32Array;
  readonly uvSets: PrimitiveUvSets;
  readonly scale: number;
  readonly bias: number;
} {
  if (!isFloat32Array(input.positions) || input.positions.length % 3 !== 0) {
    throw displacementError(
      primitiveId,
      ref,
      'positions must be a Float32Array whose length is divisible by 3.',
    );
  }
  const vertexCount = input.positions.length / 3;
  assertFloat32Stream(input.positions, vertexCount * 3, primitiveId, ref, 'positions');
  assertFloat32Stream(input.normals, vertexCount * 3, primitiveId, ref, 'normals');
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const x = input.normals[vertex * 3]!;
    const y = input.normals[vertex * 3 + 1]!;
    const z = input.normals[vertex * 3 + 2]!;
    if (Math.hypot(x, y, z) === 0) {
      throw displacementError(primitiveId, ref, `normals vertex ${vertex} is zero-length.`);
    }
  }
  const texCoord = validateTextureContract(ref, primitiveId);
  const { selected, all } = validateUvStreams(input, vertexCount, texCoord, primitiveId, ref);
  const scale = inputMaterialScale(input, primitiveId, ref);
  return { vertexCount, uvSource: selected, uvSets: all, ...scale };
}

function inputMaterialScale(
  input: { readonly material?: MaterialSpec },
  primitiveId: string,
  ref: TextureRef,
): { readonly scale: number; readonly bias: number } {
  const scale = input.material?.displacementScale ?? 1;
  const bias = input.material?.displacementBias ?? 0;
  assertFiniteFloat32(scale, primitiveId, ref, 'material.displacementScale');
  assertFiniteFloat32(bias, primitiveId, ref, 'material.displacementBias');
  return { scale, bias };
}

function normalizeStrict(
  x: number,
  y: number,
  z: number,
  primitiveId: string,
  ref: TextureRef,
  path: string,
): readonly [number, number, number] {
  assertFiniteFloat32(x, primitiveId, ref, `${path}.x`);
  assertFiniteFloat32(y, primitiveId, ref, `${path}.y`);
  assertFiniteFloat32(z, primitiveId, ref, `${path}.z`);
  const len = Math.hypot(x, y, z);
  if (!(len > 0) || !Number.isFinite(len)) {
    throw displacementError(primitiveId, ref, `${path} must be non-zero and finite.`);
  }
  const nx = x / len;
  const ny = y / len;
  const nz = z / len;
  assertFiniteFloat32(nx, primitiveId, ref, `${path}.normalized.x`);
  assertFiniteFloat32(ny, primitiveId, ref, `${path}.normalized.y`);
  assertFiniteFloat32(nz, primitiveId, ref, `${path}.normalized.z`);
  return [nx, ny, nz];
}

/**
 * Apply CPU-readable displacement to authored mesh vertices only. For opt-in
 * diced geometry, use {@link maybeMicrodisplaceMeshGeometry}.
 */
export function maybeDisplaceMeshPositions(
  input: {
    readonly primitiveId: string;
    readonly material: MaterialSpec;
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly uvs?: Float32Array;
    readonly uv1?: Float32Array;
    readonly uvSets?: PrimitiveUvSets;
  },
): Float32Array | null {
  const ref = input.material.displacementMap;
  if (ref == null) return null;
  const { vertexCount, uvSource, scale, bias } = validateBaseGeometry(
    input,
    input.primitiveId,
    ref,
  );
  const pixels = readHeightPixels(ref, input.primitiveId);
  assertAllocationBudget(BigInt(input.positions.length) * 4n, input.primitiveId, ref, 'vertex displacement output');
  const out = new Float32Array(input.positions.length);
  for (let i = 0; i < vertexCount; i += 1) {
    const px = input.positions[i * 3]!;
    const py = input.positions[i * 3 + 1]!;
    const pz = input.positions[i * 3 + 2]!;
    const [nx, ny, nz] = normalizeStrict(
      input.normals[i * 3]!,
      input.normals[i * 3 + 1]!,
      input.normals[i * 3 + 2]!,
      input.primitiveId,
      ref,
      `normals vertex ${i}`,
    );
    const u = uvSource[i * 2]!;
    const v = uvSource[i * 2 + 1]!;
    const amount = sampleHeight(pixels, u, v, ref, input.primitiveId) * scale + bias;
    assertFiniteFloat32(amount, input.primitiveId, ref, `vertex ${i} displacement amount`);
    const ox = px + nx * amount;
    const oy = py + ny * amount;
    const oz = pz + nz * amount;
    assertFiniteFloat32(ox, input.primitiveId, ref, `output positions[${i * 3}]`);
    assertFiniteFloat32(oy, input.primitiveId, ref, `output positions[${i * 3 + 1}]`);
    assertFiniteFloat32(oz, input.primitiveId, ref, `output positions[${i * 3 + 2}]`);
    out[i * 3] = ox;
    out[i * 3 + 1] = oy;
    out[i * 3 + 2] = oz;
  }
  return out;
}


function resolveDisplacementSubdivisions(
  material: MaterialSpec,
  primitiveId: string,
): number {
  const ref = material.displacementMap;
  const raw = material.displacementSubdivisions;
  if (raw == null || raw === 0) return 0;
  if (!Number.isSafeInteger(raw) || raw < 0 || raw > MAX_DISPLACEMENT_SUBDIVISIONS) {
    throw displacementError(
      primitiveId,
      ref,
      `displacementSubdivisions must be an integer in [0, ${MAX_DISPLACEMENT_SUBDIVISIONS}] (got ${String(raw)}).`,
    );
  }
  return raw;
}

function lerpScalar(
  values: Float32Array,
  stride: number,
  ia: number,
  ib: number,
  ic: number,
  component: number,
  wa: number,
  wb: number,
  wc: number,
): number {
  return values[ia * stride + component]! * wa +
    values[ib * stride + component]! * wb +
    values[ic * stride + component]! * wc;
}

function validateTriangleSource(
  indices: Uint32Array | Uint16Array | undefined,
  vertexCount: number,
  primitiveId: string,
  ref: TextureRef,
): { readonly triangleCount: number; readonly indexAt: (offset: number) => number } {
  if (indices == null) {
    if (vertexCount % 3 !== 0) {
      throw displacementError(
        primitiveId,
        ref,
        `non-indexed vertexCount must be divisible by 3 (got ${vertexCount}).`,
      );
    }
    return { triangleCount: vertexCount / 3, indexAt: (offset) => offset };
  }
  const tag = Object.prototype.toString.call(indices);
  if (!ArrayBuffer.isView(indices) || (tag !== UINT16_TAG && tag !== UINT32_TAG)) {
    throw displacementError(
      primitiveId,
      ref,
      'indices must be an exact Uint16Array or Uint32Array.',
    );
  }
  if (indices.length % 3 !== 0) {
    throw displacementError(
      primitiveId,
      ref,
      `indices.length must be divisible by 3 (got ${indices.length}).`,
    );
  }
  for (let offset = 0; offset < indices.length; offset += 1) {
    const index = indices[offset]!;
    if (index >= vertexCount) {
      throw displacementError(
        primitiveId,
        ref,
        `indices[${offset}]=${index} is outside vertexCount ${vertexCount}.`,
      );
    }
  }
  return {
    triangleCount: indices.length / 3,
    indexAt: (offset) => indices[offset]!,
  };
}

function accumulateFaceNormal(
  positions: Float32Array,
  normals: Float64Array,
  ia: number,
  ib: number,
  ic: number,
  primitiveId: string,
  ref: TextureRef,
): void {
  const ax = positions[ia * 3]!;
  const ay = positions[ia * 3 + 1]!;
  const az = positions[ia * 3 + 2]!;
  const bx = positions[ib * 3]!;
  const by = positions[ib * 3 + 1]!;
  const bz = positions[ib * 3 + 2]!;
  const cx = positions[ic * 3]!;
  const cy = positions[ic * 3 + 1]!;
  const cz = positions[ic * 3 + 2]!;
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  assertFiniteFloat32(nx, primitiveId, ref, 'generated face normal.x');
  assertFiniteFloat32(ny, primitiveId, ref, 'generated face normal.y');
  assertFiniteFloat32(nz, primitiveId, ref, 'generated face normal.z');
  if (nx === 0 && ny === 0 && nz === 0) return;
  for (const i of [ia, ib, ic]) {
    const sx = normals[i * 3]! + nx;
    const sy = normals[i * 3 + 1]! + ny;
    const sz = normals[i * 3 + 2]! + nz;
    assertFiniteFloat32(sx, primitiveId, ref, 'accumulated face normal.x');
    assertFiniteFloat32(sy, primitiveId, ref, 'accumulated face normal.y');
    assertFiniteFloat32(sz, primitiveId, ref, 'accumulated face normal.z');
    normals[i * 3] = sx;
    normals[i * 3 + 1] = sy;
    normals[i * 3 + 2] = sz;
  }
}

/**
 * Dice triangle-list geometry and apply CPU-readable displacement at generated
 * vertices before BVH construction. This is intentionally uniform and bounded:
 * it is the first real microgeometry contract, not an adaptive micropolygon
 * renderer. Returns null when `displacementSubdivisions` is absent/zero or
 * when the bounded dicing budget is exceeded; the latter emits a warning so
 * callers can deliberately fall back to authored-vertex displacement.
 * Malformed authored data still throws synchronously.
 */
export function maybeMicrodisplaceMeshGeometry(input: {
  readonly primitiveId: string;
  readonly material: MaterialSpec;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;
  readonly onWarning?: DisplacementWarningSink;
}): MicrodisplacedMeshGeometry | null {
  const ref = input.material.displacementMap;
  if (ref == null) return null;
  const subdivisions = resolveDisplacementSubdivisions(input.material, input.primitiveId);
  if (subdivisions === 0) return null;
  const {
    vertexCount,
    uvSource,
    uvSets: sourceUvSets,
    scale,
    bias,
  } = validateBaseGeometry(input, input.primitiveId, ref);
  const pixels = readHeightPixels(ref, input.primitiveId);
  const { triangleCount: sourceTriCount, indexAt } = validateTriangleSource(
    input.indices,
    vertexCount,
    input.primitiveId,
    ref,
  );
  if (input.tangents != null) {
    assertFloat32Stream(
      input.tangents,
      vertexCount * 4,
      input.primitiveId,
      ref,
      'tangents',
    );
  }
  let colorStride: 0 | 3 | 4 = 0;
  if (input.colors != null) {
    if (!isFloat32Array(input.colors)) {
      throw displacementError(input.primitiveId, ref, 'colors must be a Float32Array.');
    }
    if (input.colors.length === vertexCount * 3) colorStride = 3;
    else if (input.colors.length === vertexCount * 4) colorStride = 4;
    else {
      throw displacementError(
        input.primitiveId,
        ref,
        `colors.length must be exactly vertexCount*3 or vertexCount*4 (got ${input.colors.length}).`,
      );
    }
    assertFloat32Stream(
      input.colors,
      vertexCount * colorStride,
      input.primitiveId,
      ref,
      'colors',
    );
  }
  const steps = 1 << subdivisions;
  const generatedTriCountBig = BigInt(sourceTriCount) * BigInt(steps) * BigInt(steps);
  if (generatedTriCountBig > BigInt(MAX_MICRODISPLACED_TRIANGLES)) {
    input.onWarning?.(
      `${warningPrefix(input.primitiveId, ref)} displacementSubdivisions=${subdivisions} ` +
      `would generate ${generatedTriCountBig.toString()} triangles, above the shared-BVH ` +
      `safety cap ${MAX_MICRODISPLACED_TRIANGLES}; falling back to vertex displacement.`,
    );
    return null;
  }
  const generatedVertexCountBig = BigInt(sourceTriCount) *
    BigInt((steps + 1) * (steps + 2) / 2);
  if (
    generatedVertexCountBig > BigInt(Number.MAX_SAFE_INTEGER) ||
    generatedVertexCountBig > 0xffff_ffffn
  ) {
    throw displacementError(input.primitiveId, ref, 'generated vertex count is not Uint32-addressable.');
  }
  const generatedTriCount = Number(generatedTriCountBig);
  const generatedVertexCount = Number(generatedVertexCountBig);
  const sourceUvSetIndices = sparseArrayOwnIndices(sourceUvSets);
  let presentUvSetCount = 0;
  for (const setIndex of sourceUvSetIndices) {
    if (sourceUvSets[setIndex] != null) presentUvSetCount += 1;
  }
  const bytesPerVertex = 12n + 12n + 24n + 12n +
    BigInt(presentUvSetCount * 8) +
    (input.tangents != null ? 16n : 0n) +
    BigInt(colorStride * 4);
  const allocationBytes = BigInt(generatedVertexCount) * bytesPerVertex +
    BigInt(generatedTriCount) * 12n;
  if (allocationBytes > BigInt(MAX_DISPLACEMENT_CPU_BYTES)) {
    input.onWarning?.(
      `${warningPrefix(input.primitiveId, ref)} microdisplacement outputs and normal staging ` +
      `require ${allocationBytes.toString()} CPU bytes, above the 512 MiB staging budget; ` +
      'falling back to vertex displacement.',
    );
    return null;
  }
  const outPositions = new Float32Array(generatedVertexCount * 3);
  const outNormalFallback = new Float32Array(generatedVertexCount * 3);
  const outUvSets: Array<Float32Array | undefined> = [];
  for (const setIndex of sourceUvSetIndices) {
    if (sourceUvSets[setIndex] != null) {
      outUvSets[setIndex] = new Float32Array(generatedVertexCount * 2);
    }
  }
  const outTangents = input.tangents != null
    ? new Float32Array(generatedVertexCount * 4)
    : undefined;
  const outColors = colorStride > 0
    ? new Float32Array(generatedVertexCount * colorStride)
    : undefined;
  const outIndices = new Uint32Array(generatedTriCount * 3);
  let vertexCursor = 0;
  let indexCursor = 0;

  const pushVertex = (
    ia: number,
    ib: number,
    ic: number,
    wa: number,
    wb: number,
    wc: number,
  ): number => {
    const px = lerpScalar(input.positions, 3, ia, ib, ic, 0, wa, wb, wc);
    const py = lerpScalar(input.positions, 3, ia, ib, ic, 1, wa, wb, wc);
    const pz = lerpScalar(input.positions, 3, ia, ib, ic, 2, wa, wb, wc);
    const [nx, ny, nz] = normalizeStrict(
      lerpScalar(input.normals, 3, ia, ib, ic, 0, wa, wb, wc),
      lerpScalar(input.normals, 3, ia, ib, ic, 1, wa, wb, wc),
      lerpScalar(input.normals, 3, ia, ib, ic, 2, wa, wb, wc),
      input.primitiveId,
      ref,
      `generated normal ${vertexCursor}`,
    );
    const du = lerpScalar(uvSource, 2, ia, ib, ic, 0, wa, wb, wc);
    const dv = lerpScalar(uvSource, 2, ia, ib, ic, 1, wa, wb, wc);
    const amount = sampleHeight(pixels, du, dv, ref, input.primitiveId) * scale + bias;
    assertFiniteFloat32(amount, input.primitiveId, ref, `generated displacement ${vertexCursor}`);
    const ox = px + nx * amount;
    const oy = py + ny * amount;
    const oz = pz + nz * amount;
    assertFiniteFloat32(ox, input.primitiveId, ref, `generated positions[${vertexCursor * 3}]`);
    assertFiniteFloat32(oy, input.primitiveId, ref, `generated positions[${vertexCursor * 3 + 1}]`);
    assertFiniteFloat32(oz, input.primitiveId, ref, `generated positions[${vertexCursor * 3 + 2}]`);
    const outIndex = vertexCursor;
    outPositions[vertexCursor * 3] = ox;
    outPositions[vertexCursor * 3 + 1] = oy;
    outPositions[vertexCursor * 3 + 2] = oz;
    outNormalFallback[vertexCursor * 3] = nx;
    outNormalFallback[vertexCursor * 3 + 1] = ny;
    outNormalFallback[vertexCursor * 3 + 2] = nz;
    for (const setIndex of sourceUvSetIndices) {
      const source = sourceUvSets[setIndex];
      const output = outUvSets[setIndex];
      if (source == null || output == null) continue;
      const u = lerpScalar(source, 2, ia, ib, ic, 0, wa, wb, wc);
      const v = lerpScalar(source, 2, ia, ib, ic, 1, wa, wb, wc);
      assertFiniteFloat32(u, input.primitiveId, ref, `generated uvSets[${setIndex}].u`);
      assertFiniteFloat32(v, input.primitiveId, ref, `generated uvSets[${setIndex}].v`);
      output[vertexCursor * 2] = u;
      output[vertexCursor * 2 + 1] = v;
    }
    if (outTangents != null) {
      const [tx, ty, tz] = normalizeStrict(
        lerpScalar(input.tangents!, 4, ia, ib, ic, 0, wa, wb, wc),
        lerpScalar(input.tangents!, 4, ia, ib, ic, 1, wa, wb, wc),
        lerpScalar(input.tangents!, 4, ia, ib, ic, 2, wa, wb, wc),
        input.primitiveId,
        ref,
        `generated tangent ${vertexCursor}`,
      );
      const handednessSource = lerpScalar(input.tangents!, 4, ia, ib, ic, 3, wa, wb, wc);
      assertFiniteFloat32(handednessSource, input.primitiveId, ref, 'generated tangent handedness');
      outTangents[vertexCursor * 4] = tx;
      outTangents[vertexCursor * 4 + 1] = ty;
      outTangents[vertexCursor * 4 + 2] = tz;
      outTangents[vertexCursor * 4 + 3] = handednessSource < 0 ? -1 : 1;
    }
    if (outColors != null) {
      for (let component = 0; component < colorStride; component += 1) {
        const value = lerpScalar(
          input.colors!,
          colorStride,
          ia,
          ib,
          ic,
          component,
          wa,
          wb,
          wc,
        );
        assertFiniteFloat32(value, input.primitiveId, ref, `generated color.${component}`);
        outColors[vertexCursor * colorStride + component] = value;
      }
    }
    vertexCursor += 1;
    return outIndex;
  };

  for (let tri = 0; tri < sourceTriCount; tri += 1) {
    const ia = indexAt(tri * 3);
    const ib = indexAt(tri * 3 + 1);
    const ic = indexAt(tri * 3 + 2);
    const rows: number[][] = [];
    for (let row = 0; row <= steps; row += 1) {
      const cells = steps - row;
      rows[row] = [];
      for (let col = 0; col <= cells; col += 1) {
        const wb = row / steps;
        const wc = col / steps;
        const wa = 1 - wb - wc;
        rows[row]![col] = pushVertex(ia, ib, ic, wa, wb, wc);
      }
    }
    for (let row = 0; row < steps; row += 1) {
      for (let col = 0; col < steps - row; col += 1) {
        const v0 = rows[row]![col]!;
        const v1 = rows[row + 1]![col]!;
        const v2 = rows[row]![col + 1]!;
        outIndices[indexCursor++] = v0;
        outIndices[indexCursor++] = v1;
        outIndices[indexCursor++] = v2;
        if (col < steps - row - 1) {
          const v3 = rows[row + 1]![col + 1]!;
          outIndices[indexCursor++] = v1;
          outIndices[indexCursor++] = v3;
          outIndices[indexCursor++] = v2;
        }
      }
    }
  }

  if (vertexCursor !== generatedVertexCount || indexCursor !== outIndices.length) {
    throw displacementError(
      input.primitiveId,
      ref,
      `internal tessellation count mismatch (${vertexCursor}/${generatedVertexCount} vertices, ${indexCursor}/${outIndices.length} indices).`,
    );
  }
  const normalAccum = new Float64Array(generatedVertexCount * 3);
  for (let offset = 0; offset < outIndices.length; offset += 3) {
    accumulateFaceNormal(
      outPositions,
      normalAccum,
      outIndices[offset]!,
      outIndices[offset + 1]!,
      outIndices[offset + 2]!,
      input.primitiveId,
      ref,
    );
  }
  const outNormals = new Float32Array(generatedVertexCount * 3);
  for (let i = 0; i < outNormals.length; i += 3) {
    const ax = normalAccum[i]!;
    const ay = normalAccum[i + 1]!;
    const az = normalAccum[i + 2]!;
    const [nx, ny, nz] = ax === 0 && ay === 0 && az === 0
      ? [outNormalFallback[i]!, outNormalFallback[i + 1]!, outNormalFallback[i + 2]!] as const
      : normalizeStrict(ax, ay, az, input.primitiveId, ref, `output normal ${i / 3}`);
    outNormals[i] = nx;
    outNormals[i + 1] = ny;
    outNormals[i + 2] = nz;
  }

  const generatedUvSets: PrimitiveUvSets = outUvSets;
  return {
    positions: outPositions,
    normals: outNormals,
    indices: outIndices,
    ...(generatedUvSets[0] != null ? { uvs: generatedUvSets[0] } : {}),
    ...(generatedUvSets[1] != null ? { uv1: generatedUvSets[1] } : {}),
    ...(sparseArrayHasDefinedEntry(generatedUvSets)
      ? { uvSets: generatedUvSets }
      : {}),
    ...(outTangents != null ? { tangents: outTangents } : {}),
    ...(outColors != null ? { colors: outColors } : {}),
    subdivisions,
  };
}

/** Mesh-like primitive attributes consumed by {@link resolveDisplacedGeometry}. */
export interface DisplaceablePrimitive {
  readonly id: string;
  readonly material: MaterialSpec;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;
  readonly colorSets?: PrimitiveColorSets;
  readonly vertexColorSet?: number | null;
}

/** Resolved base attributes after (optional) microdisplacement + vertex displacement. */
export interface ResolvedDisplacedGeometry {
  /** Whether microdisplacement diced the source geometry. */
  readonly microdisplaced: boolean;
  readonly basePositions: Float32Array;
  readonly baseNormals: Float32Array;
  readonly baseTangents: Float32Array | undefined;
  readonly baseColors: Float32Array | undefined;
  readonly baseUvs: Float32Array | undefined;
  readonly baseUv1: Float32Array | undefined;
  readonly baseUvSets: PrimitiveUvSets | undefined;
  readonly baseIndicesSource: Uint32Array | Uint16Array | undefined;
  /**
   * `basePositions` after CPU vertex displacement — equal to `basePositions`
   * when the primitive was microdisplaced (which already bakes displacement) or
   * when there is no displacement map to apply.
   */
  readonly sourcePositions: Float32Array;
}

/**
 * Fold the microdisplacement + base-attribute resolution + vertex-displacement
 * preamble that was triplicated across `scenePack.packOneMeshLikePrimitive`,
 * `worldSpaceMerge.mergeWorldSpaceFromCore`, and (partially) `mergeUv1FromCore`
 * (D12-6). Behavior is bit-for-bit identical to the former inline blocks:
 *   1. `maybeMicrodisplaceMeshGeometry` (diced geometry bakes displacement);
 *   2. `base*` = microdisplaced attribute ?? authored attribute;
 *   3. `sourcePositions` = when NOT microdisplaced, `maybeDisplaceMeshPositions`
 *      over the base positions/normals (?? basePositions); else basePositions.
 */
export function resolveDisplacedGeometry(
  primitive: DisplaceablePrimitive,
  warn: DisplacementWarningSink,
): ResolvedDisplacedGeometry {
  const selectedColors = getPrimitiveActiveColorSet(primitive);
  const microdisplaced = maybeMicrodisplaceMeshGeometry({
    primitiveId: primitive.id,
    material: primitive.material,
    positions: primitive.positions,
    normals: primitive.normals,
    ...(primitive.indices != null ? { indices: primitive.indices } : {}),
    ...(primitive.uvs != null ? { uvs: primitive.uvs } : {}),
    ...(primitive.uv1 != null ? { uv1: primitive.uv1 } : {}),
    ...(primitive.uvSets != null ? { uvSets: primitive.uvSets } : {}),
    ...(primitive.tangents != null ? { tangents: primitive.tangents } : {}),
    ...(selectedColors != null ? { colors: selectedColors } : {}),
    onWarning: warn,
  });
  const basePositions = microdisplaced?.positions ?? primitive.positions;
  const baseNormals = microdisplaced?.normals ?? primitive.normals;
  const baseTangents = microdisplaced?.tangents ?? primitive.tangents;
  const baseColors = microdisplaced?.colors ?? selectedColors;
  const resolvedUvStreams = microdisplaced ?? primitive;
  const resolvedUvSets: Array<Float32Array | undefined> = [];
  if (resolvedUvStreams.uvSets != null) {
    for (const setIndex of sparseArrayOwnIndices(resolvedUvStreams.uvSets)) {
      resolvedUvSets[setIndex] = getPrimitiveUvSet(resolvedUvStreams, setIndex);
    }
  }
  if (resolvedUvStreams.uvs != null) resolvedUvSets[0] = resolvedUvStreams.uvs;
  if (resolvedUvStreams.uv1 != null) resolvedUvSets[1] = resolvedUvStreams.uv1;
  const baseUvSets: PrimitiveUvSets | undefined =
    sparseArrayHasDefinedEntry(resolvedUvSets) ? resolvedUvSets : undefined;
  const baseUvs = baseUvSets?.[0];
  const baseUv1 = baseUvSets?.[1];
  const baseIndicesSource = microdisplaced?.indices ?? primitive.indices;
  const sourcePositions = microdisplaced == null
    ? maybeDisplaceMeshPositions({
        primitiveId: primitive.id,
        material: primitive.material,
        positions: basePositions,
        normals: baseNormals,
        ...(baseUvs != null ? { uvs: baseUvs } : {}),
        ...(baseUv1 != null ? { uv1: baseUv1 } : {}),
        ...(baseUvSets != null ? { uvSets: baseUvSets } : {}),
      }) ?? basePositions
    : basePositions;
  return {
    microdisplaced: microdisplaced != null,
    basePositions,
    baseNormals,
    baseTangents,
    baseColors,
    baseUvs,
    baseUv1,
    baseUvSets,
    baseIndicesSource,
    sourcePositions,
  };
}
