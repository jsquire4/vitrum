export type IntrinsicTypedArrayBrand =
  | 'Int8Array'
  | 'Uint8Array'
  | 'Uint8ClampedArray'
  | 'Int16Array'
  | 'Uint16Array'
  | 'Int32Array'
  | 'Uint32Array'
  | 'Float32Array'
  | 'Float64Array'
  | 'BigInt64Array'
  | 'BigUint64Array';

export interface IntrinsicTypedArrayInfo {
  readonly brand: IntrinsicTypedArrayBrand;
  readonly length: number;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly buffer: ArrayBufferLike;
  readonly isShared: boolean;
}

const typedArrayPrototype = Object.getPrototypeOf(Uint8Array.prototype) as object;
const typedArrayLengthGetter = intrinsicGetter(typedArrayPrototype, 'length');
const typedArrayByteLengthGetter = intrinsicGetter(typedArrayPrototype, 'byteLength');
const typedArrayByteOffsetGetter = intrinsicGetter(typedArrayPrototype, 'byteOffset');
const typedArrayBufferGetter = intrinsicGetter(typedArrayPrototype, 'buffer');
const typedArrayBrandGetter = intrinsicGetter(typedArrayPrototype, Symbol.toStringTag);
const arrayBufferByteLengthGetter = intrinsicGetter(ArrayBuffer.prototype, 'byteLength');
const sharedArrayBufferConstructor = (
  globalThis as typeof globalThis & {
    SharedArrayBuffer?: {
      readonly prototype: object;
    };
  }
).SharedArrayBuffer;
const sharedArrayBufferByteLengthGetter =
  sharedArrayBufferConstructor === undefined
    ? undefined
    : intrinsicGetter(
        sharedArrayBufferConstructor.prototype,
        'byteLength',
      );

function intrinsicGetter(
  prototype: object,
  key: PropertyKey,
): ((receiver: unknown) => unknown) | undefined {
  const descriptor = Object.getOwnPropertyDescriptor(prototype, key);
  if (descriptor === undefined) return undefined;
  const getter: unknown = Reflect.get(descriptor, 'get');
  if (typeof getter !== 'function') return undefined;
  return (receiver: unknown): unknown => Reflect.apply(getter, receiver, []);
}

const TYPED_ARRAY_BRANDS = new Set<string>([
  'Int8Array',
  'Uint8Array',
  'Uint8ClampedArray',
  'Int16Array',
  'Uint16Array',
  'Int32Array',
  'Uint32Array',
  'Float32Array',
  'Float64Array',
  'BigInt64Array',
  'BigUint64Array',
]);

/**
 * Inspect a typed array through intrinsic slot getters. Own `length`,
 * `byteLength`, `buffer`, and `Symbol.toStringTag` shadows are ignored, and
 * cross-realm typed arrays are accepted.
 *
 * Returns `null` for non-typed-arrays, DataView, proxies, or malformed values.
 *
 * @internal
 */
export function inspectIntrinsicTypedArray(
  value: unknown,
): IntrinsicTypedArrayInfo | null {
  if (
    value === null ||
    (typeof value !== 'object' && typeof value !== 'function') ||
    typedArrayLengthGetter === undefined ||
    typedArrayByteLengthGetter === undefined ||
    typedArrayByteOffsetGetter === undefined ||
    typedArrayBufferGetter === undefined ||
    typedArrayBrandGetter === undefined
  ) {
    return null;
  }

  try {
    const brand = typedArrayBrandGetter(value);
    if (typeof brand !== 'string' || !TYPED_ARRAY_BRANDS.has(brand)) return null;
    const length = typedArrayLengthGetter(value);
    const byteLength = typedArrayByteLengthGetter(value);
    const byteOffset = typedArrayByteOffsetGetter(value);
    const buffer = typedArrayBufferGetter(value);
    if (
      !Number.isSafeInteger(length) ||
      (length as number) < 0 ||
      !Number.isSafeInteger(byteLength) ||
      (byteLength as number) < 0 ||
      !Number.isSafeInteger(byteOffset) ||
      (byteOffset as number) < 0
    ) {
      return null;
    }
    const isShared = isIntrinsicSharedArrayBuffer(buffer);
    if (!isShared && !isIntrinsicArrayBuffer(buffer)) return null;
    return {
      brand: brand as IntrinsicTypedArrayBrand,
      length: length as number,
      byteLength: byteLength as number,
      byteOffset: byteOffset as number,
      buffer: buffer as ArrayBufferLike,
      isShared,
    };
  } catch {
    return null;
  }
}

/** @internal */
export function isIntrinsicArrayBuffer(value: unknown): value is ArrayBuffer {
  if (arrayBufferByteLengthGetter === undefined) return false;
  try {
    arrayBufferByteLengthGetter(value);
    return true;
  } catch {
    return false;
  }
}

/** @internal */
export function isIntrinsicSharedArrayBuffer(value: unknown): boolean {
  if (sharedArrayBufferByteLengthGetter === undefined) return false;
  try {
    sharedArrayBufferByteLengthGetter(value);
    return true;
  } catch {
    return false;
  }
}

/**
 * Return a local-realm Uint8Array view using intrinsic offsets/length. Shared
 * backing stores and non-Uint8Array values are rejected.
 *
 * @internal
 */
export function localUint8ArrayView(value: unknown): Uint8Array | null {
  const info = inspectIntrinsicTypedArray(value);
  if (info === null || info.brand !== 'Uint8Array' || info.isShared) return null;
  try {
    return new Uint8Array(info.buffer as ArrayBuffer, info.byteOffset, info.length);
  } catch {
    return null;
  }
}
