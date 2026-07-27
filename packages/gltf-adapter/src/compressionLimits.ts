/** Default ceiling for one compressed input or built-in decoded working set. */
export const COMPRESSION_DECODE_BUDGET_BYTES = 512 * 1024 * 1024;

export function checkedCompressionSum(parts: readonly number[], path: string): number {
  let total = 0;
  for (const part of parts) {
    if (!Number.isSafeInteger(part) || part < 0) {
      throw new RangeError(`${path} contains an invalid byte length ${String(part)}`);
    }
    total += part;
    if (!Number.isSafeInteger(total)) {
      throw new RangeError(`${path} exceeds the safe integer range`);
    }
  }
  return total;
}

export function checkedCompressionProduct(left: number, right: number, path: string): number {
  if (!Number.isSafeInteger(left) || left < 0) {
    throw new RangeError(`${path} contains an invalid factor ${String(left)}`);
  }
  if (!Number.isSafeInteger(right) || right < 0) {
    throw new RangeError(`${path} contains an invalid factor ${String(right)}`);
  }
  if (left !== 0 && right > Math.floor(Number.MAX_SAFE_INTEGER / left)) {
    throw new RangeError(`${path} exceeds the safe integer range`);
  }
  return left * right;
}

/**
 * Monotonic per-resolution allocation ledger. Costs remain charged because
 * hook results, validation snapshots, and retained synthetic buffers are not
 * guaranteed to be collected between sequential bufferView/primitive decodes.
 */
export class CompressionAllocationLedger {
  private chargedByteLength = 0;

  constructor(private readonly budgetByteLength = COMPRESSION_DECODE_BUDGET_BYTES) {
    if (!Number.isSafeInteger(budgetByteLength) || budgetByteLength < 0) {
      throw new RangeError(
        `compression budget must be a non-negative safe integer; received ${String(budgetByteLength)}`,
      );
    }
  }

  ensureAvailable(additionalByteLength: number, path: string): void {
    const next = checkedCompressionSum([this.chargedByteLength, additionalByteLength], path);
    if (this.budgetByteLength !== 0 && next > this.budgetByteLength) {
      throw new RangeError(
        `${path} cumulative allocation ${next} exceeds the compression budget of ` +
          `${this.budgetByteLength} bytes`,
      );
    }
  }

  charge(additionalByteLength: number, path: string): void {
    this.ensureAvailable(additionalByteLength, path);
    this.chargedByteLength += additionalByteLength;
  }

  get chargedBytes(): number {
    return this.chargedByteLength;
  }
}

export interface CompressionDecodeState {
  attemptsDisabled: boolean;
}

/**
 * Charge a decoder-owned result. When the result itself does not fit, stop
 * invoking codecs for the rest of this resolution but preserve the remaining
 * ledger capacity for exact fallback validation/materialization.
 */
export function chargeCompressionHookOutput(
  ledger: CompressionAllocationLedger,
  state: CompressionDecodeState,
  byteLength: number,
  path: string,
): void {
  try {
    ledger.charge(byteLength, path);
  } catch (error) {
    state.attemptsDisabled = true;
    throw error;
  }
}

export type CompressionTypedArrayKind =
  | 'Int8Array'
  | 'Uint8Array'
  | 'Uint8ClampedArray'
  | 'Int16Array'
  | 'Uint16Array'
  | 'Int32Array'
  | 'Uint32Array'
  | 'Float16Array'
  | 'Float32Array'
  | 'Float64Array'
  | 'BigInt64Array'
  | 'BigUint64Array';

export interface CompressionTypedArrayInfo {
  readonly kind: CompressionTypedArrayKind;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly length: number;
  readonly shared: boolean;
  readonly buffer: ArrayBufferLike;
}

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint8Array.prototype) as object;
type IntrinsicGetter = (receiver: unknown) => unknown;

function intrinsicGetter(prototype: object, property: PropertyKey): IntrinsicGetter | undefined {
  // eslint-disable-next-line @typescript-eslint/unbound-method -- The accessor is deliberately invoked later with an explicit receiver.
  const getter = Object.getOwnPropertyDescriptor(prototype, property)?.get;
  return getter === undefined
    ? undefined
    : (receiver: unknown): unknown => Reflect.apply(getter, receiver, []);
}

const TYPED_ARRAY_BYTE_LENGTH_GETTER = intrinsicGetter(TYPED_ARRAY_PROTOTYPE, 'byteLength');
const TYPED_ARRAY_LENGTH_GETTER = intrinsicGetter(TYPED_ARRAY_PROTOTYPE, 'length');
const TYPED_ARRAY_BYTE_OFFSET_GETTER = intrinsicGetter(TYPED_ARRAY_PROTOTYPE, 'byteOffset');
const TYPED_ARRAY_BUFFER_GETTER = intrinsicGetter(TYPED_ARRAY_PROTOTYPE, 'buffer');
const TYPED_ARRAY_TAG_GETTER = intrinsicGetter(TYPED_ARRAY_PROTOTYPE, Symbol.toStringTag);
const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER =
  typeof SharedArrayBuffer === 'undefined'
    ? undefined
    : intrinsicGetter(SharedArrayBuffer.prototype, 'byteLength');

const SUPPORTED_TYPED_ARRAY_KINDS: ReadonlySet<string> = new Set([
  'Int8Array',
  'Uint8Array',
  'Uint8ClampedArray',
  'Int16Array',
  'Uint16Array',
  'Int32Array',
  'Uint32Array',
  'Float16Array',
  'Float32Array',
  'Float64Array',
  'BigInt64Array',
  'BigUint64Array',
]);

/**
 * Brand and size a supported typed array through unshadowable intrinsics.
 * Works across realms and ignores instance-owned byteLength/toStringTag fields.
 */
export function compressionTypedArrayInfo(value: unknown): CompressionTypedArrayInfo | undefined {
  if (
    !ArrayBuffer.isView(value) ||
    TYPED_ARRAY_BYTE_LENGTH_GETTER === undefined ||
    TYPED_ARRAY_LENGTH_GETTER === undefined ||
    TYPED_ARRAY_BYTE_OFFSET_GETTER === undefined ||
    TYPED_ARRAY_BUFFER_GETTER === undefined ||
    TYPED_ARRAY_TAG_GETTER === undefined
  ) {
    return undefined;
  }
  try {
    const kind = TYPED_ARRAY_TAG_GETTER(value);
    if (typeof kind !== 'string' || !SUPPORTED_TYPED_ARRAY_KINDS.has(kind)) {
      return undefined;
    }
    const byteLength = TYPED_ARRAY_BYTE_LENGTH_GETTER(value);
    const length = TYPED_ARRAY_LENGTH_GETTER(value);
    const byteOffset = TYPED_ARRAY_BYTE_OFFSET_GETTER(value);
    const buffer = TYPED_ARRAY_BUFFER_GETTER(value);
    let shared = false;
    if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER !== undefined) {
      try {
        SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER(buffer);
        shared = true;
      } catch {
        // The intrinsic rejected an ordinary ArrayBuffer.
      }
    }
    if (
      !Number.isSafeInteger(byteLength) ||
      (byteLength as number) < 0 ||
      !Number.isSafeInteger(length) ||
      (length as number) < 0 ||
      !Number.isSafeInteger(byteOffset) ||
      (byteOffset as number) < 0
    ) {
      return undefined;
    }
    return {
      kind: kind as CompressionTypedArrayKind,
      byteLength: byteLength as number,
      byteOffset: byteOffset as number,
      length: length as number,
      shared,
      buffer: buffer as ArrayBufferLike,
    };
  } catch {
    return undefined;
  }
}

/**
 * Reject hostile declarations before a resolver makes a defensive byte copy.
 * A zero budget keeps safe-integer validation but disables this byte ceiling,
 * matching the public import-resource policy.
 */
export function validateCompressionInputBudget(
  byteLength: number,
  path: string,
  budgetByteLength = COMPRESSION_DECODE_BUDGET_BYTES,
): void {
  if (!Number.isSafeInteger(budgetByteLength) || budgetByteLength < 0) {
    throw new RangeError(
      `${path} compression input budget must be a non-negative safe integer; ` +
        `received ${String(budgetByteLength)}`,
    );
  }
  if (
    !Number.isSafeInteger(byteLength) ||
    byteLength < 0 ||
    (budgetByteLength !== 0 && byteLength > budgetByteLength)
  ) {
    throw new RangeError(
      `${path} exceeds the compression input budget of ` + `${budgetByteLength} bytes`,
    );
  }
}
