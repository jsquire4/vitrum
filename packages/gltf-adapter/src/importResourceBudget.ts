import { GltfAdapterError } from './errors.js';

const MIB = 1024 * 1024;

// Capture the intrinsic once so host-owned objects cannot under-report an
// encoded allocation through an own `byteLength` property. Calling this getter
// also provides a cross-realm ArrayBuffer brand check and rejects
// SharedArrayBuffer.
// eslint-disable-next-line @typescript-eslint/unbound-method -- Invoked only with an explicit candidate receiver.
const ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength',
)?.get;

export function gltfArrayBufferByteLength(value: unknown): number | undefined {
  if (ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return undefined;
  try {
    const byteLength = Reflect.apply(
      ARRAY_BUFFER_BYTE_LENGTH_GETTER,
      value,
      [],
    ) as unknown;
    return Number.isSafeInteger(byteLength) && (byteLength as number) >= 0
      ? byteLength as number
      : undefined;
  } catch {
    return undefined;
  }
}

/** Stable dedup keys shared by low-level conversion and URL/data loaders. */
export const GLTF_INPUT_RESOURCE_KEY = 'asset:input';

export function gltfBufferResourceKey(bufferIndex: number): string {
  return `buffer:${bufferIndex}`;
}

export function gltfImageResourceKey(imageIndex: number): string {
  return `image:${imageIndex}`;
}

export interface GltfImportResourceLimits {
  /** Aggregate bytes allocated by decoded and adapter-generated geometry arrays. */
  readonly maxDecodedGeometryBytes?: number;
  /** Maximum encoded byte length of any one asset, buffer, or image resource. */
  readonly maxEncodedResourceBytes?: number;
  /** Aggregate encoded bytes across distinct resources during one import. */
  readonly maxTotalEncodedBytes?: number;
  /**
   * Maximum source or adapter-derived image dimensions, measured in pixels.
   * Host decoder internals are opaque; this limit is enforced before accepting
   * decoder output and before each adapter-owned resize/bake allocation.
   */
  readonly maxDecodedTexturePixels?: number;
  /**
   * Aggregate pixels across accepted dimensioned decoder surfaces and
   * adapter-derived RGBA outputs created by normalization, resizing, or baking
   * during one import. Opaque temporary memory retained inside a host decoder
   * is outside the adapter's accounting.
   */
  readonly maxTotalDecodedTexturePixels?: number;
  /** Maximum simultaneously active fetch/decode resource operations. */
  readonly maxConcurrentResourceOperations?: number;
}

export interface NormalizedGltfImportResourceLimits {
  readonly maxDecodedGeometryBytes: number;
  readonly maxEncodedResourceBytes: number;
  readonly maxTotalEncodedBytes: number;
  readonly maxDecodedTexturePixels: number;
  readonly maxTotalDecodedTexturePixels: number;
  readonly maxConcurrentResourceOperations: number;
}

export const DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS: Readonly<NormalizedGltfImportResourceLimits> =
  Object.freeze({
    maxDecodedGeometryBytes: 512 * MIB,
    maxEncodedResourceBytes: 256 * MIB,
    maxTotalEncodedBytes: 512 * MIB,
    maxDecodedTexturePixels: 16_777_216,
    maxTotalDecodedTexturePixels: 16_777_216,
    maxConcurrentResourceOperations: 4,
  });

function positiveSafeInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} must be a positive safe integer; received ${String(value)}.`,
    );
  }
  return value as number;
}

function nonNegativeSafeInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} must be a non-negative safe integer; received ${String(value)}.`,
    );
  }
  return value as number;
}

export function normalizeGltfImportResourceLimits(
  limits: GltfImportResourceLimits | undefined,
): Readonly<NormalizedGltfImportResourceLimits> {
  if (limits === undefined) return DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS;
  if (limits === null || typeof limits !== 'object' || Array.isArray(limits)) {
    throw new TypeError('[vitrum/gltf-adapter] resourceLimits must be an object when supplied.');
  }
  return Object.freeze({
    maxDecodedGeometryBytes:
      limits.maxDecodedGeometryBytes === undefined
        ? DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxDecodedGeometryBytes
        : nonNegativeSafeInteger(
            limits.maxDecodedGeometryBytes,
            'resourceLimits.maxDecodedGeometryBytes',
          ),
    maxEncodedResourceBytes:
      limits.maxEncodedResourceBytes === undefined
        ? DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxEncodedResourceBytes
        : nonNegativeSafeInteger(
            limits.maxEncodedResourceBytes,
            'resourceLimits.maxEncodedResourceBytes',
          ),
    maxTotalEncodedBytes:
      limits.maxTotalEncodedBytes === undefined
        ? DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxTotalEncodedBytes
        : nonNegativeSafeInteger(
            limits.maxTotalEncodedBytes,
            'resourceLimits.maxTotalEncodedBytes',
          ),
    maxDecodedTexturePixels:
      limits.maxDecodedTexturePixels === undefined
        ? DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxDecodedTexturePixels
        : nonNegativeSafeInteger(
            limits.maxDecodedTexturePixels,
            'resourceLimits.maxDecodedTexturePixels',
          ),
    maxTotalDecodedTexturePixels:
      limits.maxTotalDecodedTexturePixels === undefined
        ? DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxTotalDecodedTexturePixels
        : nonNegativeSafeInteger(
            limits.maxTotalDecodedTexturePixels,
            'resourceLimits.maxTotalDecodedTexturePixels',
          ),
    maxConcurrentResourceOperations:
      limits.maxConcurrentResourceOperations === undefined
        ? DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxConcurrentResourceOperations
        : positiveSafeInteger(
            limits.maxConcurrentResourceOperations,
            'resourceLimits.maxConcurrentResourceOperations',
          ),
  });
}

export type GltfResourceLimitKind =
  | 'decoded-geometry-bytes'
  | 'encoded-resource-bytes'
  | 'total-encoded-bytes'
  | 'decoded-texture-pixels'
  | 'total-decoded-texture-pixels';

export interface GltfResourceLimitErrorInit {
  readonly limitKind: GltfResourceLimitKind;
  readonly limit: number;
  readonly actual: number;
  readonly path: string;
  readonly resourceKey?: string;
}

export class GltfResourceLimitError extends GltfAdapterError {
  readonly limitKind: GltfResourceLimitKind;
  readonly limit: number;
  readonly actual: number;
  readonly path: string;
  readonly resourceKey?: string;

  constructor(init: GltfResourceLimitErrorInit) {
    const resource =
      init.resourceKey === undefined ? '' : ` for resource "${init.resourceKey}"`;
    super(
      'GLTF_RESOURCE_LIMIT_EXCEEDED',
      `[vitrum/gltf-adapter] ${init.limitKind}${resource} at ${init.path} ` +
        `would reach ${init.actual}, exceeding limit ${init.limit}.`,
    );
    this.limitKind = init.limitKind;
    this.limit = init.limit;
    this.actual = init.actual;
    this.path = init.path;
    if (init.resourceKey !== undefined) this.resourceKey = init.resourceKey;
  }
}

function checkedTotal(current: number, additional: number, path: string): number {
  nonNegativeSafeInteger(current, `${path} current total`);
  nonNegativeSafeInteger(additional, `${path} additional amount`);
  if (additional > Number.MAX_SAFE_INTEGER - current) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path} exceeds the safe integer range.`);
  }
  return current + additional;
}

/**
 * Monotonic accounting for one complete import. No charge is released because
 * decoded arrays and fetched resources can remain reachable until import
 * completion, and garbage collection is not an allocation boundary.
 */
export class ImportResourceLedger {
  private currentLimits: Readonly<NormalizedGltfImportResourceLimits>;

  private decodedGeometryByteTotal = 0;
  private decodedGeometryLastPath: string | undefined;
  private encodedByteTotal = 0;
  private encodedTotalLastPath: string | undefined;
  private encodedTotalLastResourceKey: string | undefined;
  private decodedTexturePixelTotal = 0;
  private decodedTextureLastPath: string | undefined;
  private largestDecodedTexturePixelCharge = 0;
  private largestDecodedTexturePixelPath: string | undefined;
  private readonly encodedBytesByKey = new Map<string, number>();
  private readonly encodedPathsByKey = new Map<string, string>();

  constructor(limits?: GltfImportResourceLimits | NormalizedGltfImportResourceLimits) {
    this.currentLimits = normalizeGltfImportResourceLimits(limits);
  }

  get limits(): Readonly<NormalizedGltfImportResourceLimits> {
    return this.currentLimits;
  }

  /**
   * Apply a later import-stage policy without discarding any charges already
   * accumulated by this import. Subsequent observations are checked against
   * the new limits while encoded-resource deduplication remains intact.
   */
  reconfigureLimits(
    limits: GltfImportResourceLimits | NormalizedGltfImportResourceLimits,
  ): void {
    const candidate = normalizeGltfImportResourceLimits(limits);
    if (
      candidate.maxDecodedGeometryBytes !== 0 &&
      this.decodedGeometryByteTotal > candidate.maxDecodedGeometryBytes
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'decoded-geometry-bytes',
        limit: candidate.maxDecodedGeometryBytes,
        actual: this.decodedGeometryByteTotal,
        path: this.decodedGeometryLastPath ?? 'resourceLimits.maxDecodedGeometryBytes',
      });
    }
    if (candidate.maxEncodedResourceBytes !== 0) {
      for (const [resourceKey, byteLength] of this.encodedBytesByKey) {
        if (byteLength <= candidate.maxEncodedResourceBytes) continue;
        throw new GltfResourceLimitError({
          limitKind: 'encoded-resource-bytes',
          limit: candidate.maxEncodedResourceBytes,
          actual: byteLength,
          path:
            this.encodedPathsByKey.get(resourceKey) ??
            'resourceLimits.maxEncodedResourceBytes',
          resourceKey,
        });
      }
    }
    if (
      candidate.maxTotalEncodedBytes !== 0 &&
      this.encodedByteTotal > candidate.maxTotalEncodedBytes
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'total-encoded-bytes',
        limit: candidate.maxTotalEncodedBytes,
        actual: this.encodedByteTotal,
        path: this.encodedTotalLastPath ?? 'resourceLimits.maxTotalEncodedBytes',
        ...(this.encodedTotalLastResourceKey !== undefined
          ? { resourceKey: this.encodedTotalLastResourceKey }
          : {}),
      });
    }
    if (
      candidate.maxDecodedTexturePixels !== 0 &&
      this.largestDecodedTexturePixelCharge > candidate.maxDecodedTexturePixels
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'decoded-texture-pixels',
        limit: candidate.maxDecodedTexturePixels,
        actual: this.largestDecodedTexturePixelCharge,
        path:
          this.largestDecodedTexturePixelPath ??
          'resourceLimits.maxDecodedTexturePixels',
      });
    }
    if (
      candidate.maxTotalDecodedTexturePixels !== 0 &&
      this.decodedTexturePixelTotal > candidate.maxTotalDecodedTexturePixels
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'total-decoded-texture-pixels',
        limit: candidate.maxTotalDecodedTexturePixels,
        actual: this.decodedTexturePixelTotal,
        path:
          this.decodedTextureLastPath ??
          'resourceLimits.maxTotalDecodedTexturePixels',
      });
    }
    this.currentLimits = candidate;
  }

  ensureDecodedGeometryBytes(byteLength: number, path: string): void {
    const next = checkedTotal(
      this.decodedGeometryByteTotal,
      byteLength,
      `${path} decoded geometry bytes`,
    );
    if (
      this.limits.maxDecodedGeometryBytes !== 0 &&
      next > this.limits.maxDecodedGeometryBytes
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'decoded-geometry-bytes',
        limit: this.limits.maxDecodedGeometryBytes,
        actual: next,
        path,
      });
    }
  }

  chargeDecodedGeometryBytes(byteLength: number, path: string): void {
    this.ensureDecodedGeometryBytes(byteLength, path);
    const next = checkedTotal(
      this.decodedGeometryByteTotal,
      byteLength,
      `${path} decoded geometry bytes`,
    );
    this.decodedGeometryByteTotal = next;
    this.decodedGeometryLastPath = path;
  }

  /**
   * Charge one encoded resource identity. Repeated observations of the same key
   * count once; if a later observation is larger, only the monotonic delta is
   * added to the aggregate.
   */
  ensureEncodedBytes(resourceKey: string, byteLength: number, path: string): void {
    if (typeof resourceKey !== 'string' || resourceKey.length === 0) {
      throw new TypeError('[vitrum/gltf-adapter] encoded resource key must be a non-empty string.');
    }
    nonNegativeSafeInteger(byteLength, `${path} encoded byte length`);
    if (
      this.limits.maxEncodedResourceBytes !== 0 &&
      byteLength > this.limits.maxEncodedResourceBytes
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'encoded-resource-bytes',
        limit: this.limits.maxEncodedResourceBytes,
        actual: byteLength,
        path,
        resourceKey,
      });
    }
    const previous = this.encodedBytesByKey.get(resourceKey) ?? 0;
    if (
      this.limits.maxTotalEncodedBytes !== 0 &&
      this.encodedByteTotal > this.limits.maxTotalEncodedBytes
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'total-encoded-bytes',
        limit: this.limits.maxTotalEncodedBytes,
        actual: this.encodedByteTotal,
        path,
        resourceKey,
      });
    }
    if (byteLength <= previous) return;
    const next = checkedTotal(
      this.encodedByteTotal,
      byteLength - previous,
      `${path} total encoded bytes`,
    );
    if (
      this.limits.maxTotalEncodedBytes !== 0 &&
      next > this.limits.maxTotalEncodedBytes
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'total-encoded-bytes',
        limit: this.limits.maxTotalEncodedBytes,
        actual: next,
        path,
        resourceKey,
      });
    }
  }

  chargeEncodedBytes(resourceKey: string, byteLength: number, path: string): void {
    this.ensureEncodedBytes(resourceKey, byteLength, path);
    const previous = this.encodedBytesByKey.get(resourceKey) ?? 0;
    if (byteLength <= previous) return;
    const next = checkedTotal(
      this.encodedByteTotal,
      byteLength - previous,
      `${path} total encoded bytes`,
    );
    this.encodedBytesByKey.set(resourceKey, byteLength);
    this.encodedPathsByKey.set(resourceKey, path);
    this.encodedByteTotal = next;
    this.encodedTotalLastPath = path;
    this.encodedTotalLastResourceKey = resourceKey;
  }

  ensureDecodedTexturePixels(pixelCount: number, path: string): void {
    nonNegativeSafeInteger(pixelCount, `${path} decoded pixel count`);
    if (
      this.limits.maxDecodedTexturePixels !== 0 &&
      pixelCount > this.limits.maxDecodedTexturePixels
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'decoded-texture-pixels',
        limit: this.limits.maxDecodedTexturePixels,
        actual: pixelCount,
        path,
      });
    }
    const next = checkedTotal(
      this.decodedTexturePixelTotal,
      pixelCount,
      `${path} total decoded texture pixels`,
    );
    if (
      this.limits.maxTotalDecodedTexturePixels !== 0 &&
      next > this.limits.maxTotalDecodedTexturePixels
    ) {
      throw new GltfResourceLimitError({
        limitKind: 'total-decoded-texture-pixels',
        limit: this.limits.maxTotalDecodedTexturePixels,
        actual: next,
        path,
      });
    }
  }

  chargeDecodedTexturePixels(pixelCount: number, path: string): void {
    this.ensureDecodedTexturePixels(pixelCount, path);
    const next = checkedTotal(
      this.decodedTexturePixelTotal,
      pixelCount,
      `${path} total decoded texture pixels`,
    );
    this.decodedTexturePixelTotal = next;
    this.decodedTextureLastPath = path;
    if (pixelCount > this.largestDecodedTexturePixelCharge) {
      this.largestDecodedTexturePixelCharge = pixelCount;
      this.largestDecodedTexturePixelPath = path;
    }
  }

  get decodedGeometryBytes(): number {
    return this.decodedGeometryByteTotal;
  }

  get totalEncodedBytes(): number {
    return this.encodedByteTotal;
  }

  get totalDecodedTexturePixels(): number {
    return this.decodedTexturePixelTotal;
  }
}

interface PendingOperation<T> {
  readonly operation: () => T | PromiseLike<T>;
  readonly resolve: (value: T | PromiseLike<T>) => void;
  readonly reject: (reason?: unknown) => void;
}

/** Fair FIFO limiter shared by fetch and decode work. */
export class AsyncResourceLimiter {
  readonly maxConcurrency: number;

  private active = 0;
  private readonly pending: Array<PendingOperation<unknown>> = [];

  constructor(maxConcurrency: number) {
    this.maxConcurrency = positiveSafeInteger(
      maxConcurrency,
      'resource operation concurrency',
    );
  }

  run<T>(operation: () => T | PromiseLike<T>): Promise<T> {
    if (typeof operation !== 'function') {
      return Promise.reject(
        new TypeError('[vitrum/gltf-adapter] limited resource operation must be a function.'),
      );
    }
    return new Promise<T>((resolve, reject) => {
      this.pending.push({
        operation,
        resolve,
        reject,
      } as PendingOperation<unknown>);
      this.pump();
    });
  }

  get activeCount(): number {
    return this.active;
  }

  get pendingCount(): number {
    return this.pending.length;
  }

  private pump(): void {
    while (this.active < this.maxConcurrency) {
      const pending = this.pending.shift();
      if (pending === undefined) return;
      this.active += 1;
      void Promise.resolve()
        .then(pending.operation)
        .then(pending.resolve, pending.reject)
        .finally(() => {
          this.active -= 1;
          this.pump();
        });
    }
  }
}

export function createAsyncResourceLimiter(
  maxConcurrency = DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS.maxConcurrentResourceOperations,
): AsyncResourceLimiter {
  return new AsyncResourceLimiter(maxConcurrency);
}

/**
 * Transaction-local ownership for handles created by glTF image decoders.
 *
 * Only decoder results are registered here. Texture handles that were already
 * present on a caller-owned Scene never enter this owner. Rejection closes each
 * registered identity at most once; success retains ownership on the returned
 * result until normalization supersedes the handle or the host releases it.
 */
export class DecodedImageHandleOwner {
  private readonly rollbackHandles = new Set<object>();
  private readonly closedHandles = new WeakSet<object>();

  track(handle: unknown): void {
    if (handle === null || (typeof handle !== 'object' && typeof handle !== 'function')) {
      return;
    }
    if (this.closedHandles.has(handle)) return;
    this.rollbackHandles.add(handle);
  }

  closeTracked(handle: unknown): void {
    if (handle === null || (typeof handle !== 'object' && typeof handle !== 'function')) {
      return;
    }
    const identity = handle;
    if (!this.rollbackHandles.delete(identity)) return;
    this.closeOnce(identity);
  }

  rollback(): void {
    const handles = [...this.rollbackHandles];
    this.rollbackHandles.clear();
    for (const handle of handles) this.closeOnce(handle);
  }

  retainOnly(reachableHandles: Iterable<unknown>): void {
    const reachable = new Set<object>();
    for (const handle of reachableHandles) {
      if (handle !== null && (typeof handle === 'object' || typeof handle === 'function')) {
        reachable.add(handle);
      }
    }
    for (const handle of [...this.rollbackHandles]) {
      if (reachable.has(handle)) continue;
      this.rollbackHandles.delete(handle);
      this.closeOnce(handle);
    }
  }

  private closeOnce(handle: object): void {
    if (this.closedHandles.has(handle)) return;
    this.closedHandles.add(handle);
    closeDecodedImageHandle(handle);
  }
}

function closeDecodedImageHandle(handle: object): void {
  try {
    const close = (handle as { readonly close?: unknown }).close;
    if (typeof close !== 'function') return;
    Reflect.apply(close, handle, []);
  } catch {
    // Preserve the import's original failure; handle cleanup is best effort.
  }
}

const GLTF_DECODED_IMAGE_HANDLE_OWNER = Symbol('vitrum.gltf.decodedImageHandleOwner');

type GltfResourceOwnerCarrier = {
  readonly [GLTF_DECODED_IMAGE_HANDLE_OWNER]?: DecodedImageHandleOwner;
};

/** @internal Attach decoder-handle ownership without changing public result enumeration. */
export function attachGltfResourceOwner<T extends object>(
  result: T,
  owner: DecodedImageHandleOwner,
): T {
  Object.defineProperty(result, GLTF_DECODED_IMAGE_HANDLE_OWNER, {
    value: owner,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  return result;
}

/**
 * Close decoder-created image handles retained by a successful glTF result.
 *
 * The operation is identity-deduplicated and idempotent. It accepts low-level
 * scene results, asset results, or engine bridge results (whose owner lives on
 * their nested `asset`). Callers must not release an engine result until that
 * engine/controller no longer uses the imported scene.
 */
export function releaseGltfResources(result: unknown): void {
  const directOwner = gltfResourceOwner(result);
  if (directOwner !== undefined) {
    directOwner.rollback();
    return;
  }
  if (result === null || (typeof result !== 'object' && typeof result !== 'function')) {
    return;
  }
  let asset: unknown;
  try {
    asset = (result as { readonly asset?: unknown }).asset;
  } catch {
    return;
  }
  gltfResourceOwner(asset)?.rollback();
}

function gltfResourceOwner(value: unknown): DecodedImageHandleOwner | undefined {
  if (value === null || (typeof value !== 'object' && typeof value !== 'function')) {
    return undefined;
  }
  try {
    const owner = (value as GltfResourceOwnerCarrier)[GLTF_DECODED_IMAGE_HANDLE_OWNER];
    return owner instanceof DecodedImageHandleOwner ? owner : undefined;
  } catch {
    return undefined;
  }
}

/** Shared state passed through every stage of one top-level import. */
export interface GltfImportResourceContext {
  readonly ledger: ImportResourceLedger;
  readonly limiter: AsyncResourceLimiter;
  readonly decodedImageHandles: DecodedImageHandleOwner;
}
