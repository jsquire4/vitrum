import { runInNewContext } from 'node:vm';
import { describe, expect, it } from 'vitest';
import {
  AsyncResourceLimiter,
  DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS,
  GltfResourceLimitError,
  ImportResourceLedger,
  gltfArrayBufferByteLength,
  normalizeGltfImportResourceLimits,
} from './importResourceBudget.js';

describe('glTF import resource limits', () => {
  it('normalizes partial limits without treating explicit zero as absent', () => {
    const normalized = normalizeGltfImportResourceLimits({
      maxDecodedGeometryBytes: 0,
      maxConcurrentResourceOperations: 2,
    });

    expect(normalized).toEqual({
      ...DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS,
      maxDecodedGeometryBytes: 0,
      maxConcurrentResourceOperations: 2,
    });
    expect(() => normalizeGltfImportResourceLimits({
      maxConcurrentResourceOperations: 0,
    })).toThrow(/positive safe integer/);
    expect(() => normalizeGltfImportResourceLimits({
      maxTotalEncodedBytes: -1,
    })).toThrow(/non-negative safe integer/);
  });

  it('enforces one monotonic aggregate decoded-geometry ceiling', () => {
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 20,
    });
    ledger.chargeDecodedGeometryBytes(12, 'first accessor');

    expect(() => ledger.chargeDecodedGeometryBytes(9, 'derived normals'))
      .toThrowError(GltfResourceLimitError);
    expect(ledger.decodedGeometryBytes).toBe(12);

    const disabled = new ImportResourceLedger({
      maxDecodedGeometryBytes: 0,
    });
    disabled.chargeDecodedGeometryBytes(Number.MAX_SAFE_INTEGER, 'disabled');
    expect(disabled.decodedGeometryBytes).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('preflights decoded geometry without mutating the monotonic total', () => {
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 20,
    });
    ledger.chargeDecodedGeometryBytes(12, 'first accessor');

    ledger.ensureDecodedGeometryBytes(8, 'exact remaining allocation');
    expect(ledger.decodedGeometryBytes).toBe(12);
    expect(() => ledger.ensureDecodedGeometryBytes(9, 'oversized allocation'))
      .toThrowError(GltfResourceLimitError);
    expect(ledger.decodedGeometryBytes).toBe(12);
  });

  it('deduplicates encoded resources by key and charges only larger deltas', () => {
    const ledger = new ImportResourceLedger({
      maxEncodedResourceBytes: 10,
      maxTotalEncodedBytes: 12,
    });

    ledger.ensureEncodedBytes('buffer:0', 8, 'preflight');
    expect(ledger.totalEncodedBytes).toBe(0);
    ledger.chargeEncodedBytes('buffer:0', 8, 'load');
    ledger.chargeEncodedBytes('buffer:0', 6, 'repeat');
    expect(ledger.totalEncodedBytes).toBe(8);
    ledger.chargeEncodedBytes('buffer:0', 10, 'larger repeat');
    expect(ledger.totalEncodedBytes).toBe(10);

    let encodedError: unknown;
    try {
      ledger.chargeEncodedBytes('image:0', 3, 'image');
    } catch (error) {
      encodedError = error;
    }
    expect(encodedError).toMatchObject({
        code: 'GLTF_RESOURCE_LIMIT_EXCEEDED',
        limitKind: 'total-encoded-bytes',
        limit: 12,
        actual: 13,
        path: 'image',
        resourceKey: 'image:0',
      });
    expect(ledger.totalEncodedBytes).toBe(10);
  });

  it('enforces per-image and aggregate decoded-pixel limits independently', () => {
    const ledger = new ImportResourceLedger({
      maxDecodedTexturePixels: 8,
      maxTotalDecodedTexturePixels: 10,
    });
    ledger.chargeDecodedTexturePixels(6, 'image 0');
    let perImageError: unknown;
    try {
      ledger.chargeDecodedTexturePixels(9, 'image 1');
    } catch (error) {
      perImageError = error;
    }
    expect(perImageError)
      .toMatchObject({ limitKind: 'decoded-texture-pixels', actual: 9 });
    let totalError: unknown;
    try {
      ledger.chargeDecodedTexturePixels(5, 'image 2');
    } catch (error) {
      totalError = error;
    }
    expect(totalError)
      .toMatchObject({ limitKind: 'total-decoded-texture-pixels', actual: 11 });
    expect(ledger.totalDecodedTexturePixels).toBe(6);
  });

  it('reconfigures one ledger without resetting prior charges or resource keys', () => {
    const originalLimits = {
      maxDecodedGeometryBytes: 100,
      maxEncodedResourceBytes: 100,
      maxTotalEncodedBytes: 100,
      maxDecodedTexturePixels: 100,
      maxTotalDecodedTexturePixels: 100,
      maxConcurrentResourceOperations: 4,
    } as const;
    const ledger = new ImportResourceLedger(originalLimits);
    ledger.chargeDecodedGeometryBytes(12, 'geometry');
    ledger.chargeEncodedBytes('image:0', 8, 'image acquisition');
    ledger.chargeEncodedBytes('image:1', 3, 'second image acquisition');
    ledger.chargeDecodedTexturePixels(4, 'decoded acquisition');
    ledger.chargeDecodedTexturePixels(2, 'second decoded acquisition');

    const exactUsageLimits = {
      maxDecodedGeometryBytes: 12,
      maxEncodedResourceBytes: 8,
      maxTotalEncodedBytes: 11,
      maxDecodedTexturePixels: 4,
      maxTotalDecodedTexturePixels: 6,
      maxConcurrentResourceOperations: 2,
    } as const;
    const rejected = [
      {
        limits: { ...exactUsageLimits, maxDecodedGeometryBytes: 11 },
        error: {
          limitKind: 'decoded-geometry-bytes',
          actual: 12,
          path: 'geometry',
        },
      },
      {
        limits: { ...exactUsageLimits, maxEncodedResourceBytes: 7 },
        error: {
          limitKind: 'encoded-resource-bytes',
          actual: 8,
          path: 'image acquisition',
          resourceKey: 'image:0',
        },
      },
      {
        limits: { ...exactUsageLimits, maxTotalEncodedBytes: 10 },
        error: {
          limitKind: 'total-encoded-bytes',
          actual: 11,
          path: 'second image acquisition',
          resourceKey: 'image:1',
        },
      },
      {
        limits: { ...exactUsageLimits, maxDecodedTexturePixels: 3 },
        error: {
          limitKind: 'decoded-texture-pixels',
          actual: 4,
          path: 'decoded acquisition',
        },
      },
      {
        limits: { ...exactUsageLimits, maxTotalDecodedTexturePixels: 5 },
        error: {
          limitKind: 'total-decoded-texture-pixels',
          actual: 6,
          path: 'second decoded acquisition',
        },
      },
    ] as const;

    for (const candidate of rejected) {
      expect(() => ledger.reconfigureLimits(candidate.limits))
        .toThrowError(GltfResourceLimitError);
      try {
        ledger.reconfigureLimits(candidate.limits);
      } catch (error) {
        expect(error).toMatchObject(candidate.error);
      }
      expect(ledger.limits).toEqual(originalLimits);
    }

    ledger.reconfigureLimits(exactUsageLimits);

    expect(ledger.decodedGeometryBytes).toBe(12);
    expect(ledger.totalEncodedBytes).toBe(11);
    expect(ledger.totalDecodedTexturePixels).toBe(6);
    expect(ledger.limits).toEqual(exactUsageLimits);
    ledger.chargeEncodedBytes('image:0', 8, 'raw snapshot');
    expect(ledger.totalEncodedBytes).toBe(11);
  });
});

describe('AsyncResourceLimiter', () => {
  it('starts work FIFO, caps active operations, and releases slots on rejection', async () => {
    const limiter = new AsyncResourceLimiter(2);
    const started: number[] = [];
    const releases = new Map<number, () => void>();
    let active = 0;
    let peak = 0;
    const operation = (id: number, reject = false): Promise<number> =>
      limiter.run(async () => {
        started.push(id);
        active += 1;
        peak = Math.max(peak, active);
        await new Promise<void>((resolve) => releases.set(id, resolve));
        active -= 1;
        if (reject) throw new Error(`failed ${id}`);
        return id;
      });

    const first = operation(0);
    const second = operation(1, true);
    const third = operation(2);
    await Promise.resolve();
    await Promise.resolve();
    expect(started).toEqual([0, 1]);
    expect(limiter.activeCount).toBe(2);
    expect(limiter.pendingCount).toBe(1);

    releases.get(1)!();
    await expect(second).rejects.toThrow('failed 1');
    await Promise.resolve();
    expect(started).toEqual([0, 1, 2]);
    expect(peak).toBe(2);

    releases.get(0)!();
    releases.get(2)!();
    await expect(Promise.all([first, third])).resolves.toEqual([0, 2]);
    expect(limiter.activeCount).toBe(0);
    expect(limiter.pendingCount).toBe(0);
  });
});

describe('gltfArrayBufferByteLength', () => {
  it('uses the intrinsic brand/length across realms and ignores shadowing', () => {
    const foreign = runInNewContext('new ArrayBuffer(7)') as ArrayBuffer;
    Object.defineProperty(foreign, 'byteLength', { value: 0 });
    expect(gltfArrayBufferByteLength(foreign)).toBe(7);
    expect(gltfArrayBufferByteLength(new Uint8Array(7))).toBeUndefined();
    expect(gltfArrayBufferByteLength({ byteLength: 7 })).toBeUndefined();
    if (typeof SharedArrayBuffer !== 'undefined') {
      expect(gltfArrayBufferByteLength(new SharedArrayBuffer(7))).toBeUndefined();
    }
  });
});
