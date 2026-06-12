import { describe, expect, it } from 'vitest';
import {
  fingerprintBuffer,
  fingerprintBufferExact,
  fingerprintBuffers,
  fingerprintBuffersExact,
  isTlasOnlyVersionBump,
} from '../bufferFingerprint.js';

describe('bufferFingerprint', () => {
  it('changes when same-length buffer content changes', () => {
    const a = new Uint8Array(32);
    const b = new Uint8Array(32);
    a[0] = 1;
    b[0] = 2;
    expect(fingerprintBuffer(a.buffer)).not.toBe(fingerprintBuffer(b.buffer));
  });

  it('fingerprintBuffers combines multiple parts', () => {
    const x = new Uint8Array([1, 2, 3]);
    const y = new Uint8Array([4, 5, 6]);
    const combined = fingerprintBuffers(x.buffer, y.buffer);
    expect(combined).not.toBe(fingerprintBuffer(x.buffer));
    expect(combined).not.toBe(fingerprintBuffer(y.buffer));
  });

  it('exact fingerprint catches an unsampled interior byte in a large buffer', () => {
    const a = new Uint8Array(1024 * 1024);
    const b = new Uint8Array(1024 * 1024);
    b[1] = 1;

    expect(fingerprintBuffer(a.buffer)).toBe(fingerprintBuffer(b.buffer));
    expect(fingerprintBufferExact(a.buffer)).not.toBe(fingerprintBufferExact(b.buffer));
  });

  it('fingerprintBuffersExact combines exact per-buffer fingerprints', () => {
    const a = new Uint8Array(1024 * 1024);
    const b = new Uint8Array(1024 * 1024);
    b[1] = 1;

    expect(fingerprintBuffers(a.buffer)).toBe(fingerprintBuffers(b.buffer));
    expect(fingerprintBuffersExact(a.buffer)).not.toBe(fingerprintBuffersExact(b.buffer));
  });

  it('isTlasOnlyVersionBump detects transform-only TLAS bumps', () => {
    expect(isTlasOnlyVersionBump(10, 11, { blasContentVersion: 10, tlasContentVersion: 10 })).toBe(true);
    expect(isTlasOnlyVersionBump(10, 10, { blasContentVersion: 10, tlasContentVersion: 10 })).toBe(false);
    expect(isTlasOnlyVersionBump(11, 12, { blasContentVersion: 10, tlasContentVersion: 10 })).toBe(false);
  });
});
