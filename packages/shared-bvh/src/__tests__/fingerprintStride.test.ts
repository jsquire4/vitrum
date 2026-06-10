/**
 * H34-f unit tests — bufferFingerprint fixed-stride sampling
 *
 * Verifies that:
 *  1. A single-byte change at a sampled offset (stride-aligned) changes the fingerprint.
 *  2. The sampling guarantee is documented: miss probability for an interior
 *     non-stride-aligned byte is (stride-1)/stride.
 *  3. Small buffers (≤ 65536 bytes) still use the exact path.
 */

import { describe, expect, it } from 'vitest';
import { fingerprintBuffer } from '../bufferFingerprint.js';

const MAX_SAMPLE_POINTS = 65536;

describe('H34-f: bufferFingerprint evenly-spaced sampling', () => {
  it('small buffer (exact path): any single-byte change changes the fingerprint', () => {
    const a = new Uint8Array(MAX_SAMPLE_POINTS);
    const b = new Uint8Array(MAX_SAMPLE_POINTS);
    a.fill(0);
    b.fill(0);
    b[1000] = 42;  // interior byte change
    expect(fingerprintBuffer(a.buffer)).not.toBe(fingerprintBuffer(b.buffer));
  });

  it('large buffer: a single-byte change at a stride-aligned offset changes the fingerprint', () => {
    // 1 MiB buffer → stride = floor(1048576 / 65536) = 16
    const SIZE = 1024 * 1024;
    const stride = Math.max(1, Math.floor(SIZE / MAX_SAMPLE_POINTS));
    const a = new Uint8Array(SIZE);
    const b = new Uint8Array(SIZE);
    a.fill(0);
    b.fill(0);
    // Change a byte that is stride-aligned (guaranteed to be sampled).
    const sampledOffset = stride * 100;  // 100 stride steps in
    b[sampledOffset] = 0xff;
    expect(fingerprintBuffer(a.buffer)).not.toBe(fingerprintBuffer(b.buffer));
  });

  it('large buffer: the last byte is always included', () => {
    const SIZE = 1024 * 1024;
    const a = new Uint8Array(SIZE);
    const b = new Uint8Array(SIZE);
    a.fill(0);
    b.fill(0);
    b[SIZE - 1] = 0xab;  // last byte
    expect(fingerprintBuffer(a.buffer)).not.toBe(fingerprintBuffer(b.buffer));
  });

  it('large buffer: the first byte is always included', () => {
    const SIZE = 1024 * 1024;
    const a = new Uint8Array(SIZE);
    const b = new Uint8Array(SIZE);
    a.fill(0);
    b.fill(0);
    b[0] = 0xcd;  // first byte
    expect(fingerprintBuffer(a.buffer)).not.toBe(fingerprintBuffer(b.buffer));
  });

  it('identical large buffers produce the same fingerprint', () => {
    const SIZE = 1024 * 1024;
    const a = new Uint8Array(SIZE);
    const b = new Uint8Array(SIZE);
    a.fill(0xaa);
    b.fill(0xaa);
    expect(fingerprintBuffer(a.buffer)).toBe(fingerprintBuffer(b.buffer));
  });
});
