import { describe, expect, it } from 'vitest';
import { fingerprintBuffer, fingerprintBuffers } from '../bufferFingerprint.js';

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
});
