// giStateSnapshot.test.ts — the cached-light-field serialization round-trips
// losslessly (the GPU export→import round-trip is validated separately on a
// wsl-gpu hybrid capture; this pins the pure-CPU container).
import { describe, it, expect } from 'vitest';
import {
  serializeGIState,
  deserializeGIState,
  type GIStateSnapshot,
} from '../src/giStateSnapshot.js';

function makeSnapshot(): GIStateSnapshot {
  const irrW = 6, irrH = 8, visW = 10, visH = 12;
  const irrData = new Uint16Array(irrW * irrH * 4);
  const visData = new Uint16Array(visW * visH * 4);
  for (let i = 0; i < irrData.length; i++) irrData[i] = (i * 37 + 1) & 0xffff;
  for (let i = 0; i < visData.length; i++) visData[i] = (i * 53 + 7) & 0xffff;
  return {
    dims: { x: 3, y: 4, z: 5 },
    origin: [-1.5, 2.25, -3.75],
    spacing: 24,
    irrW, irrH, visW, visH,
    irrData, visData,
  };
}

describe('GI state snapshot serialization', () => {
  it('round-trips every field + the raw atlas bytes losslessly', () => {
    const s = makeSnapshot();
    const back = deserializeGIState(serializeGIState(s));
    expect(back.dims).toEqual(s.dims);
    expect(back.origin).toEqual(s.origin);
    expect(back.spacing).toBe(s.spacing);
    expect([back.irrW, back.irrH, back.visW, back.visH]).toEqual([s.irrW, s.irrH, s.visW, s.visH]);
    expect(Array.from(back.irrData)).toEqual(Array.from(s.irrData)); // identity
    expect(Array.from(back.visData)).toEqual(Array.from(s.visData));
  });

  it('produces a single transferable ArrayBuffer sized header + both atlases', () => {
    const s = makeSnapshot();
    const buf = serializeGIState(s);
    expect(buf).toBeInstanceOf(ArrayBuffer);
    expect(buf.byteLength).toBe(64 + s.irrData.byteLength + s.visData.byteLength);
  });

  it('rejects a buffer with a bad magic', () => {
    const bogus = new ArrayBuffer(64);
    expect(() => deserializeGIState(bogus)).toThrow(/bad magic/);
  });

  it('rejects a truncated buffer (dims declare more than is present)', () => {
    const s = makeSnapshot();
    const full = serializeGIState(s);
    const truncated = full.slice(0, full.byteLength - 16);
    expect(() => deserializeGIState(truncated)).toThrow(/too small/);
  });
});
