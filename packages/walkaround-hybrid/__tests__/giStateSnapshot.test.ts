// giStateSnapshot.test.ts — the cached-light-field serialization round-trips
// losslessly (the GPU export→import round-trip is validated separately on a
// wsl-gpu hybrid capture; this pins the pure-CPU container).
import { describe, it, expect } from 'vitest';
import {
  serializeGIState,
  deserializeGIState,
  type GIStateSnapshot,
  type RestirGISnapshot,
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

const RESTIR_GI_STRIDE = 30; // RESERVOIR_GI_STRIDE (u32 per reservoir pixel)

function makeRestirSection(): RestirGISnapshot {
  const halfW = 5, halfH = 7;
  const bufU32Len = halfW * halfH * RESTIR_GI_STRIDE;
  const mk = (salt: number): Uint32Array => {
    const a = new Uint32Array(bufU32Len);
    for (let i = 0; i < a.length; i++) a[i] = (i * 2654435761 + salt) >>> 0; // Knuth-ish fill
    return a;
  };
  return {
    halfW, halfH, strideU32: RESTIR_GI_STRIDE,
    current: mk(1), previous: mk(2), spatial: mk(3),
  };
}

/** Hand-build a v1 (DDGI-only) buffer to assert v2 deserialize stays back-compatible. */
function makeV1Buffer(s: GIStateSnapshot): ArrayBuffer {
  const irrBytes = s.irrData.byteLength;
  const visBytes = s.visData.byteLength;
  const buf = new ArrayBuffer(64 + irrBytes + visBytes);
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint32(o, 0x47495353, true); o += 4; // magic "GISS"
  dv.setUint32(o, 1, true); o += 4;           // version 1
  dv.setUint32(o, s.dims.x, true); o += 4;
  dv.setUint32(o, s.dims.y, true); o += 4;
  dv.setUint32(o, s.dims.z, true); o += 4;
  dv.setFloat32(o, s.origin[0], true); o += 4;
  dv.setFloat32(o, s.origin[1], true); o += 4;
  dv.setFloat32(o, s.origin[2], true); o += 4;
  dv.setFloat32(o, s.spacing, true); o += 4;
  dv.setUint32(o, s.irrW, true); o += 4;
  dv.setUint32(o, s.irrH, true); o += 4;
  dv.setUint32(o, s.visW, true); o += 4;
  dv.setUint32(o, s.visH, true); o += 4;
  // o = 52: v1 left the remaining header bytes zero (no section-flags word).
  new Uint8Array(buf, 64, irrBytes).set(new Uint8Array(s.irrData.buffer, s.irrData.byteOffset, irrBytes));
  new Uint8Array(buf, 64 + irrBytes, visBytes).set(new Uint8Array(s.visData.buffer, s.visData.byteOffset, visBytes));
  return buf;
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

  it('round-trips the ReSTIR-GI reservoir section losslessly (v2)', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
    const back = deserializeGIState(serializeGIState(s));
    // DDGI half unchanged.
    expect(Array.from(back.irrData)).toEqual(Array.from(s.irrData));
    expect(Array.from(back.visData)).toEqual(Array.from(s.visData));
    // Reservoir section present + byte-identical.
    expect(back.restirGI).toBeDefined();
    const r = back.restirGI!;
    expect([r.halfW, r.halfH, r.strideU32]).toEqual([s.restirGI.halfW, s.restirGI.halfH, s.restirGI.strideU32]);
    expect(Array.from(r.current)).toEqual(Array.from(s.restirGI.current));
    expect(Array.from(r.previous)).toEqual(Array.from(s.restirGI.previous));
    expect(Array.from(r.spatial)).toEqual(Array.from(s.restirGI.spatial));
  });

  it('sizes the buffer for header + atlases + reservoir sub-block when reservoirs are present', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
    const buf = serializeGIState(s);
    const r = s.restirGI;
    const reservoirBytes = 20 /* sub-header */ + r.current.byteLength + r.previous.byteLength + r.spatial.byteLength;
    expect(buf.byteLength).toBe(64 + s.irrData.byteLength + s.visData.byteLength + reservoirBytes);
  });

  it('omits the reservoir section when restirGI is absent (DDGI-only payload)', () => {
    const s = makeSnapshot(); // no restirGI
    const buf = serializeGIState(s);
    expect(buf.byteLength).toBe(64 + s.irrData.byteLength + s.visData.byteLength);
    expect(deserializeGIState(buf).restirGI).toBeUndefined();
  });

  it('back-compat: deserializes a v1 (DDGI-only) buffer with restirGI undefined', () => {
    const s = makeSnapshot();
    const v1 = makeV1Buffer(s);
    const back = deserializeGIState(v1);
    expect(back.dims).toEqual(s.dims);
    expect(Array.from(back.irrData)).toEqual(Array.from(s.irrData));
    expect(Array.from(back.visData)).toEqual(Array.from(s.visData));
    expect(back.restirGI).toBeUndefined();
  });

  it('rejects an unsupported version', () => {
    const s = makeSnapshot();
    const buf = serializeGIState(s);
    new DataView(buf).setUint32(4, 99, true); // clobber the version word
    expect(() => deserializeGIState(buf)).toThrow(/unsupported version/);
  });

  it('rejects a reservoir section truncated below its declared dims', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
    const full = serializeGIState(s);
    // Drop the last reservoir buffer's worth of bytes so the section under-runs.
    const truncated = full.slice(0, full.byteLength - s.restirGI.spatial.byteLength);
    expect(() => deserializeGIState(truncated)).toThrow(/ReSTIR-GI reservoir/);
  });
});
