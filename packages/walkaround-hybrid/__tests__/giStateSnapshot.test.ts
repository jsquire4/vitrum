// giStateSnapshot.test.ts — the cached-light-field serialization round-trips
// losslessly (the GPU export→import round-trip is validated separately on a
// wsl-gpu hybrid capture; this pins the pure-CPU container).
import { describe, it, expect } from 'vitest';
import {
  serializeGIState,
  deserializeGIState,
  type GIStateSnapshot,
  type RestirGISnapshot,
  type PpgSnapshot,
} from '../src/giStateSnapshot.js';
import { serialiseSTree } from '../src/ppg/serialise.js';
import { buildSTree, splitOverflowLeaves } from '../src/ppg/sTree.js';
import { dTreeAccumulateFlux } from '../src/ppg/dTree.js';

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

const RESTIR_GI_GRIS_STRIDE = 30; // GRIS/ReSTIR-PT reservoir stride (u32 per reservoir pixel)
const RESTIR_GI_COMPACT_STRIDE = 20; // default Sprint-16/17 reservoir stride

function makeRestirSection(strideU32 = RESTIR_GI_GRIS_STRIDE): RestirGISnapshot {
  const halfW = 5, halfH = 7;
  const bufU32Len = halfW * halfH * strideU32;
  const mk = (salt: number): Uint32Array => {
    const a = new Uint32Array(bufU32Len);
    for (let i = 0; i < a.length; i++) a[i] = (i * 2654435761 + salt) >>> 0; // Knuth-ish fill
    return a;
  };
  return {
    halfW, halfH, strideU32,
    current: mk(1), previous: mk(2), spatial: mk(3),
  };
}

/** Hand-build a v1 octahedral-irradiance buffer to assert the v3 SH break rejects it. */
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

  it('round-trips the ReSTIR-GI reservoir section losslessly (v3)', () => {
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

  it('round-trips the compact default ReSTIR-GI reservoir stride losslessly', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection(RESTIR_GI_COMPACT_STRIDE) };
    const back = deserializeGIState(serializeGIState(s));
    expect(back.restirGI).toBeDefined();
    const r = back.restirGI!;
    expect(r.strideU32).toBe(RESTIR_GI_COMPACT_STRIDE);
    expect(r.current.length).toBe(s.restirGI.halfW * s.restirGI.halfH * RESTIR_GI_COMPACT_STRIDE);
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

  it('rejects a v1 (octahedral-irradiance) buffer at the SH break', () => {
    // v1/v2 stored an OCTAHEDRAL irradiance atlas; the SH-era sampler would read
    // those bytes as garbage, so the v2->v3 break intentionally drops backward
    // compatibility for irradiance — a v1 buffer must now throw, not silently
    // decode.
    const s = makeSnapshot();
    const v1 = makeV1Buffer(s);
    expect(() => deserializeGIState(v1)).toThrow(/unsupported version 1/);
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

  // ── v3 backward-compat: v3 buffers accepted, ppg absent (cold start) ───────
  it('accepts a v3 (no PPG section) buffer and returns ppg:undefined', () => {
    const s = makeSnapshot();
    const buf = serializeGIState(s);
    // v3 snapshots have no SECTION_PPG bit, so deserialising returns ppg:undefined.
    // Confirm that the current serializer produces v5 and the v3 path is via
    // hand-patching the version word down to 3 (same layout, only section flags differ).
    // Since no v3 snaps exist in the wild yet, we simulate one by writing version=3.
    new DataView(buf).setUint32(4, 3, true);
    const back = deserializeGIState(buf);
    expect(back.ppg).toBeUndefined();
    // Core fields still decoded correctly.
    expect(back.dims).toEqual(s.dims);
  });
});

// ── PPG snapshot section (v5) ────────────────────────────────────────────────

/**
 * Build a multi-cell STree with non-trivial directional distributions,
 * then serialise it to a PpgSnapshot. Mirrors the real coordinator path.
 */
function makePpgSnapshot(maxSpatialCells = 1024, maxDTreeNodesPerCell = 341): PpgSnapshot {
  const sceneBounds = { min: [-5, -5, -5] as [number, number, number], max: [5, 5, 5] as [number, number, number] };
  const sTree = buildSTree(sceneBounds);

  // Accumulate flux into the root dTree so refineDTree has signal to split.
  const rootDTree = sTree.dTrees[0]!;
  for (let i = 0; i < 200; i++) {
    dTreeAccumulateFlux(rootDTree, [i / 200, 0.25], 1.0 + i * 0.01);
  }
  // Force a split by manually bumping sampleCount above the default threshold.
  sTree.nodes[0]!.sampleCount = 50_000;
  splitOverflowLeaves(sTree, 12_000, maxSpatialCells);

  const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(sTree);
  return {
    maxSpatialCells,
    maxDTreeNodesPerCell,
    sTreeBuf,
    dTreeBuf,
    dTreeOffsets,
    sceneBoundsMin: sceneBounds.min,
    sceneBoundsMax: sceneBounds.max,
  };
}

describe('GI state snapshot v5 PPG section', () => {
  it('round-trips the PPG section byte-identically (sTreeBuf, dTreeBuf, dTreeOffsets)', () => {
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), ppg };
    const back = deserializeGIState(serializeGIState(s));
    expect(back.ppg).toBeDefined();
    const p = back.ppg!;
    expect(p.maxSpatialCells).toBe(ppg.maxSpatialCells);
    expect(p.maxDTreeNodesPerCell).toBe(ppg.maxDTreeNodesPerCell);
    expect(Array.from(p.sTreeBuf)).toEqual(Array.from(ppg.sTreeBuf));
    expect(Array.from(p.dTreeBuf)).toEqual(Array.from(ppg.dTreeBuf));
    expect(Array.from(p.dTreeOffsets)).toEqual(Array.from(ppg.dTreeOffsets));
    expect(p.sceneBoundsMin).toEqual(ppg.sceneBoundsMin);
    expect(p.sceneBoundsMax).toEqual(ppg.sceneBoundsMax);
  });

  it('re-serialization after round-trip is byte-identical (mirror DDGI atlas pattern)', () => {
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), ppg };
    const first  = serializeGIState(s);
    const back   = deserializeGIState(first);
    const second = serializeGIState(back);
    expect(second.byteLength).toBe(first.byteLength);
    expect(Array.from(new Uint8Array(second))).toEqual(Array.from(new Uint8Array(first)));
  });

  it('PPG section is absent when ppg is not provided (DDGI-only payload)', () => {
    const s = makeSnapshot(); // no ppg
    const back = deserializeGIState(serializeGIState(s));
    expect(back.ppg).toBeUndefined();
  });

  it('coexists with ReSTIR-GI section: both sections round-trip correctly', () => {
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), restirGI: makeRestirSection(), ppg };
    const back = deserializeGIState(serializeGIState(s));
    // DDGI intact.
    expect(Array.from(back.irrData)).toEqual(Array.from(s.irrData));
    // ReSTIR-GI intact.
    expect(back.restirGI).toBeDefined();
    expect(Array.from(back.restirGI!.current)).toEqual(Array.from(s.restirGI.current));
    // PPG intact.
    expect(back.ppg).toBeDefined();
    expect(Array.from(back.ppg!.sTreeBuf)).toEqual(Array.from(ppg.sTreeBuf));
    expect(Array.from(back.ppg!.dTreeOffsets)).toEqual(Array.from(ppg.dTreeOffsets));
  });

  it('correctly sizes the buffer: header + atlases + restirGI + PPG blobs', () => {
    const ppg = makePpgSnapshot();
    const restirGI = makeRestirSection();
    const s = { ...makeSnapshot(), restirGI, ppg };
    const buf = serializeGIState(s);
    const reservoirBytes = 20 + restirGI.current.byteLength + restirGI.previous.byteLength + restirGI.spatial.byteLength;
    const ppgBytes = 48 /* PPG_SUBHEADER_BYTES */ + ppg.sTreeBuf.byteLength + ppg.dTreeBuf.byteLength + ppg.dTreeOffsets.byteLength;
    expect(buf.byteLength).toBe(64 + s.irrData.byteLength + s.visData.byteLength + reservoirBytes + ppgBytes);
  });

  it('round-trips a non-default maxDTreeNodesPerCell in the v5 PPG sub-header', () => {
    const ppg = makePpgSnapshot(2048, 97);
    const s = { ...makeSnapshot(), ppg };
    const back = deserializeGIState(serializeGIState(s));
    expect(back.ppg?.maxSpatialCells).toBe(2048);
    expect(back.ppg?.maxDTreeNodesPerCell).toBe(97);
  });

  it('defaults v4 PPG snapshots to the historical 341-node dTree cap', () => {
    const ppg = makePpgSnapshot(1024, 97);
    const s = { ...makeSnapshot(), ppg };
    const buf = serializeGIState(s);
    new DataView(buf).setUint32(4, 4, true); // v4 field [3] was not maxDTreeNodesPerCell.
    const back = deserializeGIState(buf);
    expect(back.ppg?.maxDTreeNodesPerCell).toBe(341);
  });

  it('rejects a PPG blob truncated below its declared size', () => {
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), ppg };
    const full = serializeGIState(s);
    // Drop enough bytes to truncate the dTreeOffsets blob.
    const truncated = full.slice(0, full.byteLength - ppg.dTreeOffsets.byteLength);
    expect(() => deserializeGIState(truncated)).toThrow(/PPG tree data/);
  });

  it('rejects a PPG blob where dTreeOffsets length does not match declared dTreeCount', () => {
    // Build a valid snapshot, then corrupt the dTreeCount field in the sub-header.
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), ppg };
    const buf = serializeGIState(s);
    // The PPG sub-header starts at HEADER_BYTES(64) + irrBytes + visBytes.
    const irrBytes = s.irrW * s.irrH * 8;
    const visBytes = s.visW * s.visH * 8;
    const ppgSubheaderOffset = 64 + irrBytes + visBytes; // no ReSTIR section in this snapshot
    // Field [2] (offset +8 from sub-header start) is dTreeCount.
    new DataView(buf).setUint32(ppgSubheaderOffset + 8, ppg.dTreeOffsets.length + 99, true);
    expect(() => deserializeGIState(buf)).toThrow(/dTreeOffsets length/);
  });
});
