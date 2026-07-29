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
import {
  GI_STATE_COMPATIBILITY_SCHEMA,
  GI_STATE_COMPATIBILITY_WORDS,
} from '../src/giStateCompatibility.js';

function makeSnapshot(): GIStateSnapshot {
  // Canonical atlas geometry for dims 3×4×5: irradiance stride 5,
  // visibility stride 18, with Y/Z stacked vertically.
  const irrW = 15, irrH = 100, visW = 54, visH = 360;
  const irrData = new Uint16Array(irrW * irrH * 4);
  const visData = new Uint16Array(visW * visH * 4);
  const probeStateW = 3, probeStateH = 20;
  const probeStateData = new Float32Array(probeStateW * probeStateH * 4);
  for (let i = 0; i < irrData.length; i++) irrData[i] = (i * 37 + 1) & 0x7bff;
  for (let i = 0; i < visData.length; i++) visData[i] = (i * 53 + 7) & 0x7bff;
  for (let i = 0; i < probeStateData.length; i++) {
    probeStateData[i] = i % 4 === 3 ? Number(i % 8 === 3) : (i - 17) * 0.001;
  }
  const compatibility = new Uint32Array(GI_STATE_COMPATIBILITY_WORDS);
  compatibility[0] = GI_STATE_COMPATIBILITY_SCHEMA;
  return {
    dims: { x: 3, y: 4, z: 5 },
    origin: [-1.5, 2.25, -3.75],
    spacing: 24,
    irrW, irrH, visW, visH,
    irrData, visData,
    probeStateW, probeStateH, probeStateData,
    compatibility,
  };
}

const RESTIR_GI_GRIS_STRIDE = 28; // sole live generalized reservoir stride
const RESTIR_GI_COMPACT_STRIDE = 20; // legacy snapshot-only Sprint-16/17 stride
const COMPAT_EXTENSION_BYTES =
  32 + GI_STATE_COMPATIBILITY_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const RESTIR_BASE_FLOAT_LANES = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
] as const;
const RESTIR_GRIS_FLOAT_LANES = [20, 21, 22, 23, 26] as const;

function makeRestirSection(strideU32 = RESTIR_GI_GRIS_STRIDE): RestirGISnapshot {
  const halfW = 5, halfH = 7;
  const bufU32Len = halfW * halfH * strideU32;
  const mk = (salt: number): Uint32Array => {
    const a = new Uint32Array(bufU32Len);
    const f = new Float32Array(a.buffer);
    const baseFloatLanes = [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
    ];
    for (let record = 0; record < halfW * halfH; record += 1) {
      const base = record * strideU32;
      for (const lane of baseFloatLanes) {
        f[base + lane] = (salt + record + lane) * 0.001;
      }
      a[base + 15] = (salt + record) >>> 0;
      a[base + 19] = (salt * 17 + record) >>> 0;
      if (strideU32 === RESTIR_GI_GRIS_STRIDE) {
        for (const lane of [20, 21, 22, 23, 26]) {
          f[base + lane] = (salt + record + lane) * 0.0001;
        }
        a[base + 24] = 1; // reconnectable generalized sample
        a[base + 25] = record & 1;
        a[base + 27] = (salt + record) >>> 0;
      }
    }
    return a;
  };
  return {
    halfW, halfH, strideU32,
    current: mk(1), previous: mk(2), spatial: mk(3),
  };
}

/** Build a historical v6 container without asking the v7 serializer to emit it. */
function makeLegacyCompactV6Buffer(
  snapshot: GIStateSnapshot,
  restirGI: RestirGISnapshot,
): ArrayBuffer {
  const current = serializeGIState(snapshot);
  const base = current.slice(0, current.byteLength - COMPAT_EXTENSION_BYTES);
  const restirBytes =
    20 +
    restirGI.current.byteLength +
    restirGI.previous.byteLength +
    restirGI.spatial.byteLength;
  const buffer = new ArrayBuffer(base.byteLength + restirBytes);
  new Uint8Array(buffer, 0, base.byteLength).set(new Uint8Array(base));
  const view = new DataView(buffer);
  view.setUint32(4, 6, true);
  view.setUint32(52, (view.getUint32(52, true) & 0x7) | 1, true);
  let offset = base.byteLength;
  view.setUint32(offset, restirGI.halfW, true); offset += 4;
  view.setUint32(offset, restirGI.halfH, true); offset += 4;
  view.setUint32(offset, restirGI.strideU32, true); offset += 4;
  view.setUint32(offset, restirGI.current.length, true); offset += 4;
  view.setUint32(offset, 0, true); offset += 4;
  for (const data of [
    restirGI.current,
    restirGI.previous,
    restirGI.spatial,
  ]) {
    new Uint8Array(buffer, offset, data.byteLength).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    );
    offset += data.byteLength;
  }
  return buffer;
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
    expect([back.probeStateW, back.probeStateH]).toEqual([
      s.probeStateW,
      s.probeStateH,
    ]);
    expect(Array.from(back.probeStateData)).toEqual(
      Array.from(s.probeStateData),
    );
  });

  it('produces a single transferable ArrayBuffer sized header + both atlases', () => {
    const s = makeSnapshot();
    const buf = serializeGIState(s);
    expect(buf).toBeInstanceOf(ArrayBuffer);
    expect(buf.byteLength).toBe(
      64 +
      s.irrData.byteLength +
      s.visData.byteLength +
      s.probeStateData.byteLength +
      COMPAT_EXTENSION_BYTES,
    );
  });

  it('rejects a buffer with a bad magic', () => {
    const bogus = new ArrayBuffer(64);
    expect(() => deserializeGIState(bogus)).toThrow(/bad magic/);
  });

  it('rejects a truncated buffer (dims declare more than is present)', () => {
    const s = makeSnapshot();
    const full = serializeGIState(s);
    const basePayloadEnd =
      64 +
      s.irrData.byteLength +
      s.visData.byteLength +
      s.probeStateData.byteLength;
    const truncated = full.slice(0, basePayloadEnd - 16);
    expect(() => deserializeGIState(truncated)).toThrow(/too small/);
  });

  it('rejects non-finite and non-binary persisted probe-state texels', () => {
    const s = makeSnapshot();
    const stateOffset = 64 + s.irrData.byteLength + s.visData.byteLength;

    const nonFinite = serializeGIState(s);
    new DataView(nonFinite).setFloat32(stateOffset, Number.NaN, true);
    expect(() => deserializeGIState(nonFinite)).toThrow(
      /malformed DDGI probe-state/,
    );

    const nonBinary = serializeGIState(s);
    new DataView(nonBinary).setFloat32(stateOffset + 12, 0.5, true);
    expect(() => deserializeGIState(nonBinary)).toThrow(
      /malformed DDGI probe-state/,
    );
  });

  it('rejects malformed live probe state before allocating a snapshot buffer', () => {
    const s = makeSnapshot();
    s.probeStateData[0] = Number.POSITIVE_INFINITY;
    expect(() => serializeGIState(s)).toThrow(/probe-state dimensions\/data/);
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

  it('decodes a legacy v6 compact ReSTIR-GI section for cold migration', () => {
    const snapshot = makeSnapshot();
    const restirGI = makeRestirSection(RESTIR_GI_COMPACT_STRIDE);
    expect(() =>
      serializeGIState({ ...snapshot, restirGI }),
    ).toThrow(/cannot publish retired 20-u32/);
    const back = deserializeGIState(
      makeLegacyCompactV6Buffer(snapshot, restirGI),
    );
    expect(back.restirGI).toBeDefined();
    const r = back.restirGI!;
    expect(r.strideU32).toBe(RESTIR_GI_COMPACT_STRIDE);
    expect(r.current.length).toBe(restirGI.halfW * restirGI.halfH * RESTIR_GI_COMPACT_STRIDE);
    expect(Array.from(r.current)).toEqual(Array.from(restirGI.current));
    expect(Array.from(r.previous)).toEqual(Array.from(restirGI.previous));
    expect(Array.from(r.spatial)).toEqual(Array.from(restirGI.spatial));
  });

  it('sizes the buffer for header + atlases + reservoir sub-block when reservoirs are present', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
    const buf = serializeGIState(s);
    const r = s.restirGI;
    const reservoirBytes = 20 /* sub-header */ + r.current.byteLength + r.previous.byteLength + r.spatial.byteLength;
    expect(buf.byteLength).toBe(
      64 + s.irrData.byteLength + s.visData.byteLength
        + s.probeStateData.byteLength + reservoirBytes
        + COMPAT_EXTENSION_BYTES,
    );
  });

  it('omits the reservoir section when restirGI is absent (DDGI-only payload)', () => {
    const s = makeSnapshot(); // no restirGI
    const buf = serializeGIState(s);
    expect(buf.byteLength).toBe(
      64 +
        s.irrData.byteLength +
        s.visData.byteLength +
        s.probeStateData.byteLength +
        COMPAT_EXTENSION_BYTES,
    );
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

  it('rejects a buffer shorter than the fixed header', () => {
    expect(() => deserializeGIState(new ArrayBuffer(63))).toThrow(
      /smaller than the fixed header/,
    );
  });

  it.each([
    ['dims.x = 0', 8, 0, /dims\.x/],
    ['dims.y = 0', 12, 0, /dims\.y/],
    ['dims.z = 0', 16, 0, /dims\.z/],
    ['irrW = 0', 36, 0, /irrW/],
    ['irrH = 0', 40, 0, /irrH/],
    ['visW = 0', 44, 0, /visW/],
    ['visH = 0', 48, 0, /visH/],
    ['irrW disagrees with dims', 36, 16, /atlas dimensions/],
  ] as const)('rejects corrupted header integer metadata: %s', (_label, offset, value, pattern) => {
    const buf = serializeGIState(makeSnapshot());
    new DataView(buf).setUint32(offset, value, true);
    expect(() => deserializeGIState(buf)).toThrow(pattern);
  });

  it.each([
    ['origin.x NaN', 20, Number.NaN, /origin/],
    ['origin.y +Infinity', 24, Number.POSITIVE_INFINITY, /origin/],
    ['origin.z -Infinity', 28, Number.NEGATIVE_INFINITY, /origin/],
    ['spacing NaN', 32, Number.NaN, /spacing/],
    ['spacing zero', 32, 0, /spacing/],
    ['spacing +Infinity', 32, Number.POSITIVE_INFINITY, /spacing/],
  ] as const)('rejects corrupted float metadata: %s', (_label, offset, value, pattern) => {
    const buf = serializeGIState(makeSnapshot());
    new DataView(buf).setFloat32(offset, value, true);
    expect(() => deserializeGIState(buf)).toThrow(pattern);
  });

  it('rejects unknown section flags and undeclared trailing bytes', () => {
    const valid = serializeGIState(makeSnapshot());
    const unknownFlag = valid.slice(0);
    const flags = new DataView(unknownFlag).getUint32(52, true);
    new DataView(unknownFlag).setUint32(52, flags | 0x8000_0000, true);
    expect(() => deserializeGIState(unknownFlag)).toThrow(/unknown section flags/);

    const trailing = new Uint8Array(valid.byteLength + 4);
    trailing.set(new Uint8Array(valid));
    expect(() => deserializeGIState(trailing.buffer)).toThrow(
      /trailing snapshot bytes|extension header/,
    );
  });

  it.each([0x7c00, 0xfc00, 0x7e00])(
    'rejects a non-finite float16 atlas lane (0x%s)',
    (bits) => {
      const buf = serializeGIState(makeSnapshot());
      new DataView(buf).setUint16(64, bits, true);
      expect(() => deserializeGIState(buf)).toThrow(/non-finite float16/);
    },
  );

  it('round-trips large-world metadata at its declared float32 precision', () => {
    const origin = [
      1_000_000_033,
      -2_000_000_017,
      3_000_000_049,
    ] as const;
    const spacing = 1_000_000.03125;
    const back = deserializeGIState(
      serializeGIState({ ...makeSnapshot(), origin, spacing }),
    );

    expect(back.origin).toEqual(origin.map(Math.fround));
    expect(back.spacing).toBe(Math.fround(spacing));
  });

  it.each([
    ['zero dimension', { dims: { x: 0, y: 4, z: 5 } }],
    ['fractional dimension', { dims: { x: 3.5, y: 4, z: 5 } }],
    ['NaN origin', { origin: [Number.NaN, 0, 0] as const }],
    ['infinite origin', { origin: [0, Number.POSITIVE_INFINITY, 0] as const }],
    ['zero spacing', { spacing: 0 }],
    ['NaN spacing', { spacing: Number.NaN }],
    ['unrepresentable spacing', { spacing: Number.MAX_VALUE }],
  ])('rejects invalid live snapshot metadata before allocation: %s', (_label, overrides) => {
    expect(() =>
      serializeGIState({ ...makeSnapshot(), ...overrides }),
    ).toThrow();
  });

  it('rejects live atlas lengths and non-finite half payloads that disagree with metadata', () => {
    const shortIrr = makeSnapshot();
    expect(() =>
      serializeGIState({
        ...shortIrr,
        irrData: shortIrr.irrData.subarray(1),
      }),
    ).toThrow(/irradiance atlas length/);

    const longVis = makeSnapshot();
    expect(() =>
      serializeGIState({
        ...longVis,
        visData: new Uint16Array(longVis.visData.length + 1),
      }),
    ).toThrow(/visibility atlas length/);

    const nonFinite = makeSnapshot();
    nonFinite.irrData[9] = 0x7e00;
    expect(() => serializeGIState(nonFinite)).toThrow(/finite float16/);
  });

  it('rejects live ReSTIR buffers that do not match their declared layout', () => {
    const restirGI = makeRestirSection();
    expect(() =>
      serializeGIState({
        ...makeSnapshot(),
        restirGI: {
          ...restirGI,
          spatial: restirGI.spatial.subarray(1),
        },
      }),
    ).toThrow(/reservoir buffers do not match/);
  });

  it('rejects an unknown ReSTIR reservoir stride ABI', () => {
    const restirGI = makeRestirSection(RESTIR_GI_GRIS_STRIDE);
    expect(() =>
      serializeGIState({
        ...makeSnapshot(),
        restirGI: {
          ...restirGI,
          strideU32: 30,
          current: new Uint32Array(5 * 7 * 30),
          previous: new Uint32Array(5 * 7 * 30),
          spatial: new Uint32Array(5 * 7 * 30),
        },
      }),
    ).toThrow(/recognized reservoir ABI/);
  });

  it.each(
    (['current', 'previous', 'spatial'] as const).flatMap((bufferName) =>
      [...RESTIR_BASE_FLOAT_LANES, ...RESTIR_GRIS_FLOAT_LANES].map(
        (lane) => [bufferName, lane] as const,
      ),
    ),
  )('rejects non-finite ReSTIR %s float lane %i before serialization', (bufferName, lane) => {
    const restirGI = makeRestirSection();
    restirGI[bufferName][lane] = 0x7fc0_0000;
    expect(() =>
      serializeGIState({ ...makeSnapshot(), restirGI }),
    ).toThrow(/non-finite or invalid logical reservoir/);
  });

  it.each([0x7f80_0000, 0xff80_0000])(
    'rejects ReSTIR infinity payload bits 0x%s during deserialization',
    (bits) => {
      const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
      const buf = serializeGIState(s);
      const restirOffset =
        64 +
        s.irrData.byteLength +
        s.visData.byteLength +
        s.probeStateData.byteLength;
      new DataView(buf).setUint32(restirOffset + 20, bits, true);
      expect(() => deserializeGIState(buf)).toThrow(
        /invalid logical reservoir/,
      );
    },
  );

  it('accepts full-domain u32 counters/IDs but rejects an unknown GRIS sampleKind', () => {
    const restirGI = makeRestirSection();
    for (const data of [
      restirGI.current,
      restirGI.previous,
      restirGI.spatial,
    ]) {
      data[15] = 0xffff_ffff;
      data[19] = 0xffff_ffff;
      data[24] = 2;
      data[25] = 1;
      data[27] = 0xffff_ffff;
    }
    expect(() =>
      serializeGIState({ ...makeSnapshot(), restirGI }),
    ).not.toThrow();

    restirGI.previous[25] = 2;
    expect(() =>
      serializeGIState({ ...makeSnapshot(), restirGI }),
    ).toThrow(/invalid logical reservoir/);
  });

  it('requires zero floor-padding tail words while ignoring them as float records', () => {
    const makeFloorBuffer = (): Uint32Array => new Uint32Array(64);
    const restirGI: RestirGISnapshot = {
      halfW: 1,
      halfH: 1,
      strideU32: RESTIR_GI_GRIS_STRIDE,
      current: makeFloorBuffer(),
      previous: makeFloorBuffer(),
      spatial: makeFloorBuffer(),
    };
    expect(() =>
      serializeGIState({ ...makeSnapshot(), restirGI }),
    ).not.toThrow();

    restirGI.spatial[63] = 0x7fc0_0000;
    expect(() =>
      serializeGIState({ ...makeSnapshot(), restirGI }),
    ).toThrow(/invalid logical reservoir/);
  });

  it('rejects a non-zero ReSTIR reserved sub-header word', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
    const buf = serializeGIState(s);
    const restirOffset =
      64 +
      s.irrData.byteLength +
      s.visData.byteLength +
      s.probeStateData.byteLength;
    new DataView(buf).setUint32(restirOffset + 16, 1, true);
    expect(() => deserializeGIState(buf)).toThrow(/reserved sub-header/);
  });

  it.each([
    ['PPG on v3', 3, 0x2],
    ['DDGI probe state on v3', 3, 0x4],
    ['DDGI probe state on v4', 4, 0x4],
    ['DDGI probe state on v5', 5, 0x4],
  ] as const)('rejects section flags invalid for the declared version: %s', (_label, version, invalidFlag) => {
    const buf = serializeGIState(makeSnapshot());
    const header = new DataView(buf);
    header.setUint32(4, version, true);
    header.setUint32(52, invalidFlag, true);
    expect(() => deserializeGIState(buf)).toThrow(
      /section flags are incompatible/,
    );
  });

  it('rejects a reservoir section truncated below its declared dims', () => {
    const s = { ...makeSnapshot(), restirGI: makeRestirSection() };
    const full = serializeGIState(s);
    // Drop the last reservoir buffer's worth of bytes so the section under-runs.
    const truncated = full.slice(0, full.byteLength - s.restirGI.spatial.byteLength);
    expect(() => deserializeGIState(truncated)).toThrow(/ReSTIR-GI reservoir/);
  });

  // ── v3-v5 backward-compat: no explicit probe state; synthesize active ─────
  it.each([3, 4, 5])(
    'accepts a v%i DDGI-only buffer and synthesizes zero-offset active state',
    (version) => {
    const s = makeSnapshot();
    const current = serializeGIState(s);
    const oldLength = 64 + s.irrData.byteLength + s.visData.byteLength;
    const old = current.slice(0, oldLength);
    const header = new DataView(old);
    header.setUint32(4, version, true);
    header.setUint32(52, 0, true); // no ReSTIR, PPG, or explicit DDGI-state section
    header.setUint32(56, 0, true);
    header.setUint32(60, 0, true);

    const back = deserializeGIState(old);
    expect(back.ppg).toBeUndefined();
    expect(back.dims).toEqual(s.dims);
    expect([back.probeStateW, back.probeStateH]).toEqual([
      s.dims.x,
      s.dims.y * s.dims.z,
    ]);
    for (let index = 0; index < back.probeStateData.length; index += 4) {
      expect(Array.from(back.probeStateData.subarray(index, index + 4))).toEqual([
        0,
        0,
        0,
        1,
      ]);
    }
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
    expect(buf.byteLength).toBe(
      64 + s.irrData.byteLength + s.visData.byteLength
        + s.probeStateData.byteLength + reservoirBytes + ppgBytes
        + COMPAT_EXTENSION_BYTES,
    );
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
    const current = new Uint8Array(serializeGIState(s));
    const atlasEnd = 64 + s.irrData.byteLength + s.visData.byteLength;
    const probeStateEnd = atlasEnd + s.probeStateData.byteLength;
    const v4 = new Uint8Array(
      current.byteLength -
        s.probeStateData.byteLength -
        COMPAT_EXTENSION_BYTES,
    );
    v4.set(current.subarray(0, atlasEnd));
    v4.set(
      current.subarray(
        probeStateEnd,
        current.byteLength - COMPAT_EXTENSION_BYTES,
      ),
      atlasEnd,
    );
    const header = new DataView(v4.buffer);
    header.setUint32(4, 4, true);
    header.setUint32(52, 0x2, true); // v4 PPG only; explicit DDGI state is v6.
    header.setUint32(56, 0, true);
    header.setUint32(60, 0, true);
    const back = deserializeGIState(v4.buffer);
    expect(back.ppg?.maxDTreeNodesPerCell).toBe(341);
  });

  it('rejects a PPG blob truncated below its declared size', () => {
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), ppg };
    const full = serializeGIState(s);
    // Drop enough bytes to truncate the dTreeOffsets blob.
    const truncated = full.slice(
      0,
      full.byteLength -
        COMPAT_EXTENSION_BYTES -
        ppg.dTreeOffsets.byteLength,
    );
    const header = new DataView(truncated);
    header.setUint32(4, 7, true);
    header.setUint32(52, header.getUint32(52, true) & 0x7, true);
    expect(() => deserializeGIState(truncated)).toThrow(/PPG tree data/);
  });

  it('rejects a PPG blob where dTreeOffsets length does not match declared dTreeCount', () => {
    // Build a valid snapshot, then corrupt the dTreeCount field in the sub-header.
    const ppg = makePpgSnapshot();
    const s = { ...makeSnapshot(), ppg };
    const buf = serializeGIState(s);
    // The PPG sub-header starts after the fixed header and all DDGI payloads.
    const irrBytes = s.irrW * s.irrH * 8;
    const visBytes = s.visW * s.visH * 8;
    const ppgSubheaderOffset =
      64 + irrBytes + visBytes + s.probeStateData.byteLength; // no ReSTIR section
    // Field [2] (offset +8 from sub-header start) is dTreeCount.
    new DataView(buf).setUint32(ppgSubheaderOffset + 8, ppg.dTreeOffsets.length + 99, true);
    expect(() => deserializeGIState(buf)).toThrow(/dTreeOffsets .*length/);
  });
});
