import { describe, expect, it } from 'vitest';

import {
  NRC_ARENA_ALIGNMENT,
  NRC_INFERENCE_HEADER_FIELD,
  NRC_RUNTIME_HEADER_FIELD,
  buildNrcInferenceArenaHeader,
  buildNrcRuntimeArenaHeader,
  createNrcInferenceArenaLayout,
  createNrcRuntimeArenaLayout,
  validateNrcArenaLayouts,
  type NrcInferenceArenaLayout,
  type NrcRuntimeArenaLayout,
} from '../nrcArena.js';

const PAYLOAD = {
  weightsBytes: 1_028,
  biasesBytes: 260,
  tablesBytes: 4_100,
  levelsBytes: 132,
} as const;

function inference(): NrcInferenceArenaLayout {
  return createNrcInferenceArenaLayout(PAYLOAD);
}

function runtime(): NrcRuntimeArenaLayout {
  return createNrcRuntimeArenaLayout({
    diagnosticsBytes: 20,
    claimsBytes: 64 * 4,
    recordsBytes: 64 * 17 * 4,
  });
}

describe('NRC packed arena validation', () => {
  it('round-trips every shader-visible offset and length through the headers', () => {
    const inf = inference();
    const rt = runtime();
    validateNrcArenaLayouts({
      inference: {
        layout: inf,
        payload: PAYLOAD,
        allocationBytes: inf.byteSize,
        epoch: 7,
        generation: 11,
      },
      runtime: {
        layout: rt,
        allocationBytes: rt.byteSize,
        epoch: 7,
        generation: 11,
        recordCap: 64,
        recordStride: 17,
      },
      limits: {
        maxBufferSize: Math.max(inf.byteSize, rt.byteSize),
        maxStorageBufferBindingSize: Math.max(inf.byteSize, rt.byteSize),
      },
    });

    for (const offset of [
      inf.weightsByteOffset,
      inf.biasesByteOffset,
      inf.tablesByteOffset,
      inf.levelsByteOffset,
      rt.headerByteOffset,
      rt.claimsByteOffset,
      rt.recordsByteOffset,
      inf.byteSize,
      rt.byteSize,
    ]) {
      expect(offset % NRC_ARENA_ALIGNMENT).toBe(0);
    }

    const ih = buildNrcInferenceArenaHeader(inf, PAYLOAD, 7, 11);
    const i = NRC_INFERENCE_HEADER_FIELD;
    expect({
      weights: ih[i.weightsOffset]! * 4,
      weightsLength: ih[i.weightsLength]! * 4,
      biases: ih[i.biasesOffset]! * 4,
      biasesLength: ih[i.biasesLength]! * 4,
      tables: ih[i.tablesOffset]! * 4,
      tablesLength: ih[i.tablesLength]! * 4,
      levels: ih[i.levelsOffset]! * 4,
      levelsLength: ih[i.levelsLength]! * 4,
      arena: ih[i.arenaLength]! * 4,
    }).toEqual({
      weights: inf.weightsByteOffset,
      weightsLength: PAYLOAD.weightsBytes,
      biases: inf.biasesByteOffset,
      biasesLength: PAYLOAD.biasesBytes,
      tables: inf.tablesByteOffset,
      tablesLength: PAYLOAD.tablesBytes,
      levels: inf.levelsByteOffset,
      levelsLength: PAYLOAD.levelsBytes,
      arena: inf.byteSize,
    });

    const rh = buildNrcRuntimeArenaHeader(rt, 7, 11, 64, 17);
    const r = NRC_RUNTIME_HEADER_FIELD;
    expect({
      claims: rh[r.claimsOffset]! * 4,
      claimsLength: rh[r.claimsLength]! * 4,
      records: rh[r.recordsOffset]! * 4,
      recordsLength: rh[r.recordsLength]! * 4,
      diagnostics: rh[r.diagnosticsOffset]! * 4,
      diagnosticsLength: rh[r.diagnosticsLength]! * 4,
      recordCap: rh[r.recordCap],
      recordStride: rh[r.recordStride],
      arena: rh[r.arenaLength]! * 4,
    }).toEqual({
      claims: rt.claimsByteOffset,
      claimsLength: rt.claimsBytes,
      records: rt.recordsByteOffset,
      recordsLength: rt.recordsBytes,
      diagnostics: rt.diagnosticsByteOffset,
      diagnosticsLength: rt.diagnosticsBytes,
      recordCap: 64,
      recordStride: 17,
      arena: rt.byteSize,
    });
  });

  it('rejects inference misalignment, overlap, under-allocation, and adapter overflow', () => {
    const layout = inference();
    expect(() => validateNrcArenaLayouts({
      inference: {
        layout: { ...layout, biasesByteOffset: layout.biasesByteOffset + 4 },
      },
    })).toThrow(/256-byte aligned/);
    expect(() => validateNrcArenaLayouts({
      inference: {
        layout: { ...layout, biasesByteOffset: layout.weightsByteOffset },
      },
    })).toThrow(/overlaps its predecessor/);
    expect(() => validateNrcArenaLayouts({
      inference: { layout, allocationBytes: layout.byteSize - 4 },
    })).toThrow(/arena allocation/);
    expect(() => validateNrcArenaLayouts({
      inference: { layout },
      limits: { maxStorageBufferBindingSize: layout.byteSize - 1 },
    })).toThrow(/storage binding size/);
  });

  it('rejects non-word payloads, unsafe arithmetic, and u32-inexpressible headers', () => {
    const layout = inference();
    expect(() => buildNrcInferenceArenaHeader(
      layout,
      { ...PAYLOAD, weightsBytes: 1_027 },
      1,
      0,
    )).toThrow(/4-byte aligned/);
    expect(() => createNrcInferenceArenaLayout({
      ...PAYLOAD,
      weightsBytes: Number.MAX_SAFE_INTEGER,
    })).toThrow(/safe integer|aligned arena byte count|end/);
    expect(() => validateNrcArenaLayouts({
      inference: {
        layout: {
          ...layout,
          byteSize: (0xffff_ffff + 1) * 4,
        },
      },
    })).toThrow(/exact u32 word/);
    expect(() => buildNrcInferenceArenaHeader(layout, PAYLOAD, 0, 0))
      .toThrow(/epoch must be non-zero/);
    expect(() => buildNrcInferenceArenaHeader(
      layout, PAYLOAD, 1, 0x1_0000_0000,
    )).toThrow(/exact u32/);
  });

  it('rejects malformed runtime partitions and record products before publication', () => {
    const layout = runtime();
    expect(() => validateNrcArenaLayouts({
      runtime: {
        layout: { ...layout, headerByteOffset: layout.headerByteOffset + 256 },
      },
    })).toThrow(/shader constant/);
    expect(() => validateNrcArenaLayouts({
      runtime: {
        layout: { ...layout, diagnosticsBytes: layout.headerByteOffset + 4 },
      },
    })).toThrow(/overlap the publication header/);
    expect(() => validateNrcArenaLayouts({
      runtime: {
        layout: { ...layout, recordsByteOffset: layout.claimsByteOffset },
      },
    })).toThrow(/overlaps the claims region/);
    expect(() => validateNrcArenaLayouts({
      runtime: {
        layout,
        recordCap: 65,
        recordStride: 17,
      },
    })).toThrow(/claims region/);
    expect(() => validateNrcArenaLayouts({
      runtime: {
        layout,
        recordCap: 64,
        recordStride: 18,
      },
    })).toThrow(/records region/);
    expect(() => validateNrcArenaLayouts({
      runtime: {
        layout: {
          ...layout,
          claimsBytes: 400_000_000,
          recordsByteOffset: 400_000_512,
          recordsBytes: 4,
          byteSize: 400_000_768,
        },
        recordCap: 100_000_000,
        recordStride: 100_000_000,
      },
    })).toThrow(/safe integer/);
  });
});
