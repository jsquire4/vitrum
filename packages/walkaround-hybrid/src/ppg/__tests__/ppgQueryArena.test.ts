import { describe, expect, it } from 'vitest';
import {
  PPG_QUERY_ARENA_EPOCH_WORD,
  PPG_QUERY_ARENA_MAGIC,
  PPG_QUERY_ARENA_SCHEMA,
  PPG_QUERY_ARENA_VERSION,
  assertPpgQueryArenaPayloadFits,
  buildPpgQueryArenaHeader,
  createPpgQueryArenaLayout,
  nextPpgQueryArenaEpoch,
} from '../ppgQueryArena.js';

const serialised = {
  sTreeBuf: new Float32Array([1, 2, 3, 4]),
  dTreeBuf: new Float32Array([5, 6, 7, 8, 9]),
  dTreeOffsets: new Uint32Array([0, 4]),
};

describe('packed PPG query arena', () => {
  it('pins aligned non-overlapping ranges and a versioned publication header', () => {
    const layout = createPpgQueryArenaLayout({
      sTreeCapacityBytes: 64,
      dTreeCapacityBytes: 96,
      dTreeOffsetsCapacityBytes: 16,
      maxSpatialCells: 4,
      maxDTreeNodesPerCell: 5,
    });
    expect(layout.sTreeByteOffset % 256).toBe(0);
    expect(layout.dTreeByteOffset % 256).toBe(0);
    expect(layout.dTreeOffsetsByteOffset % 256).toBe(0);
    expect(layout.sTreeByteOffset + layout.sTreeCapacityBytes)
      .toBeLessThanOrEqual(layout.dTreeByteOffset);
    expect(layout.dTreeByteOffset + layout.dTreeCapacityBytes)
      .toBeLessThanOrEqual(layout.dTreeOffsetsByteOffset);

    const header = buildPpgQueryArenaHeader(layout, serialised, 41);
    expect(header[0]).toBe(PPG_QUERY_ARENA_MAGIC);
    expect(header[1]).toBe(PPG_QUERY_ARENA_VERSION);
    expect(header[PPG_QUERY_ARENA_EPOCH_WORD]).toBe(41);
    expect(header[3]).toBe(PPG_QUERY_ARENA_SCHEMA);
    expect(header[4]! * 4).toBe(layout.sTreeByteOffset);
    expect(header[7]! * 4).toBe(layout.dTreeByteOffset);
    expect(header[10]! * 4).toBe(layout.dTreeOffsetsByteOffset);
  });

  it('rejects every over-capacity segment before publication', () => {
    const layout = createPpgQueryArenaLayout({
      sTreeCapacityBytes: 20,
      dTreeCapacityBytes: 20,
      dTreeOffsetsCapacityBytes: 4,
      maxSpatialCells: 1,
      maxDTreeNodesPerCell: 1,
    });
    expect(() => assertPpgQueryArenaPayloadFits(layout, {
      ...serialised,
      sTreeBuf: new Float32Array(6),
    })).toThrow(/sTree/);
    expect(() => assertPpgQueryArenaPayloadFits(layout, {
      ...serialised,
      dTreeBuf: new Float32Array(6),
    })).toThrow(/dTree/);
    expect(() => assertPpgQueryArenaPayloadFits(layout, {
      ...serialised,
      dTreeOffsets: new Uint32Array(2),
    })).toThrow(/offsets/);
  });

  it('wraps epoch zero out of the published domain', () => {
    expect(nextPpgQueryArenaEpoch(0xffff_ffff)).toBe(1);
  });
});
