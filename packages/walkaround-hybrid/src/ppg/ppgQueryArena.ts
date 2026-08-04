import type { SerialisedSTree } from './serialise.js';

export const PPG_QUERY_ARENA_MAGIC = 0x50504741;
// v2 overlays exact represented proposal bucket counts into the query-only
// dTree lanes (interior lane 5 / leaf lane 6). Persistent snapshot bytes keep
// their original serialisation ABI and are transformed immediately before an
// arena upload.
export const PPG_QUERY_ARENA_VERSION = 2;
export const PPG_QUERY_ARENA_SCHEMA = 0x4d5bc8a7;
export const PPG_QUERY_ARENA_HEADER_WORDS = 16;
export const PPG_QUERY_ARENA_EPOCH_WORD = 2;
const BINDING_ALIGNMENT_BYTES = 256;

export interface PpgQueryArenaLayout {
  readonly byteLength: number;
  readonly sTreeByteOffset: number;
  readonly sTreeCapacityBytes: number;
  readonly dTreeByteOffset: number;
  readonly dTreeCapacityBytes: number;
  readonly dTreeOffsetsByteOffset: number;
  readonly dTreeOffsetsCapacityBytes: number;
  readonly maxSpatialCells: number;
  readonly maxDTreeNodesPerCell: number;
}

function alignBytes(value: number): number {
  return Math.ceil(value / BINDING_ALIGNMENT_BYTES) * BINDING_ALIGNMENT_BYTES;
}

export function createPpgQueryArenaLayout(input: {
  sTreeCapacityBytes: number;
  dTreeCapacityBytes: number;
  dTreeOffsetsCapacityBytes: number;
  maxSpatialCells: number;
  maxDTreeNodesPerCell: number;
}): PpgQueryArenaLayout {
  const sTreeByteOffset = BINDING_ALIGNMENT_BYTES;
  const dTreeByteOffset = alignBytes(sTreeByteOffset + input.sTreeCapacityBytes);
  const dTreeOffsetsByteOffset = alignBytes(dTreeByteOffset + input.dTreeCapacityBytes);
  return {
    byteLength: alignBytes(dTreeOffsetsByteOffset + input.dTreeOffsetsCapacityBytes),
    sTreeByteOffset,
    sTreeCapacityBytes: input.sTreeCapacityBytes,
    dTreeByteOffset,
    dTreeCapacityBytes: input.dTreeCapacityBytes,
    dTreeOffsetsByteOffset,
    dTreeOffsetsCapacityBytes: input.dTreeOffsetsCapacityBytes,
    maxSpatialCells: input.maxSpatialCells,
    maxDTreeNodesPerCell: input.maxDTreeNodesPerCell,
  };
}

export function nextPpgQueryArenaEpoch(epoch: number): number {
  const next = (epoch + 1) >>> 0;
  return next === 0 ? 1 : next;
}

export function assertPpgQueryArenaPayloadFits(
  layout: PpgQueryArenaLayout,
  data: Pick<SerialisedSTree, 'sTreeBuf' | 'dTreeBuf' | 'dTreeOffsets'>,
): void {
  if (data.sTreeBuf.byteLength > layout.sTreeCapacityBytes) {
    throw new RangeError('[PPG] serialised sTree exceeds packed query-arena capacity.');
  }
  if (data.dTreeBuf.byteLength > layout.dTreeCapacityBytes) {
    throw new RangeError('[PPG] serialised dTree exceeds packed query-arena capacity.');
  }
  if (data.dTreeOffsets.byteLength > layout.dTreeOffsetsCapacityBytes) {
    throw new RangeError('[PPG] serialised dTree offsets exceed packed query-arena capacity.');
  }
}

export function buildPpgQueryArenaHeader(
  layout: PpgQueryArenaLayout,
  data: Pick<SerialisedSTree, 'sTreeBuf' | 'dTreeBuf' | 'dTreeOffsets'>,
  epoch: number,
): Uint32Array<ArrayBuffer> {
  assertPpgQueryArenaPayloadFits(layout, data);
  if (!Number.isInteger(epoch) || epoch <= 0 || epoch > 0xffff_ffff) {
    throw new RangeError('[PPG] query-arena epoch must be a nonzero u32.');
  }
  const header = new Uint32Array(PPG_QUERY_ARENA_HEADER_WORDS);
  header[0] = PPG_QUERY_ARENA_MAGIC;
  header[1] = PPG_QUERY_ARENA_VERSION;
  header[PPG_QUERY_ARENA_EPOCH_WORD] = epoch >>> 0;
  header[3] = PPG_QUERY_ARENA_SCHEMA;
  header[4] = layout.sTreeByteOffset / 4;
  header[5] = data.sTreeBuf.length;
  header[6] = layout.sTreeCapacityBytes / 4;
  header[7] = layout.dTreeByteOffset / 4;
  header[8] = data.dTreeBuf.length;
  header[9] = layout.dTreeCapacityBytes / 4;
  header[10] = layout.dTreeOffsetsByteOffset / 4;
  header[11] = data.dTreeOffsets.length;
  header[12] = layout.dTreeOffsetsCapacityBytes / 4;
  header[13] = layout.maxSpatialCells;
  header[14] = layout.maxDTreeNodesPerCell;
  header[15] = layout.byteLength / 4;
  return header;
}
