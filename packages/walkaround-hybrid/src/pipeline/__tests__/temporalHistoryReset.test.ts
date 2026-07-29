import { describe, expect, it, vi } from 'vitest';
import {
  clearTemporalReservoirHistory,
  sceneMutationRequiresTemporalReset,
} from '../temporalHistoryReset.js';

function buffer(label: string): GPUBuffer {
  return { label, size: 256 } as unknown as GPUBuffer;
}

describe('temporal-history fail-safe reset', () => {
  it.each([
    ['BLAS nodes', { nodes: [{ byteOffset: 0, data: new ArrayBuffer(4) }] }],
    ['positions/skinning', { positions: [{ byteOffset: 0, data: new ArrayBuffer(4) }] }],
    ['learning positions', { learningPositions: [{ byteOffset: 0, data: new ArrayBuffer(4) }] }],
    ['normals', { normals: [{ byteOffset: 0, data: new ArrayBuffer(4) }] }],
    ['TLAS transform', { tlas: { nodes: [], worldToLocal: [], localToWorld: [] } }],
    ['replacement', { replacement: {} }],
    ['material', { material: {} }],
    ['texture atlas', { atlas: {} }],
    ['emitters', { emitters: {} }],
  ])('treats %s mutation as a history boundary without an explicit reset flag', (
    _label,
    payload,
  ) => {
    expect(sceneMutationRequiresTemporalReset({
      ...payload,
      resetAccumulator: false,
    } as never)).toBe(true);
  });

  it('does not invent a reset for an empty mutation transaction', () => {
    expect(sceneMutationRequiresTemporalReset({
      resetAccumulator: false,
    })).toBe(false);
  });

  it('clears current, previous, and spatial reservoirs for both DI and GI', () => {
    const buffers = {
      diCurrent: buffer('di-current'),
      diPrevious: buffer('di-previous'),
      diSpatial: buffer('di-spatial'),
      giCurrent: buffer('gi-current'),
      giPrevious: buffer('gi-previous'),
      giSpatial: buffer('gi-spatial'),
    };
    const clearBuffer = vi.fn();
    clearTemporalReservoirHistory(
      { clearBuffer } as unknown as GPUCommandEncoder,
      {
        restirDI: {
          reservoirCurrentBuffer: buffers.diCurrent,
          reservoirPreviousBuffer: buffers.diPrevious,
          reservoirSpatialBuffer: buffers.diSpatial,
        },
        restirGI: {
          reservoirGiCurrentBuffer: buffers.giCurrent,
          reservoirGiPreviousBuffer: buffers.giPrevious,
          reservoirGiSpatialBuffer: buffers.giSpatial,
        },
      },
    );

    expect(clearBuffer.mock.calls.map(([target]) => target)).toEqual([
      buffers.diCurrent,
      buffers.diPrevious,
      buffers.diSpatial,
      buffers.giCurrent,
      buffers.giPrevious,
      buffers.giSpatial,
    ]);
  });
});
