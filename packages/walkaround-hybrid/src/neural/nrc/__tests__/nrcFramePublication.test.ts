import { describe, expect, it, vi } from 'vitest';

import { installWebGPUPolyfills } from '../../../../__tests__/helpers/webgpuPolyfills.js';
import { FramePublicationTransaction } from '../../../pipeline/FramePublication.js';
import { createNrcRuntimeArenaLayout } from '../nrcArena.js';
import { NrcSubsystem } from '../nrcSubsystem.js';

installWebGPUPolyfills();

describe('NrcSubsystem frame publication', () => {
  it('keeps sequence/ticket idle on abort, destroys the candidate, and publishes retry once', () => {
    const candidates: Array<{
      readonly label: string;
      readonly destroy: ReturnType<typeof vi.fn>;
    }> = [];
    const device = {
      createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
        const candidate = {
          label: String(descriptor.label),
          destroy: vi.fn(),
          mapState: 'unmapped',
          unmap: vi.fn(),
        };
        candidates.push(candidate);
        return candidate;
      }),
    } as unknown as GPUDevice;
    const runtimeLayout = createNrcRuntimeArenaLayout({
      diagnosticsBytes: 20,
      claimsBytes: 4,
      recordsBytes: 28,
    });
    const runtimeArena = { label: 'runtime-arena' } as GPUBuffer;
    const subsystem = new NrcSubsystem(device, {}, { recordCap: 1 });
    Object.assign(subsystem, {
      _lifecycleState: 'ready',
      _runtimeLayout: runtimeLayout,
      _runtimeArena: runtimeArena,
      _recordByteSize: 28,
      _readbackByteSize: 48,
      _readbackSequence: 4,
      _readbackState: { kind: 'idle' },
      _generation: 9,
    });
    const copies: Array<readonly [number, number, number]> = [];
    const encoder = {
      copyBufferToBuffer: vi.fn((
        _source: GPUBuffer,
        sourceOffset: number,
        _destination: GPUBuffer,
        destinationOffset: number,
        size: number,
      ) => copies.push([sourceOffset, destinationOffset, size])),
    } as unknown as GPUCommandEncoder;
    const internals = subsystem as unknown as {
      _readbackSequence: number;
      _readbackState: { readonly kind: string; readonly ticket?: { readonly sequence: number } };
    };

    const failed = new FramePublicationTransaction();
    subsystem.recordCopyForReadback(encoder, failed);
    expect(internals._readbackSequence).toBe(4);
    expect(internals._readbackState).toMatchObject({
      kind: 'copy-pending',
      ticket: { sequence: 5 },
    });
    expect(copies).toEqual([
      [runtimeLayout.recordsByteOffset, 0, 28],
      [runtimeLayout.diagnosticsByteOffset, 28, 20],
    ]);
    failed.abort();
    expect(internals._readbackSequence).toBe(4);
    expect(internals._readbackState).toEqual({ kind: 'idle' });
    expect(candidates[0]!.destroy).toHaveBeenCalledOnce();

    copies.length = 0;
    const retry = new FramePublicationTransaction();
    subsystem.recordCopyForReadback(encoder, retry);
    expect(internals._readbackSequence).toBe(4);
    expect(internals._readbackState).toMatchObject({
      kind: 'copy-pending',
      ticket: { sequence: 5 },
    });
    expect(copies).toEqual([
      [runtimeLayout.recordsByteOffset, 0, 28],
      [runtimeLayout.diagnosticsByteOffset, 28, 20],
    ]);
    retry.accept();
    expect(internals._readbackSequence).toBe(5);
    expect(internals._readbackState).toMatchObject({
      kind: 'copy-recorded',
      ticket: { sequence: 5 },
    });
    expect(candidates[1]!.destroy).not.toHaveBeenCalled();

    retry.accept();
    expect(internals._readbackSequence).toBe(5);
  });

  it('reserves a staged ticket immediately and cannot resurrect it after dispose', () => {
    const destroy = vi.fn();
    const device = {
      createBuffer: vi.fn(() => ({
        destroy,
        mapState: 'unmapped',
        unmap: vi.fn(),
      })),
    } as unknown as GPUDevice;
    const runtimeLayout = createNrcRuntimeArenaLayout({
      diagnosticsBytes: 20,
      claimsBytes: 4,
      recordsBytes: 28,
    });
    const subsystem = new NrcSubsystem(device, {}, { recordCap: 1 });
    Object.assign(subsystem, {
      _lifecycleState: 'ready',
      _runtimeLayout: runtimeLayout,
      _runtimeArena: { label: 'runtime-arena' },
      _recordByteSize: 28,
      _readbackByteSize: 48,
      _readbackSequence: 2,
      _readbackState: { kind: 'idle' },
      _generation: 3,
    });
    const encoder = {
      copyBufferToBuffer: vi.fn(),
    } as unknown as GPUCommandEncoder;
    const publication = new FramePublicationTransaction();

    subsystem.recordCopyForReadback(encoder, publication);
    subsystem.recordCopyForReadback(encoder, new FramePublicationTransaction());

    const internals = subsystem as unknown as {
      _readbackSequence: number;
      _readbackState: { readonly kind: string };
      _readbackOverlapSkips: number;
    };
    expect(device.createBuffer).toHaveBeenCalledOnce();
    expect(encoder.copyBufferToBuffer).toHaveBeenCalledTimes(2);
    expect(internals._readbackState.kind).toBe('copy-pending');
    expect(internals._readbackOverlapSkips).toBe(1);

    subsystem.dispose();
    expect(destroy).toHaveBeenCalledOnce();
    publication.accept();
    expect(destroy).toHaveBeenCalledOnce();
    expect(internals._readbackSequence).toBe(2);
    expect(internals._readbackState.kind).toBe('disposed');
  });
});
