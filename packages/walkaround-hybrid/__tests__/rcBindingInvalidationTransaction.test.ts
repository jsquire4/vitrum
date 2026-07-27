import { describe, expect, it, vi } from 'vitest';
import { RCSubsystem } from '../src/HybridEngineRC.js';

function subsystemWithDispatcher(invalidateBindings: () => void): RCSubsystem {
  const rc = new RCSubsystem({} as GPUDevice);
  (rc as unknown as {
    _dispatcher: { invalidateBindings(): void } | null;
  })._dispatcher = { invalidateBindings };
  return rc;
}

describe('RC binding invalidation transaction', () => {
  it('rejects a partial env pair before reaching the raw dispatcher', () => {
    const rc = new RCSubsystem({} as GPUDevice);
    const dispatchFrameRaw = vi.fn();
    (rc as unknown as { _dispatcher: { dispatchFrameRaw: typeof dispatchFrameRaw } })
      ._dispatcher = { dispatchFrameRaw };

    expect(() => rc.dispatchFrame({
      sunDirection: [0, 1, 0],
      sunColor: [0, 0, 0],
      frameSeed: 1,
      triIntersectEpsilon: 1e-5,
      envTextureView: {} as GPUTextureView,
    })).toThrow(/envTextureView and envSampler must be supplied together/);
    expect(dispatchFrameRaw).not.toHaveBeenCalled();
  });

  it('infers a supplied env pair as directional while preserving explicit placeholder false', () => {
    const rc = new RCSubsystem({} as GPUDevice);
    const rawDispatches: Array<Record<string, unknown>> = [];
    const buffer = {} as GPUBuffer;
    Object.assign(rc as unknown as Record<string, unknown>, {
      _dispatcher: {
        dispatchFrameRaw: (opts: Record<string, unknown>) => { rawDispatches.push(opts); },
      },
      _bvhBuffers: {
        bvhNodesBuf: buffer,
        bvhIndicesBuf: buffer,
        bvhPositionsBuf: buffer,
        bvhNormalsBuf: buffer,
        materialsBuf: buffer,
        triMaterialIdBuf: buffer,
      },
      _cascadeBufs: [buffer],
      _probeOriginWorld: [0, 0, 0],
      _roomSize: [1, 1, 1],
    });
    const envTextureView = {} as GPUTextureView;
    const envSampler = {} as GPUSampler;
    const base = {
      sunDirection: [0, 1, 0] as const,
      sunColor: [0, 0, 0] as const,
      frameSeed: 1,
      triIntersectEpsilon: 1e-5,
      envTextureView,
      envSampler,
      scalarSkyRadiance: [0.5, 1, 2] as const,
    };

    rc.dispatchFrame(base);
    rc.dispatchFrame({ ...base, frameSeed: 2, hasDirectionalEnvironment: false });

    expect(rawDispatches[0]?.hasDirectionalEnvironment).toBe(true);
    expect(rawDispatches[1]?.hasDirectionalEnvironment).toBe(false);
    expect(rawDispatches[1]?.scalarSkyRadiance).toEqual([0.5, 1, 2]);
  });

  it('retains cached handles through commit and rollback, invalidating only on finalize', () => {
    const invalidateBindings = vi.fn();
    const rc = subsystemWithDispatcher(invalidateBindings);

    const rolledBack = rc.prepareBindingInvalidation();
    expect(invalidateBindings).not.toHaveBeenCalled();
    rolledBack.commit();
    expect(invalidateBindings).not.toHaveBeenCalled();
    rolledBack.rollback();
    expect(invalidateBindings).not.toHaveBeenCalled();
    rolledBack.finalize();
    expect(invalidateBindings).not.toHaveBeenCalled();

    const committed = rc.prepareBindingInvalidation();
    committed.commit();
    expect(invalidateBindings).not.toHaveBeenCalled();
    committed.finalize();
    expect(invalidateBindings).toHaveBeenCalledTimes(1);
    committed.finalize();
    expect(invalidateBindings).toHaveBeenCalledTimes(1);
  });

  it('keeps successful transaction retirement nonthrowing', () => {
    const rc = subsystemWithDispatcher(() => {
      throw new Error('host invalidation fault');
    });
    const mutation = rc.prepareBindingInvalidation();
    mutation.commit();
    expect(() => mutation.finalize()).not.toThrow();
  });
});
