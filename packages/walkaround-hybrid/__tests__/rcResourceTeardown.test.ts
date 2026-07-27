import { describe, expect, it, vi } from 'vitest';
import { RCSubsystem } from '../src/HybridEngineRC.js';

function hostileBuffer(throws: boolean): GPUBuffer & { destroy: ReturnType<typeof vi.fn> } {
  return {
    destroy: vi.fn(() => {
      if (throws) throw new Error('injected destroy failure');
    }),
  } as unknown as GPUBuffer & { destroy: ReturnType<typeof vi.fn> };
}

describe('RCSubsystem resource teardown', () => {
  it('releases every cascade and later owner when first and middle destroys throw', () => {
    const subsystem = new RCSubsystem({} as GPUDevice);
    const cascades = [hostileBuffer(true), hostileBuffer(false), hostileBuffer(true), hostileBuffer(false)];
    const lights = hostileBuffer(false);
    const dispatcher = {
      dispose: vi.fn(() => { throw new Error('injected dispatcher teardown failure'); }),
    };
    const internal = subsystem as unknown as {
      _cascadeBufs: GPUBuffer[] | null;
      _dispatcher: typeof dispatcher | null;
      _lightsGpuBuf: GPUBuffer | null;
    };
    internal._cascadeBufs = cascades;
    internal._dispatcher = dispatcher;
    internal._lightsGpuBuf = lights;

    expect(() => subsystem.dispose()).not.toThrow();
    for (const cascade of cascades) expect(cascade.destroy).toHaveBeenCalledOnce();
    expect(dispatcher.dispose).toHaveBeenCalledOnce();
    expect(lights.destroy).toHaveBeenCalledOnce();
    expect(internal._cascadeBufs).toBeNull();
    expect(internal._dispatcher).toBeNull();
    expect(internal._lightsGpuBuf).toBeNull();

    subsystem.dispose();
    for (const cascade of cascades) expect(cascade.destroy).toHaveBeenCalledOnce();
    expect(dispatcher.dispose).toHaveBeenCalledOnce();
    expect(lights.destroy).toHaveBeenCalledOnce();
  });
});
