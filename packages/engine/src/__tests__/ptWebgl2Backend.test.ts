// ptWebgl2Backend.test.ts — R1 V1-6 regression pin.
//
// pt-webgl2 was the only device-owning backend NOT stripping ownership-critical
// keys (device/canvas/context) from `advanced`, and it spread `advanced` AFTER
// `device: gl`. A host-supplied `advanced.device` therefore clobbered the
// createEngine-owned WebGL2 context, causing a double-dispose / owned-handle
// leak. This pins that `advanced.device` never reaches the pt-webgl2 factory.

import { describe, it, expect, vi } from 'vitest';
import type { Engine } from '@vitrum/core';

const ptWebgl2Factory = vi.hoisted(() => vi.fn());

vi.mock('@vitrum/pt-webgl2', () => ({
  createPTEngine_WebGL2: ptWebgl2Factory,
  validateWebgl2AdvancedOptions: vi.fn(),
}));

import { constructPathTracer } from '../backends/ptWebgl2.js';
import type { CreateEngineOptions } from '../createEngineInternals.js';

const scene = { primitives: [], emitters: [], environment: { kind: 'none' } } as never;

function makeEngine(): Engine {
  return {
    backendId: 'pt-webgl2',
    state: 'ready',
    capabilities: {},
    setScene: vi.fn(),
    renderFrame: vi.fn(),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
  } as unknown as Engine;
}

function makeCanvasWithGl() {
  const loseContext = vi.fn();
  const gl = {
    getExtension: vi.fn(() => ({ loseContext })),
  } as unknown as WebGL2RenderingContext;
  const canvas = {
    getContext: vi.fn((kind: string) => (kind === 'webgl2' ? gl : null)),
  } as unknown as HTMLCanvasElement;
  return { canvas, gl, loseContext };
}

describe('constructPathTracer (pt-webgl2) ownership boundary', () => {
  it('rejects advanced.device before context acquisition or factory invocation', async () => {
    ptWebgl2Factory.mockReset();
    const built = makeEngine();
    ptWebgl2Factory.mockResolvedValue(built);

    const { canvas, gl } = makeCanvasWithGl();
    const hostDevice = { tag: 'host-owned-device' } as unknown as WebGL2RenderingContext;
    const warnings: unknown[] = [];

    const opts = {
      canvas,
      scene,
      // Host smuggles a device (and canvas) through the advanced bag.
      advanced: { device: hostDevice, canvas: {} } as never,
      advancedBackend: 'pt-webgl2',
      onWarning: (w: unknown) => warnings.push(w),
    } as unknown as CreateEngineOptions;

    await expect(constructPathTracer(opts, scene)).rejects.toThrow(
      /ownership-critical key.*device, canvas/i,
    );

    expect(canvas.getContext).not.toHaveBeenCalled();
    expect(ptWebgl2Factory).not.toHaveBeenCalled();
    expect(warnings).toEqual([]);
    expect(gl).not.toBe(hostDevice);
  });

  it('releases engine resources without deliberately losing the canvas context', async () => {
    ptWebgl2Factory.mockReset();
    const built = makeEngine();
    ptWebgl2Factory.mockResolvedValue(built);
    const { canvas, loseContext } = makeCanvasWithGl();

    const wrapped = await constructPathTracer({
      canvas,
      scene,
    }, scene);

    wrapped.dispose();
    wrapped.dispose();

    expect(built.dispose).toHaveBeenCalledTimes(1);
    expect(loseContext).not.toHaveBeenCalled();
  });
});
