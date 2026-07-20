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
  const gl = {
    getExtension: vi.fn(() => ({ loseContext: vi.fn() })),
  } as unknown as WebGL2RenderingContext;
  const canvas = {
    getContext: vi.fn((kind: string) => (kind === 'webgl2' ? gl : null)),
  } as unknown as HTMLCanvasElement;
  return { canvas, gl };
}

describe('constructPathTracer (pt-webgl2) — V1-6 ownership-key strip', () => {
  it('strips advanced.device so a host-supplied device cannot reach the factory or clobber the owned gl', async () => {
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
      onWarning: (w: unknown) => warnings.push(w),
    } as unknown as CreateEngineOptions;

    await constructPathTracer(opts, scene);

    expect(ptWebgl2Factory).toHaveBeenCalledTimes(1);
    const merged = ptWebgl2Factory.mock.calls[0]![0] as { device: unknown };
    // The engine-owned gl context wins; the host device was stripped.
    expect(merged.device).toBe(gl);
    expect(merged.device).not.toBe(hostDevice);
    // The strip helper emits the ownership-key-ignored warning.
    expect(warnings).toContainEqual(expect.objectContaining({
      code: 'createEngine.advanced-ownership-key-ignored',
    }));
  });
});
