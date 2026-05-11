import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

describe('createPTEngine_WebGPU', () => {
  it('reports requested caustic strategy in capabilities', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'manifold-nee',
    });
    expect(engine.capabilities.causticStrategy).toBe('manifold-nee');
  });

  it('supports photon-map capability reporting path', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'photon-map',
    });
    expect(engine.capabilities.causticStrategy).toBe('photon-map');
  });

  it('transitions state ready → disposed across the lifecycle', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    expect(engine.state).toBe('ready');
    engine.dispose();
    expect(engine.state).toBe('disposed');
    // After dispose, lifecycle methods throw rather than no-op.
    expect(() => engine.pause()).toThrow(/disposed/);
    expect(() => engine.resume()).toThrow(/disposed/);
    expect(() => engine.renderFrame({} as never)).toThrow();
  });

  it('pause / resume toggle state when live', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    engine.pause();
    expect(engine.state).toBe('paused');
    engine.resume();
    expect(engine.state).toBe('ready');
    engine.dispose();
  });
});
