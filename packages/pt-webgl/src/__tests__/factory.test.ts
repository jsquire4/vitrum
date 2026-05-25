import { describe, it, expect, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';

// Mock the absorbed `three-gpu-pathtracer` renderer package. The test only
// exercises input validation that runs before any WebGLPathTracer construction,
// so a no-op stub is sufficient. Other tests that need richer renderer
// behaviour mock it themselves (see capabilities.test.ts).
vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    setScene(): void {}
    setCamera(): void {}
    renderSample(): void {}
    reset(): void {}
    dispose(): void {}
  }
  return { WebGLPathTracer };
});

describe('createPTEngine_WebGL2', () => {
  it('rejects null/invalid device', async () => {
    await expect(createPTEngine_WebGL2({ device: null as never })).rejects.toThrow(TypeError);
    await expect(
      createPTEngine_WebGL2({ device: {} as never }),
    ).rejects.toThrow(TypeError);
  });
});
