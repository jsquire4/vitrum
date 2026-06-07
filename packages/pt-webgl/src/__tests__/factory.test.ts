import { describe, it, expect, expectTypeOf, vi } from 'vitest';
import type { Engine } from '@vitrum/core';
import { createPTEngine_WebGL2, type PTEngineWebGL2Surface } from '../index.js';

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

  it('is typed to return Engine & PTEngineWebGL2Surface (backend-typed factory)', () => {
    // Compile-time assertion: the named backend factory narrows the return type
    // to the typed intersection (not the erased `Promise<Engine>`), so a host
    // picking this backend by name gets the backend surface methods typed.
    type Returned = Awaited<ReturnType<typeof createPTEngine_WebGL2>>;
    expectTypeOf<Returned>().toEqualTypeOf<Engine & PTEngineWebGL2Surface>();
    // The surface methods are present on the typed return.
    expectTypeOf<Returned>().toHaveProperty('bakeSkyEquirect');
    expectTypeOf<Returned>().toHaveProperty('getSceneTlasAudit');
    expectTypeOf<Returned>().toHaveProperty('getDenoisedFrame');
    // The intersection is still assignable to the universal Engine contract.
    expectTypeOf<Returned>().toMatchTypeOf<Engine>();
  });
});
