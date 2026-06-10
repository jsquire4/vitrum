// gpuStub.ts — shared WebGPU mock-device support for the scene-upload unit tests.
// (Not a *.test.ts file, so vitest does not collect it as a suite.)
//
// The upload path references the WebGPU constant globals and, since P2, creates a
// material texture_2d_array + sampler. These helpers let the bespoke per-file
// stub devices model that without a real GPU.
import { vi } from 'vitest';

/** Install the WebGPU constant globals the upload path reads. Idempotent. */
export function installGpuConstStubs(): void {
  const g = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUTextureUsage?: Record<string, number>;
    GPUShaderStage?: Record<string, number>;
  };
  if (g.GPUBufferUsage == null) {
    g.GPUBufferUsage = { STORAGE: 1 << 0, COPY_DST: 1 << 1, UNIFORM: 1 << 2, COPY_SRC: 1 << 3 };
  }
  if (g.GPUTextureUsage == null) {
    g.GPUTextureUsage = {
      COPY_SRC: 1 << 0,
      COPY_DST: 1 << 1,
      TEXTURE_BINDING: 1 << 2,
      STORAGE_BINDING: 1 << 3,
      RENDER_ATTACHMENT: 1 << 4,
    };
  }
  if (g.GPUShaderStage == null) {
    g.GPUShaderStage = { VERTEX: 1 << 0, FRAGMENT: 1 << 1, COMPUTE: 1 << 2 };
  }
}

/** Stub GPUDevice texture/sampler methods (P2 material-texture upload), to spread
 *  into a mock device literal next to its buffer stubs. `createTexture` returns an
 *  object with `createView` + `destroy` so the upload's view/dispose paths work. */
export function textureStubMethods() {
  return {
    createTexture: vi.fn((desc?: { label?: string }) => ({
      label: desc?.label ?? '',
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    })),
    createSampler: vi.fn(() => ({})),
  };
}

/**
 * Stub GPUDevice error-surface methods (item 28: EngineError surface).
 * Spread into a mock device literal to satisfy the constructor's
 * `addEventListener('uncapturederror', …)` + `device.lost.then(…)` calls.
 *
 * Returns an object with:
 *   - `addEventListener` / `removeEventListener` spy-functions (capture
 *     listeners so tests can fire fake uncapturederror events)
 *   - `lost` — a never-resolving Promise (device never "loses" in the test)
 *
 * Usage:
 *   const errStubs = deviceErrorStubMethods();
 *   const device = { createCommandEncoder: vi.fn(), ...errStubs } as unknown as GPUDevice;
 *   // To fire a fake error:
 *   const listeners = errStubs.addEventListener.mock.calls
 *     .filter(([t]) => t === 'uncapturederror').map(([, cb]) => cb);
 *   listeners.forEach(cb => cb({ error: { message: 'oops', constructor: { name: 'GPUValidationError' } } }));
 */
export function deviceErrorStubMethods() {
  return {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    // A promise that never resolves — device stays alive for the duration of
    // the test. Tests that need device.lost to settle create their own stub.
    lost: new Promise<never>(() => {}),
  };
}
