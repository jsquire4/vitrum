/**
 * Shared test utilities for @vitrum/pt-webgl. Extracted from
 * `capabilities.test.ts` (which previously held the canonical version
 * of the FakeWebGL2RenderingContext stub).
 */
import { vi } from 'vitest';

/** Minimal WebGL2 context stub that satisfies WebGLPathTracer's capability
 *  probes. Returns a finite max-fragment-uniform-vectors so caustic and
 *  spectral feature gates light up; reports a fake renderer string.
 *
 *  File-local — tests consume the class indirectly via {@link
 *  installWebGL2GlobalStub} and {@link makeRendererStub}. */
class FakeWebGL2RenderingContext {
  readonly MAX_FRAGMENT_UNIFORM_VECTORS = 0x8dfd;
  readonly MAX_TEXTURE_SIZE = 0x0d33;
  readonly MAX_RENDERBUFFER_SIZE = 0x84e8;
  readonly RENDERER = 0x1f01;

  getExtension(_name: string): null {
    return null;
  }

  getParameter(param: number): number | string {
    if (param === this.MAX_FRAGMENT_UNIFORM_VECTORS) return 512;
    if (param === this.MAX_TEXTURE_SIZE || param === this.MAX_RENDERBUFFER_SIZE) return 8192;
    if (param === this.RENDERER) return 'Fake WebGL2';
    return 0;
  }
}

/**
 * Construct a renderer stub that satisfies the small surface PTEngineWebGL2
 * consumes: `getContext`, `domElement.addEventListener`, `setSize`. Returns
 * the stub plus a spy on `setSize` so tests can assert layout calls.
 */
export function makeRendererStub(options?: { maxSize?: number }) {
  const setSize = vi.fn();
  return {
    getContext: () => new FakeWebGL2RenderingContext(),
    domElement: { addEventListener: vi.fn() },
    setSize,
    _setSize: setSize,
    _maxSize: options?.maxSize,
  };
}

/** Install the FakeWebGL2RenderingContext class on the global namespace
 *  (PTEngineWebGL2 checks `globalThis.WebGL2RenderingContext` at construct
 *  time). Returns a teardown function that should be invoked from afterAll. */
export function installWebGL2GlobalStub(): () => void {
  const key = 'WebGL2RenderingContext' as const;
  (globalThis as unknown as Record<string, unknown>)[key] = FakeWebGL2RenderingContext;
  return () => {
    delete (globalThis as unknown as Record<string, unknown>)[key];
  };
}
