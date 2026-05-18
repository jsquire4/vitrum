// W3-D19 — BackendTexture branding tests.
//
// These tests live in @vitrum/engine (which already has vitest + transitive
// access to @vitrum/core, @vitrum/walkaround-hybrid, and @vitrum/pt-webgl)
// rather than @vitrum/core (which has no test runner — adding one just for
// this would be heavier than the change warrants).
//
// The tests cover three things:
//  1. RUNTIME — the brand is zero-cost. `asWebGPUBackendTexture(x)` returns
//     `x` unchanged (referential equality), and the round-trip through
//     `narrowToWebGPUTextureView` recovers the original handle.
//  2. STRUCTURE — the brand is encoded as a `unique symbol`-keyed property
//     in the type system. The compiler treats `BackendTexture<'webgpu'>` and
//     `BackendTexture<'webgl'>` as distinct, non-interchangeable types.
//     The structural encoding is what makes the brand survive type erasure —
//     it doesn't get widened to `unknown` at assignment.
//  3. COMPILE-TIME (in-source `@ts-expect-error` markers) — these are the
//     load-bearing assertions. If the brand ever weakens (e.g. someone makes
//     the brand property optional, or removes it), these markers stop
//     erroring and `tsc --noEmit` flags the test file as broken. That's the
//     guard that catches brand-regression PRs.

import { describe, it, expect } from 'vitest';
import type {
  BackendTexture,
  BackendTextureFormat,
  FrameInput,
  Mat4,
  Vec3,
} from '../../core/src/index.js';
// Relative imports (rather than @vitrum/walkaround-hybrid / @vitrum/pt-webgl)
// so the test resolves the worktree's source rather than the parent repo's
// node_modules symlink. The walkaround-hybrid index pulls in heavy
// HybridEngine deps; we only need the brand constructor exports here.
import {
  asWebGPUBackendTexture,
  asWebGPUBackendTextureFormat,
  narrowToWebGPUTextureView,
  narrowToWebGPUTextureFormat,
} from '../../walkaround-hybrid/src/backendTextureBrand.js';
import {
  asWebGLBackendTexture,
  narrowToWebGLTexture,
} from '../../pt-webgl/src/backendTextureBrand.js';

// ─── Fake handles (we don't need a real WebGPU device to test branding) ──────

// We model a GPUTextureView / GPUTexture / WebGLTexture as an arbitrary
// object reference. The brand is zero-cost; what matters is referential
// equality after the round trip and that the type system accepts/rejects
// the right things.
const fakeView = { __kind: 'GPUTextureView' } as unknown as GPUTextureView;
const fakeFormat = 'bgra8unorm' as GPUTextureFormat;
const fakeWebGLTexture = { __kind: 'WebGLTexture' } as unknown as WebGLTexture;

const identity4: Mat4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);
const origin: Vec3 = [0, 0, 0];

// ────────────────────────────────────────────────────────────────────────────

describe('W3-D19 BackendTexture branding', () => {
  describe('zero-cost round trip', () => {
    it('asWebGPUBackendTexture is referentially identity', () => {
      const branded = asWebGPUBackendTexture(fakeView);
      // The brand is type-only; at runtime the same reference flows through.
      expect(branded as unknown).toBe(fakeView);
    });

    it('narrowToWebGPUTextureView round-trips to the same reference', () => {
      const branded = asWebGPUBackendTexture(fakeView);
      const recovered = narrowToWebGPUTextureView(branded);
      expect(recovered).toBe(fakeView);
    });

    it('asWebGPUBackendTextureFormat is referentially identity', () => {
      const branded = asWebGPUBackendTextureFormat(fakeFormat);
      expect(branded as unknown).toBe(fakeFormat);
      expect(narrowToWebGPUTextureFormat(branded)).toBe(fakeFormat);
    });

    it('asWebGLBackendTexture is referentially identity', () => {
      const branded = asWebGLBackendTexture(fakeWebGLTexture);
      expect(branded as unknown).toBe(fakeWebGLTexture);
      expect(narrowToWebGLTexture(branded)).toBe(fakeWebGLTexture);
    });
  });

  describe('FrameInput accepts branded swap-chain handles', () => {
    it('a properly-branded swapChainView satisfies FrameInput', () => {
      const input: FrameInput = {
        viewMatrix: identity4,
        projMatrix: identity4,
        cameraPosition: origin,
        viewport: { width: 1, height: 1, devicePixelRatio: 1 },
        frameIndex: 0,
        frameSeed: 0,
        swapChainView: asWebGPUBackendTexture(fakeView),
        swapChainFormat: asWebGPUBackendTextureFormat(fakeFormat),
      };
      // Compile-time assertion: input typechecks. Runtime smoke:
      expect(input.swapChainView).toBe(fakeView);
      expect(input.swapChainFormat).toBe(fakeFormat);
    });
  });

  describe('compile-time guards (load-bearing for the brand contract)', () => {
    // These assertions ARE the test for the brand: each `@ts-expect-error`
    // must continue to error during typecheck. If the brand weakens, the
    // `@ts-expect-error` becomes a no-op and tsc itself raises
    // "Unused @ts-expect-error directive" — which fails CI. That's how
    // we know the brand is intact.

    it('a raw GPUTextureView CANNOT be assigned to BackendTexture<"webgpu">', () => {
      // @ts-expect-error — raw GPUTextureView lacks the brand symbol; must
      // go through asWebGPUBackendTexture.
      const _bad: BackendTexture<'webgpu'> = fakeView;
      void _bad;
      // The runtime expectation is trivial — we only care about the
      // ts-expect-error compile-time check above.
      expect(true).toBe(true);
    });

    it('a raw string CANNOT be assigned to BackendTextureFormat<"webgpu">', () => {
      // @ts-expect-error — even though BackendTextureFormat is `string & Brand`,
      // a plain string lacks the brand symbol; the intersection rejects it.
      const _bad: BackendTextureFormat<'webgpu'> = 'bgra8unorm';
      void _bad;
      expect(true).toBe(true);
    });

    it('a WebGL-branded handle CANNOT be assigned to a WebGPU slot', () => {
      const webglBranded = asWebGLBackendTexture(fakeWebTexture());
      // @ts-expect-error — branded backends are nominally distinct.
      const _bad: BackendTexture<'webgpu'> = webglBranded;
      void _bad;
      expect(true).toBe(true);
    });

    it('a WebGPU-branded handle CANNOT be passed to narrowToWebGLTexture', () => {
      const webgpuBranded = asWebGPUBackendTexture(fakeView);
      // @ts-expect-error — narrowToWebGLTexture expects BackendTexture<'webgl'>.
      const _bad = narrowToWebGLTexture(webgpuBranded);
      void _bad;
      expect(true).toBe(true);
    });

    it('FrameInput.swapChainView rejects an unbranded raw view', () => {
      const _bad: FrameInput = {
        viewMatrix: identity4,
        projMatrix: identity4,
        cameraPosition: origin,
        viewport: { width: 1, height: 1, devicePixelRatio: 1 },
        frameIndex: 0,
        frameSeed: 0,
        // @ts-expect-error — swapChainView requires BackendTexture<'webgpu'>.
        swapChainView: fakeView,
      };
      void _bad;
      expect(true).toBe(true);
    });
  });

  describe('brand is structurally encoded (survives erasure)', () => {
    it('the brand property exists in the type but NOT on runtime values', () => {
      // The unique-symbol-keyed brand property is declared with `declare const`
      // and never assigned at runtime. The branded value is structurally
      // identical to the input — Object.keys() reveals no brand marker.
      const branded = asWebGPUBackendTexture(fakeView);
      // Reflective check: the runtime object is unchanged.
      expect(Object.keys(branded as object)).toEqual(Object.keys(fakeView as object));
    });
  });
});

// Helper for the cross-backend rejection test.
function fakeWebTexture(): WebGLTexture {
  return { __kind: 'WebGLTexture' } as unknown as WebGLTexture;
}
