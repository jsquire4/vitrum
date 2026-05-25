// Verifies the `wrapWithIdempotentDispose` proxy forwards optional Engine
// methods to the wrapped engine and respects the disposed-guard contract.
//
// `wrapWithIdempotentDispose` is `@internal`-exported from createEngine so we
// can exercise it without a real GPU device. Functional / end-to-end coverage
// (HybridEngine + PT engine through createEngine) lives in the
// shader-compile-ci smoke + the cornell-box example.

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  EngineCapabilities,
  EngineState,
  FrameInput,
  FrameOutput,
  Scene,
  SceneEnvironment,
} from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../src/createEngine.js';

const NULL_CAPS: EngineCapabilities = {
  supportsIncrementalScene: false,
  supportsAuxBuffers: false,
  accumulates: false,
  maxSamplesPerPixel: Infinity,
  maxBounces: 1,
  supportedAnalyticShapes: new Set(),
  supportedEmitterKinds: new Set(),
  causticStrategy: 'none',
};

/** Minimal Engine impl that records updateEnvironment calls so we can assert
 *  the proxy forwards correctly. */
function makeFakeEngine(opts: { withUpdateEnvironment: boolean }): Engine & {
  updateEnvironmentSpy?: ReturnType<typeof vi.fn>;
  setSizeSpy: ReturnType<typeof vi.fn>;
} {
  const updateEnvironmentSpy = opts.withUpdateEnvironment ? vi.fn() : undefined;
  const setSizeSpy = vi.fn();
  const engine = {
    state: 'ready' as EngineState,
    capabilities: NULL_CAPS,
    setScene(_: Scene): void {},
    renderFrame(_: FrameInput): FrameOutput {
      return { kind: 'rendered', samplesAccumulated: 1, isConverged: false, primaryRadiance: {} };
    },
    reset(): void {},
    pause(): void {},
    resume(): void {},
    // setSize is backend-specific and intentionally not part of the Engine contract.
    setSize: (w: number, h: number) => setSizeSpy(w, h),
    dispose(): void {},
    ...(updateEnvironmentSpy
      ? { updateEnvironment: (env: SceneEnvironment | null) => updateEnvironmentSpy(env) }
      : {}),
  } as Engine & {
    updateEnvironmentSpy?: ReturnType<typeof vi.fn>;
    setSize: (w: number, h: number) => void;
  };
  if (updateEnvironmentSpy) engine.updateEnvironmentSpy = updateEnvironmentSpy;
  return Object.assign(engine, { setSizeSpy });
}

describe('wrapWithIdempotentDispose — updateEnvironment forwarding (A1)', () => {
  it('exposes updateEnvironment on the proxy when the engine implements it', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.updateEnvironment).toBe('function');
  });

  it('forwards updateEnvironment calls to the wrapped engine', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    const env: SceneEnvironment = { kind: 'none' };
    proxy.updateEnvironment!(env);
    expect(engine.updateEnvironmentSpy).toHaveBeenCalledTimes(1);
    expect(engine.updateEnvironmentSpy).toHaveBeenCalledWith(env);
  });

  it('forwards a null environment (transition to no-env)', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.updateEnvironment!(null);
    expect(engine.updateEnvironmentSpy).toHaveBeenCalledTimes(1);
    expect(engine.updateEnvironmentSpy).toHaveBeenCalledWith(null);
  });

  it('omits updateEnvironment when the wrapped engine does not implement it', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: false });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.updateEnvironment).toBeUndefined();
  });

  it('treats updateEnvironment as a no-op after dispose (no throw, no forward)', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();
    // After dispose, the proxy should swallow the call rather than forward it.
    expect(() => proxy.updateEnvironment!({ kind: 'none' })).not.toThrow();
    expect(engine.updateEnvironmentSpy).not.toHaveBeenCalled();
  });

  it('dispose itself is idempotent (multiple calls fire engine.dispose once)', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const disposeSpy = vi.spyOn(engine, 'dispose');
    const postDisposeSpy = vi.fn();
    const proxy = wrapWithIdempotentDispose(engine, postDisposeSpy);
    proxy.dispose();
    proxy.dispose();
    proxy.dispose();
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(postDisposeSpy).toHaveBeenCalledTimes(1);
  });

  it('forwards backend-specific setSize when present', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    const setSize = (proxy as unknown as { setSize?: (w: number, h: number) => void }).setSize;
    expect(typeof setSize).toBe('function');
    setSize!(1280, 720);
    expect(engine.setSizeSpy).toHaveBeenCalledTimes(1);
    expect(engine.setSizeSpy).toHaveBeenCalledWith(1280, 720);
  });

  it('returns a skipped non-converged frame after dispose', () => {
    const engine = makeFakeEngine({ withUpdateEnvironment: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();
    const output = proxy.renderFrame({} as FrameInput);
    expect(output.kind).toBe('skipped');
    expect(output.samplesAccumulated).toBe(0);
    expect(output.isConverged).toBe(false);
  });

  it('omits incremental patch methods when capability is false', () => {
    const updatePrimitiveSpy = vi.fn();
    const updateEmitterSpy = vi.fn();
    const engine = {
      ...makeFakeEngine({ withUpdateEnvironment: true }),
      capabilities: { ...NULL_CAPS, supportsIncrementalScene: false },
      updatePrimitive: updatePrimitiveSpy,
      updateEmitter: updateEmitterSpy,
    } as Engine;
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.updatePrimitive).toBeUndefined();
    expect(proxy.updateEmitter).toBeUndefined();
  });

  it('forwards incremental patch methods when capability is true', () => {
    const updatePrimitiveSpy = vi.fn();
    const updateEmitterSpy = vi.fn();
    const engine = {
      ...makeFakeEngine({ withUpdateEnvironment: true }),
      capabilities: { ...NULL_CAPS, supportsIncrementalScene: true },
      updatePrimitive: updatePrimitiveSpy,
      updateEmitter: updateEmitterSpy,
    } as Engine;
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.updatePrimitive).toBe('function');
    expect(typeof proxy.updateEmitter).toBe('function');
    proxy.updatePrimitive?.('p', {});
    proxy.updateEmitter?.('e', {});
    expect(updatePrimitiveSpy).toHaveBeenCalledTimes(1);
    expect(updateEmitterSpy).toHaveBeenCalledTimes(1);
  });
});
