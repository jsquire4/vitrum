// idempotentDisposeTable.test.ts — golden characterization of the data-driven
// optional-method proxy table in idempotentDispose.ts.
//
// For EACH of the 11 optional Engine methods this asserts:
//   (a) pre-dispose, the proxy forwards the call (with args) to the backend;
//   (b) post-dispose, the proxy exhibits the EXACT current disposed-behaviour:
//         • noop        → swallowed, never reaches the backend, returns undefined
//                         (updatePrimitive/updateEmitter/addPrimitive/
//                          removePrimitive/updateEnvironment/setSize/updateLighting)
//         • empty-unsub → returns a no-op unsubscribe `() => {}`, never forwards
//                         (onFrame/onProgress)
//         • throw       → throws /disposed/, never forwards
//                         (createInverseSession)
//   plus `debug` (a property, not a callable) is passed through as the live
//   engine.debug object.
//
// These behaviours diverge ON PURPOSE — the table encodes each one as data and
// this test is the regression guard against accidental unification.

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  EngineCapabilities,
  EngineState,
  FrameInput,
  FrameOutput,
} from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../idempotentDispose.js';

// Capabilities that advertise every gate so EVERY optional method is eligible.
function allOnCapabilities(): EngineCapabilities {
  return {
    supportsIncrementalScene: true,
    supportsAddRemovePrimitive: true,
    supportsAuxBuffers: false,
    accumulates: true,
    maxSamplesPerPixel: 1,
    maxBounces: 1,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    supportedPrimitiveKinds: new Set(),
    supportedEnvironmentKinds: new Set(),
    presentationMode: 'offscreen-texture',
    experimentalFeatures: new Set(),
    causticStrategy: 'none',
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    },
  } as unknown as EngineCapabilities;
}

const SKIPPED: FrameOutput = { kind: 'skipped', samplesAccumulated: 0, isConverged: false };

/** A fully-populated engine where every optional method is a recording spy.
 *  The empty-unsub methods return a distinct sentinel unsub so we can tell the
 *  backend's unsub apart from the proxy's post-dispose no-op unsub. */
function makeFullEngine() {
  const spies = {
    updatePrimitive: vi.fn(),
    updateEmitter: vi.fn(),
    addPrimitive: vi.fn(),
    removePrimitive: vi.fn(),
    updateEnvironment: vi.fn(),
    setSize: vi.fn(),
    updateLighting: vi.fn(),
    onFrame: vi.fn(),
    onProgress: vi.fn(),
    createInverseSession: vi.fn(),
  };
  const backendUnsub = () => {};
  const session = { parameterCount: 0, method: 'finite-difference', step: vi.fn(), currentValues: () => [], dispose: vi.fn() };
  const debug = { estimatedGpuMemoryBytes: () => null };
  const caps = allOnCapabilities();
  const engine: Engine = {
    get state(): EngineState { return 'ready'; },
    get capabilities() { return caps; },
    setScene: vi.fn(),
    renderFrame: (_: FrameInput): FrameOutput => ({ ...SKIPPED }),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    updatePrimitive: spies.updatePrimitive,
    updateEmitter: spies.updateEmitter,
    addPrimitive: spies.addPrimitive,
    removePrimitive: spies.removePrimitive,
    updateEnvironment: spies.updateEnvironment,
    setSize: spies.setSize,
    updateLighting: spies.updateLighting,
    onFrame: (cb: Parameters<NonNullable<Engine['onFrame']>>[0]) => { spies.onFrame(cb); return backendUnsub; },
    onProgress: (cb: Parameters<NonNullable<Engine['onProgress']>>[0]) => { spies.onProgress(cb); return backendUnsub; },
    createInverseSession: (o: Parameters<NonNullable<Engine['createInverseSession']>>[0]) => { spies.createInverseSession(o); return session as unknown as ReturnType<NonNullable<Engine['createInverseSession']>>; },
    debug,
  };
  return { engine, spies, backendUnsub, session, debug };
}

describe('idempotentDispose proxy table — pre-dispose forwarding', () => {
  it('forwards all 7 noop-methods with their args', () => {
    const { engine, spies } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    p.updatePrimitive!('a', { transform: undefined } as never);
    p.updateEmitter!('b', {});
    p.addPrimitive!({ id: 'x' } as never);
    p.removePrimitive!('y');
    p.updateEnvironment!({ kind: 'none' } as never);
    p.setSize!(7, 9);
    p.updateLighting!({});
    expect(spies.updatePrimitive).toHaveBeenCalledWith('a', { transform: undefined });
    expect(spies.updateEmitter).toHaveBeenCalledWith('b', {});
    expect(spies.addPrimitive).toHaveBeenCalledWith({ id: 'x' });
    expect(spies.removePrimitive).toHaveBeenCalledWith('y');
    expect(spies.updateEnvironment).toHaveBeenCalledWith({ kind: 'none' });
    expect(spies.setSize).toHaveBeenCalledWith(7, 9);
    expect(spies.updateLighting).toHaveBeenCalledWith({});
  });

  it('forwards onFrame/onProgress and returns the backend unsub pre-dispose', () => {
    const { engine, spies, backendUnsub } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    const cbF = vi.fn(); const cbP = vi.fn();
    const offF = p.onFrame!(cbF);
    const offP = p.onProgress!(cbP);
    expect(spies.onFrame).toHaveBeenCalledWith(cbF);
    expect(spies.onProgress).toHaveBeenCalledWith(cbP);
    expect(offF).toBe(backendUnsub);
    expect(offP).toBe(backendUnsub);
  });

  it('forwards createInverseSession and returns the backend session pre-dispose', () => {
    const { engine, spies, session } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    const s = p.createInverseSession!({} as never);
    expect(spies.createInverseSession).toHaveBeenCalledTimes(1);
    expect(s).toBe(session);
  });

  it('passes the live engine.debug object through', () => {
    const { engine, debug } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    expect(p.debug).toBe(debug);
  });
});

describe('idempotentDispose proxy table — post-dispose behaviour (golden)', () => {
  it('noop methods: swallowed, never reach backend, return undefined', () => {
    const { engine, spies } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    p.dispose();
    expect(p.updatePrimitive!('a', {})).toBeUndefined();
    expect(p.updateEmitter!('b', {})).toBeUndefined();
    expect(p.addPrimitive!({ id: 'x' } as never)).toBeUndefined();
    expect(p.removePrimitive!('y')).toBeUndefined();
    expect(p.updateEnvironment!({ kind: 'none' } as never)).toBeUndefined();
    expect(p.setSize!(1, 2)).toBeUndefined();
    expect(p.updateLighting!({})).toBeUndefined();
    for (const k of ['updatePrimitive', 'updateEmitter', 'addPrimitive', 'removePrimitive', 'updateEnvironment', 'setSize', 'updateLighting'] as const) {
      expect(spies[k]).not.toHaveBeenCalled();
    }
  });

  it('empty-unsub methods: return a working no-op unsub, never forward', () => {
    const { engine, spies, backendUnsub } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    p.dispose();
    const offF = p.onFrame!(vi.fn());
    const offP = p.onProgress!(vi.fn());
    expect(typeof offF).toBe('function');
    expect(typeof offP).toBe('function');
    expect(offF).not.toBe(backendUnsub);
    expect(offP).not.toBe(backendUnsub);
    expect(() => offF()).not.toThrow();
    expect(() => offP()).not.toThrow();
    expect(spies.onFrame).not.toHaveBeenCalled();
    expect(spies.onProgress).not.toHaveBeenCalled();
  });

  it('throw method: createInverseSession throws /disposed/, never forwards', () => {
    const { engine, spies } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    p.dispose();
    expect(() => p.createInverseSession!({} as never)).toThrow(/disposed/);
    expect(spies.createInverseSession).not.toHaveBeenCalled();
  });

  it('renderFrame returns a skipped non-converged frame post-dispose', () => {
    const { engine } = makeFullEngine();
    const p = wrapWithIdempotentDispose(engine, () => {});
    p.dispose();
    const out = p.renderFrame({} as FrameInput);
    expect(out).toEqual({ kind: 'skipped', samplesAccumulated: 0, isConverged: false });
  });
});

describe('idempotentDispose proxy table — getRestirPtResultBuffer disposed returns null (Bug5 fix)', () => {
  it('pre-dispose: forwards to backend and returns its value', () => {
    const fakeBuffer = { kind: 'GPUBuffer' };
    const engine: Engine = {
      ...makeFullEngine().engine,
      capabilities: allOnCapabilities(),
      getRestirPtResultBuffer: vi.fn(() => fakeBuffer),
    };
    const p = wrapWithIdempotentDispose(engine, () => {});
    expect(p.getRestirPtResultBuffer!()).toBe(fakeBuffer);
  });

  it('post-dispose: returns null (not undefined)', () => {
    const getRestirPtResultBuffer = vi.fn(() => ({ kind: 'GPUBuffer' }));
    const engine: Engine = {
      ...makeFullEngine().engine,
      capabilities: allOnCapabilities(),
      getRestirPtResultBuffer,
    };
    const p = wrapWithIdempotentDispose(engine, () => {});
    p.dispose();
    const result = p.getRestirPtResultBuffer!();
    // Must be null, not undefined — the contract type is `unknown | null`.
    expect(result).toBeNull();
    expect(result).not.toBeUndefined();
    // Must not forward to the backend after dispose.
    expect(getRestirPtResultBuffer).not.toHaveBeenCalled();
  });
});

describe('idempotentDispose proxy table — eligibility gating', () => {
  it('addPrimitive/removePrimitive omitted when supportsAddRemovePrimitive is false', () => {
    const { engine } = makeFullEngine();
    const caps = { ...allOnCapabilities(), supportsAddRemovePrimitive: false } as EngineCapabilities;
    Object.defineProperty(engine, 'capabilities', { get: () => caps });
    const p = wrapWithIdempotentDispose(engine, () => {});
    expect(p.addPrimitive).toBeUndefined();
    expect(p.removePrimitive).toBeUndefined();
    // The patch methods stay (gated on incremental-patch facets, not add/remove).
    expect(typeof p.updatePrimitive).toBe('function');
    expect(typeof p.updateEmitter).toBe('function');
  });

  it('omits methods the backend does not implement', () => {
    const { engine } = makeFullEngine();
    delete (engine as Partial<Engine>).createInverseSession;
    delete (engine as Partial<Engine>).updateLighting;
    delete (engine as Partial<Engine>).onProgress;
    const p = wrapWithIdempotentDispose(engine, () => {});
    expect(p.createInverseSession).toBeUndefined();
    expect(p.updateLighting).toBeUndefined();
    expect(p.onProgress).toBeUndefined();
  });
});
