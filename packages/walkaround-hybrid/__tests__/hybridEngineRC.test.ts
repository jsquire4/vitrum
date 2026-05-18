/**
 * W8 Phase 2 — `HybridEngineOptions.rcEnabled` wiring tests.
 *
 * These pin the Phase 2 contract:
 *   1. `rcEnabled: false` (or omitted) leaves `_rc === null`.
 *   2. `rcEnabled: true` instantiates an `RCSubsystem` and stores it on `_rc`.
 *   3. `dispose()` tears down the RC subsystem when present (no throw).
 *   4. `RCSubsystem` exposes the public surface contracted by the plan doc
 *      (`setScene` / `dispatchFrame` / `getCascadeC0Buffer` / `dispose`).
 *
 * Real GPU dispatch is intentionally NOT exercised here — mock `GPUDevice`
 * doesn't support `createBuffer({mappedAtCreation: true}) → getMappedRange()`
 * which `RCSubsystem._uploadAttribute` requires. End-to-end GPU validation
 * lives in W8 Phase 4's acceptance test (opt-in via `GPU=1`).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { HybridEngine } from '../src/HybridEngine.js';
import { RCSubsystem } from '../src/HybridEngineRC.js';

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(rcEnabled?: boolean): HybridEngine {
  const optsBase = {
    device:                makeMockDevice(),
    width:                 64,
    height:                64,
    threeScene:            new THREE.Scene(),
    primaryLightDir:       [0, -1, 0] as [number, number, number],
    primaryLightIntensity: 1.0,
    skyTint:               [0.5, 0.6, 1.0] as [number, number, number],
    skyIrradiance:         0.8,
  };
  return new HybridEngine(
    rcEnabled === undefined ? optsBase : { ...optsBase, rcEnabled },
  );
}

describe('HybridEngineOptions.rcEnabled — Phase 2 wiring', () => {
  it('leaves _rc null when rcEnabled is omitted', () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;
    expect(e['_rc']).toBeNull();
  });

  it('leaves _rc null when rcEnabled: false', () => {
    const engine = makeEngine(false);
    const e = engine as unknown as Record<string, unknown>;
    expect(e['_rc']).toBeNull();
  });

  it('instantiates RCSubsystem when rcEnabled: true', () => {
    const engine = makeEngine(true);
    const e = engine as unknown as Record<string, unknown>;
    expect(e['_rc']).toBeInstanceOf(RCSubsystem);
  });

  it('dispose() is safe when rcEnabled: true (no exception)', () => {
    const engine = makeEngine(true);
    expect(() => engine.dispose()).not.toThrow();
  });

  it('dispose() clears _rc back to null when rcEnabled: true', () => {
    const engine = makeEngine(true);
    const e = engine as unknown as Record<string, unknown>;
    expect(e['_rc']).toBeInstanceOf(RCSubsystem);
    engine.dispose();
    expect(e['_rc']).toBeNull();
  });
});

describe('RCSubsystem — public surface', () => {
  it('has the methods documented by plan/w8-rc-mis-composition.md', () => {
    const rc = new RCSubsystem(makeMockDevice());
    expect(typeof rc.setScene).toBe('function');
    expect(typeof rc.dispatchFrame).toBe('function');
    expect(typeof rc.getCascadeC0Buffer).toBe('function');
    expect(typeof rc.getCascadeC0Dims).toBe('function');
    expect(typeof rc.getCascadeGeometry).toBe('function');
    expect(typeof rc.dispose).toBe('function');
  });

  it('getCascadeC0Buffer returns null before setScene', () => {
    const rc = new RCSubsystem(makeMockDevice());
    expect(rc.getCascadeC0Buffer()).toBeNull();
  });

  it('getCascadeC0Dims returns null before setScene', () => {
    const rc = new RCSubsystem(makeMockDevice());
    expect(rc.getCascadeC0Dims()).toBeNull();
  });

  it('getCascadeGeometry returns null before setScene', () => {
    const rc = new RCSubsystem(makeMockDevice());
    expect(rc.getCascadeGeometry()).toBeNull();
  });

  it('dispatchFrame is safe before setScene (no-op, no exception)', () => {
    const rc = new RCSubsystem(makeMockDevice());
    expect(() => rc.dispatchFrame({
      sunDirection:        [0, -1, 0],
      sunColor:            [1, 1, 1],
      frameSeed:           0,
      triIntersectEpsilon: 1e-5,
    })).not.toThrow();
  });

  it('dispose is safe before setScene (no-op, no exception)', () => {
    const rc = new RCSubsystem(makeMockDevice());
    expect(() => rc.dispose()).not.toThrow();
  });

  it('dispose is idempotent', () => {
    const rc = new RCSubsystem(makeMockDevice());
    rc.dispose();
    expect(() => rc.dispose()).not.toThrow();
  });
});
