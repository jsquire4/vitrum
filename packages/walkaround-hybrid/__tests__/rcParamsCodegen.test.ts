import { describe, expect, it } from 'vitest';
import { RC_PARAMS_BYTE_SIZE, RCParamsOffset } from '../src/rc/rcParamsLayout.generated.js';
import { packRCParams } from '../src/HybridEngineRC.js';

/**
 * Item 14 — rcParamsLayout codegen is the single source of truth.
 *
 * `packRCParams` (HybridEngineRC.ts) and `DDGIBindingState` now import
 * `RC_PARAMS_BYTE_SIZE` and `RCParamsOffset` from the generated file rather
 * than hardcoding `new ArrayBuffer(64)` and manual float/uint word indices.
 * These tests pin the codegen values against the WGSL struct layout so any
 * future generator change that shifts a field trips here before reaching the
 * GPU.
 */
describe('RCParamsOffset codegen alignment', () => {
  it('matches packRCParams field byte offsets', () => {
    expect(RCParamsOffset.probeOriginWorld).toBe(0);
    expect(RCParamsOffset.rcWeight).toBe(12);
    expect(RCParamsOffset.roomSize).toBe(16);
    expect(RCParamsOffset.enabled).toBe(28);
    expect(RCParamsOffset.probeCount).toBe(32);
    expect(RCParamsOffset.raysPerProbe).toBe(44);
    expect(RCParamsOffset.rayGridSize).toBe(48);
    expect(RCParamsOffset.rcWeight / 4).toBe(3);
  });

  it('packRCParams writes at generated offsets', () => {
    const buf = packRCParams([1, 2, 3], [4, 5, 6], [7, 8, 9], 64, 0.25, true);
    const view = new DataView(buf);
    expect(view.getFloat32(RCParamsOffset.probeOriginWorld, true)).toBe(1);
    expect(view.getFloat32(RCParamsOffset.rcWeight, true)).toBe(0.25);
    expect(view.getFloat32(RCParamsOffset.roomSize, true)).toBe(4);
    expect(view.getUint32(RCParamsOffset.enabled, true)).toBe(1);
    expect(view.getUint32(RCParamsOffset.probeCount, true)).toBe(7);
    expect(view.getUint32(RCParamsOffset.raysPerProbe, true)).toBe(64);
    expect(view.getUint32(RCParamsOffset.rayGridSize, true)).toBe(8);
  });

  it('packRCParams output byteLength equals RC_PARAMS_BYTE_SIZE (codegen is the single source of truth for buffer allocation)', () => {
    // Production code now allocates `new ArrayBuffer(RC_PARAMS_BYTE_SIZE)` via
    // the import — this test pins that the generated constant matches the actual
    // struct wire size so DDGIBindingState's buffer allocation is always in sync.
    const buf = packRCParams([0, 0, 0], [1, 1, 1], [1, 1, 1], 4, 1.0, false);
    expect(buf.byteLength).toBe(RC_PARAMS_BYTE_SIZE);
    expect(RC_PARAMS_BYTE_SIZE).toBe(64);
  });

  it('enabled=false writes 0u at the enabled offset (RC short-circuit)', () => {
    const buf = packRCParams([0, 0, 0], [1, 1, 1], [1, 1, 1], 4, 1.0, false);
    const view = new DataView(buf);
    expect(view.getUint32(RCParamsOffset.enabled, true)).toBe(0);
  });
});
