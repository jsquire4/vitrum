/**
 * B3b (2026-05-19) — Cornell-tuned `CASCADE_DIMS` is no longer the only
 * shape the RC dispatcher can drive. `RCDispatcher` accepts a per-instance
 * `cascadeDims` constructor argument; `allocateCascades` accepts the same
 * via parameter. Hosts on non-Cornell aspect ratios / scene scales
 * override via `HybridEngineOptions.cascadeDims`.
 *
 * These tests pin:
 *   1. `CASCADE_DIMS` still exists as the Cornell default on the raw root.
 *   2. package-root `allocateCascades(bounds)` matches `allocateCascades(bounds, CASCADE_DIMS)`.
 *   3. A custom dims array drives differently-sized cascade buffers.
 *   4. `RCDispatcher` constructor accepts the override.
 */

import { describe, it, expect } from 'vitest';
import {
  CASCADE_DIMS,
  CASCADE_COUNT,
  RCDispatcher,
  allocateCascades,
  validateCascadeDims,
  type CascadeDim,
} from '../src/index.js';

const TINY_BOUNDS = {
  min: [0, 0, 0] as const,
  max: [1, 1, 1] as const,
};

describe('cascadeDims override (B3b)', () => {
  it('CASCADE_DIMS still ships as a 5-cascade Cornell-tuned default', () => {
    expect(CASCADE_COUNT).toBe(5);
    expect(CASCADE_DIMS.length).toBe(5);
    expect(CASCADE_DIMS[0]!.probes).toEqual([16, 9, 14]);
  });

  it('allocateCascades omitted dims === allocateCascades(CASCADE_DIMS)', () => {
    const defaultAlloc = allocateCascades(TINY_BOUNDS);
    const explicitAlloc = allocateCascades(TINY_BOUNDS, CASCADE_DIMS);
    expect(defaultAlloc.cascades.length).toBe(explicitAlloc.cascades.length);
    for (let i = 0; i < defaultAlloc.cascades.length; i++) {
      expect(defaultAlloc.cascades[i]!.length).toBe(explicitAlloc.cascades[i]!.length);
    }
  });

  it('custom dims array drives proportional cascade buffer sizing', () => {
    // 2-cascade pyramid, smaller probe counts → smaller buffers.
    const custom: CascadeDim[] = [
      { probes: [4, 4, 4] as [number, number, number], rays: 16, intervalNear: 0, intervalFar: 5 },
      { probes: [2, 2, 2] as [number, number, number], rays: 64, intervalNear: 5, intervalFar: 50 },
    ];
    const alloc = allocateCascades(TINY_BOUNDS, custom);
    expect(alloc.cascades.length).toBe(2);
    // Cascade 0: 4×4×4×16 rays × 4 floats = 4096 floats
    expect(alloc.cascades[0]!.length).toBe(4 * 4 * 4 * 16 * 4);
    // Cascade 1: 2×2×2×64 rays × 4 floats = 2048 floats
    expect(alloc.cascades[1]!.length).toBe(2 * 2 * 2 * 64 * 4);
  });

  it('RCDispatcher constructor accepts a custom cascadeDims array', () => {
    const custom: CascadeDim[] = [
      {
        probes: [2, 2, 2] as [number, number, number],
        rays: 16,
        intervalNear: 0,
        intervalFar: 1e9,
      },
    ];
    // Constructed without device init (lazy pipelines) — confirms the
    // constructor doesn't reject the override.
    const dispatcher = new RCDispatcher(custom);
    expect(dispatcher).toBeInstanceOf(RCDispatcher);
    // Default constructor still works.
    const defaultDispatcher = new RCDispatcher();
    expect(defaultDispatcher).toBeInstanceOf(RCDispatcher);
  });

  it('validates cascadeDims before allocation or dispatch setup', () => {
    const validSingle: CascadeDim[] = [
      { probes: [2, 2, 2], rays: 16, intervalNear: 0, intervalFar: 4 },
    ];
    expect(validateCascadeDims(validSingle)).toBe(validSingle);
    expect(() => allocateCascades(TINY_BOUNDS, validSingle)).not.toThrow();

    expect(() => new RCDispatcher([])).toThrow(/at least one cascade/);
    expect(() => new RCDispatcher([
      { probes: [0, 2, 2], rays: 16, intervalNear: 0, intervalFar: 4 },
    ])).toThrow(/probes\[0\]/);
    expect(() => new RCDispatcher([
      { probes: [2, 2, 2], rays: 18, intervalNear: 0, intervalFar: 4 },
    ])).toThrow(/perfect square/);
    expect(() => new RCDispatcher([
      { probes: [2, 2, 2], rays: 16, intervalNear: 0, intervalFar: 4 },
      { probes: [1, 1, 1], rays: 36, intervalNear: 4, intervalFar: 8 },
    ])).toThrow(/double the previous cascade ray-grid width/);
    expect(() => new RCDispatcher([
      { probes: [2, 2, 2], rays: 16, intervalNear: 4, intervalFar: 4 },
    ])).toThrow(/interval/);
  });
});
