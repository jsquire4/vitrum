// adjointHarness.test.ts — the V24 GPU-adjoint harness kernel composes the
// path-replay BRDF partials + their primitives into a dispatchable shader. The
// EXECUTED GPU == CPU-oracle A/B runs on real hardware (wsl-gpu
// scripts/adjoint-validate.ts, lavapipe: max relative err ~2e-7 = f32 precision);
// here we pin the host-side packing + that the kernel bundles the real partials.
import { describe, it, expect } from 'vitest';
import {
  ADJOINT_HARNESS_WGSL,
  packAdjointHarnessInput,
  ADJOINT_HARNESS_INPUT_FLOATS,
} from '../inverse/adjointHarness.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from '../wgsl/pathTrace/pathTraceAdjoint.wgsl.js';

describe('adjoint harness (V24 GPU partials A/B)', () => {
  it('packs an input into the 16-float vec4-aligned AdjIn record', () => {
    const r = packAdjointHarnessInput([0.8, 0.2, 0.1], 0.5, 1.0, [0, 0, 1], [0.2, 0.1, 1], [-0.3, 0.2, 1]);
    expect(r).toHaveLength(ADJOINT_HARNESS_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([0.8, 0.2, 0.1, 0.5]);   // baseColor.xyz, roughness
    expect(r.slice(4, 8)).toEqual([0, 0, 1, 1.0]);          // normal.xyz, metallic
    expect(r.slice(8, 12)).toEqual([0.2, 0.1, 1, 0]);       // wo.xyz, pad
    expect(r.slice(12, 16)).toEqual([-0.3, 0.2, 1, 0]);     // wi.xyz, pad
  });

  it('bundles the REAL path-replay adjoint partials + a dispatch entry', () => {
    // The partials under test must be the byte-identical production WGSL.
    expect(ADJOINT_HARNESS_WGSL).toContain(PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL);
    expect(ADJOINT_HARNESS_WGSL).toContain('fn dBrdf_dBaseColor(');
    expect(ADJOINT_HARNESS_WGSL).toContain('fn dBrdf_dRoughness(');
    expect(ADJOINT_HARNESS_WGSL).toContain('@compute @workgroup_size(64)');
    expect(ADJOINT_HARNESS_WGSL).toContain('hOutBC[i] = vec4f(dBrdf_dBaseColor(');
    expect(ADJOINT_HARNESS_WGSL).toContain('hOutR[i]  = vec4f(dBrdf_dRoughness(');
    // gradAccum is declared so the bundled adjointScatter compiles.
    expect(ADJOINT_HARNESS_WGSL).toContain('gradAccum: array<atomic<i32>>');
  });
});
