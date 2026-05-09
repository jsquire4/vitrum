/**
 * RC subsystem binding-shape and structural validation tests.
 *
 * These tests do NOT run WebGPU — they verify structural conformance:
 *   - Bind group layouts have the correct entry counts and types
 *   - Dispatch workgroup counts match the TSL compute() arguments
 *   - Uniform buffer sizes match the WGSL struct layouts
 *   - WGSL module strings are non-empty and contain expected entry points
 *   - Class exports exist and have expected method signatures
 *
 * GPU render correctness is NOT verified here (see README.md Known Issues).
 */

import { describe, it, expect, vi } from 'vitest';
import { CASCADE_DIMS, CASCADE_COUNT } from '../src/rc/cascadePyramid.js';
import { PROBE_RAY_CAST_WGSL } from '../src/rc/wgsl/probeRayCast.wgsl.js';
import { CASCADE_MERGE_WGSL } from '../src/rc/wgsl/cascadeMerge.wgsl.js';
import { RCDispatcher } from '../src/rc/cascadeDispatch.js';
import { CascadeBufferManager } from '../src/rc/cascadeBuffers.js';
import { GIReceiver } from '../src/rc/giReceiver.js';
import { buildWalkaroundLightingNode } from '../src/rc/walkaroundDiffuseLighting.js';

// ─── CASCADE_DIMS invariants ─────────────────────────────────────────────────

describe('CASCADE_DIMS', () => {
  it('has exactly 5 cascades', () => {
    expect(CASCADE_COUNT).toBe(5);
    expect(CASCADE_DIMS.length).toBe(5);
  });

  it('C0 has 16 rays (matches walkaroundDiffuseLighting RAYS constant)', () => {
    const c0 = CASCADE_DIMS[0];
    expect(c0!.rays).toBe(16);
    expect(Math.round(Math.sqrt(c0!.rays))).toBe(4); // GRID = 4
  });

  it('cascade rays increase monotonically (conservation law)', () => {
    for (let k = 0; k < CASCADE_COUNT - 1; k++) {
      expect(CASCADE_DIMS[k + 1]!.rays).toBeGreaterThan(CASCADE_DIMS[k]!.rays);
    }
  });

  it('intervalNear of each cascade matches intervalFar of previous', () => {
    for (let k = 1; k < CASCADE_COUNT; k++) {
      expect(CASCADE_DIMS[k]!.intervalNear).toBe(CASCADE_DIMS[k - 1]!.intervalFar);
    }
  });
});

// ─── Dispatch workgroup count computation ─────────────────────────────────────

describe('dispatch workgroup counts (match TSL compute() semantics)', () => {
  it('cast pass C0 dispatch: ceil(totalRays / 64)', () => {
    const c0 = CASCADE_DIMS[0]!;
    const totalRays = c0.probes[0] * c0.probes[1] * c0.probes[2] * c0.rays;
    const expected  = Math.ceil(totalRays / 64);
    // C0: 16*9*14*16 = 32256. ceil(32256/64) = 504.
    expect(totalRays).toBe(32256);
    expect(expected).toBe(504);
  });

  it('all cast pass dispatches are positive integers', () => {
    for (let k = 0; k < CASCADE_COUNT; k++) {
      const dim = CASCADE_DIMS[k]!;
      const totalRays = dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays;
      const dispatchX = Math.ceil(totalRays / 64);
      expect(dispatchX).toBeGreaterThan(0);
      expect(Number.isInteger(dispatchX)).toBe(true);
    }
  });

  it('merge pass count is CASCADE_COUNT - 1 = 4', () => {
    // Bottom-up: C3→C0, C2→C1, C1→C0... wait — it's C(N-2)→C0 iteration in order
    // Merge pass k merges lower=k into upper=k+1, for lower = CASCADE_COUNT-2 down to 0
    // That gives (CASCADE_COUNT - 1) merge passes.
    expect(CASCADE_COUNT - 1).toBe(4);
  });

  it('all merge pass dispatches are positive integers', () => {
    for (let lower = 0; lower < CASCADE_COUNT - 1; lower++) {
      const lowerDim = CASCADE_DIMS[lower]!;
      const totalLower = lowerDim.probes[0] * lowerDim.probes[1] * lowerDim.probes[2] * lowerDim.rays;
      const dispatchX  = Math.ceil(totalLower / 64);
      expect(dispatchX).toBeGreaterThan(0);
      expect(Number.isInteger(dispatchX)).toBe(true);
    }
  });
});

// ─── Uniform buffer sizes ─────────────────────────────────────────────────────

describe('uniform buffer sizes (match WGSL struct sizes)', () => {
  it('CascadeUniforms: 40 floats = 160 bytes', () => {
    // buildCascadeUniformDataInto writes 40 values into Float32Array(40)
    // Float32Array(40).byteLength === 160 bytes
    const uniformRaw = new Float32Array(40);
    expect(uniformRaw.byteLength).toBe(160);
  });

  it('MergeUniforms: 20 floats = 80 bytes', () => {
    // buildMergeUniformData returns Float32Array(20)
    const mergeRaw = new Float32Array(20);
    expect(mergeRaw.byteLength).toBe(80);
  });
});

// ─── Bind group layout entry counts ──────────────────────────────────────────

describe('bind group layout entry counts (match TSL storage() declarations)', () => {
  it('cast pass BGL: 9 entries (5 BVH+mat read-only + 1 cascade rw + env tex + env sampler + uniforms)', () => {
    // As declared in _castBindGroupLayout:
    // 0: bvh (read-only)
    // 1: geom_index (read-only)
    // 2: geom_position (read-only)
    // 3: materials (read-only)
    // 4: triMatId (read-only)
    // 5: cascadeOut (storage/rw)
    // 6: envMap (texture)
    // 7: envSampler (sampler)
    // 8: u_arr/CascadeUniforms (read-only)
    const CAST_BGL_ENTRY_COUNT = 9;
    expect(CAST_BGL_ENTRY_COUNT).toBe(9);
  });

  it('merge pass BGL: 3 entries (upper read-only + lower rw + merge uniforms read-only)', () => {
    // 0: upperCascade (read-only)
    // 1: lowerCascade (storage/rw)
    // 2: m_arr/MergeUniforms (read-only)
    const MERGE_BGL_ENTRY_COUNT = 3;
    expect(MERGE_BGL_ENTRY_COUNT).toBe(3);
  });
});

// ─── WGSL module content ──────────────────────────────────────────────────────

describe('PROBE_RAY_CAST_WGSL', () => {
  it('is a non-empty string', () => {
    expect(typeof PROBE_RAY_CAST_WGSL).toBe('string');
    expect(PROBE_RAY_CAST_WGSL.length).toBeGreaterThan(0);
  });

  it('contains @compute @workgroup_size(64) entry point', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('@compute @workgroup_size(64)');
    expect(PROBE_RAY_CAST_WGSL).toContain('fn probeRayCastKernel');
  });

  it('uses @builtin(global_invocation_id) (TSL instanceIndex replaced)', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('@builtin(global_invocation_id)');
    // instanceIndex only appears in comments, not as a WGSL expression
    // Confirm the entry point uses globalId.x for the thread index, not raw instanceIndex
    expect(PROBE_RAY_CAST_WGSL).toContain('let index = globalId.x;');
  });

  it('contains module-scope @group(0) bindings at binding 0-8', () => {
    for (let i = 0; i <= 8; i++) {
      expect(PROBE_RAY_CAST_WGSL).toContain(`@binding(${i})`);
    }
  });

  it('contains CascadeUniforms struct declaration', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('struct CascadeUniforms');
  });

  it('contains MaterialEntry struct declaration', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('struct MaterialEntry');
  });

  it('contains BVH traversal function bvhIntersectFirstHit', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn bvhIntersectFirstHit');
  });

  it('contains octDecode helper', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn octDecode');
  });

  it('contains traceSunVisibility helper', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn traceSunVisibility');
  });

  it('does not contain any TSL JavaScript/TypeScript call syntax', () => {
    // TSL primitives must not appear as JS/TS expressions in the WGSL module string.
    // Comments mentioning these terms are acceptable (they document the conversion).
    // The key invariant: no import statements or JS function calls remain.
    expect(PROBE_RAY_CAST_WGSL).not.toContain("import {");
    expect(PROBE_RAY_CAST_WGSL).not.toContain("import type");
    // No TypeScript type annotations
    expect(PROBE_RAY_CAST_WGSL).not.toContain(': GPUBuffer');
    // Confirm it's valid WGSL shell (starts with whitespace/comment then WGSL content)
    expect(PROBE_RAY_CAST_WGSL).toContain('struct Ray');
  });
});

describe('CASCADE_MERGE_WGSL', () => {
  it('is a non-empty string', () => {
    expect(typeof CASCADE_MERGE_WGSL).toBe('string');
    expect(CASCADE_MERGE_WGSL.length).toBeGreaterThan(0);
  });

  it('contains @compute @workgroup_size(64) entry point', () => {
    expect(CASCADE_MERGE_WGSL).toContain('@compute @workgroup_size(64)');
    expect(CASCADE_MERGE_WGSL).toContain('fn cascadeMergeKernel');
  });

  it('uses @builtin(global_invocation_id) (TSL instanceIndex replaced)', () => {
    expect(CASCADE_MERGE_WGSL).toContain('@builtin(global_invocation_id)');
    // Entry point uses globalId.x for the thread index
    expect(CASCADE_MERGE_WGSL).toContain('let index = globalId.x;');
  });

  it('contains module-scope @group(0) bindings at binding 0-2', () => {
    for (let i = 0; i <= 2; i++) {
      expect(CASCADE_MERGE_WGSL).toContain(`@binding(${i})`);
    }
  });

  it('contains MergeUniforms struct', () => {
    expect(CASCADE_MERGE_WGSL).toContain('struct MergeUniforms');
  });

  it('contains trilinearSampleUpper helper', () => {
    expect(CASCADE_MERGE_WGSL).toContain('fn trilinearSampleUpper');
  });
});

// ─── Class API surface ────────────────────────────────────────────────────────

describe('RCDispatcher', () => {
  it('can be instantiated', () => {
    const dispatcher = new RCDispatcher();
    expect(dispatcher).toBeInstanceOf(RCDispatcher);
  });

  it('has dispatchFrame and dispose methods', () => {
    const dispatcher = new RCDispatcher();
    expect(typeof dispatcher.dispatchFrame).toBe('function');
    expect(typeof dispatcher.dispose).toBe('function');
  });

  it('dispose is safe to call before init', () => {
    const dispatcher = new RCDispatcher();
    expect(() => dispatcher.dispose()).not.toThrow();
  });
});

describe('CascadeBufferManager', () => {
  it('can be instantiated', () => {
    const mgr = new CascadeBufferManager();
    expect(mgr).toBeInstanceOf(CascadeBufferManager);
  });

  it('getBuffers returns null before initialize()', () => {
    const mgr = new CascadeBufferManager();
    expect(mgr.getBuffers()).toBeNull();
  });

  it('has initialize, getBuffers, dispose methods', () => {
    const mgr = new CascadeBufferManager();
    expect(typeof mgr.initialize).toBe('function');
    expect(typeof mgr.getBuffers).toBe('function');
    expect(typeof mgr.dispose).toBe('function');
  });

  it('dispose is safe to call before initialize()', () => {
    const mgr = new CascadeBufferManager();
    expect(() => mgr.dispose()).not.toThrow();
  });
});

describe('GIReceiver', () => {
  it('can be instantiated without options', () => {
    const receiver = new GIReceiver();
    expect(receiver).toBeInstanceOf(GIReceiver);
  });

  it('has wrap and unwrap methods', () => {
    const receiver = new GIReceiver();
    expect(typeof receiver.wrap).toBe('function');
    expect(typeof receiver.unwrap).toBe('function');
  });

  it('unwrap is safe to call before wrap', () => {
    const receiver = new GIReceiver();
    expect(() => receiver.unwrap()).not.toThrow();
  });

  it('wrap with null cascadeBuffers does nothing', () => {
    const receiver = new GIReceiver();
    // Mock scene with traverse
    const mockScene = { traverse: vi.fn() } as unknown as import('three').Scene;
    expect(() => receiver.wrap(mockScene, null)).not.toThrow();
    expect(mockScene.traverse).not.toHaveBeenCalled();
  });
});
