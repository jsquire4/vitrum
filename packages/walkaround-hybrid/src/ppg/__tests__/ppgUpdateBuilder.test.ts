/**
 * ppgUpdateBuilder.test.ts — H29 contract tests for the buildPpgUpdateWgsl
 * builder. Verifies that the dTree per-cell stride constant is single-sourced
 * and correctly interpolated for any caller-supplied cap.
 */

import { describe, it, expect } from 'vitest';
import { buildPpgUpdateWgsl, PPG_DEFAULT_SPATIAL_CELLS } from '../ppgUpdate.wgsl.js';

describe('buildPpgUpdateWgsl (H29 single-source dTree stride)', () => {
  it('default build (341) contains 341u in the MAX_DTREE_NODES_PER_CELL declaration', () => {
    const wgsl = buildPpgUpdateWgsl(341);
    expect(wgsl).toContain('341u');
  });

  it('custom build (128) contains 128u and NOT 341u for the stride constant', () => {
    const wgsl = buildPpgUpdateWgsl(128);
    expect(wgsl).toContain('128u');
    // The stride literal in the declaration line must be the custom value.
    expect(wgsl).toMatch(/MAX_DTREE_NODES_PER_CELL\s*:\s*u32\s*=\s*128u/);
  });

  it('default call with no argument produces the same output as explicit 341', () => {
    const defaultBuild = buildPpgUpdateWgsl();
    const explicitBuild = buildPpgUpdateWgsl(341);
    expect(defaultBuild).toBe(explicitBuild);
  });

  it('PPG_DEFAULT_SPATIAL_CELLS is exported and equals 1024', () => {
    expect(PPG_DEFAULT_SPATIAL_CELLS).toBe(1_024);
  });

  it('each distinct cap produces distinct WGSL (not accidentally cached)', () => {
    const a = buildPpgUpdateWgsl(64);
    const b = buildPpgUpdateWgsl(512);
    expect(a).not.toBe(b);
    expect(a).toContain('64u');
    expect(b).toContain('512u');
  });
});
