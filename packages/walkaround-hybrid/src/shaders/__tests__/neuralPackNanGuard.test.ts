/**
 * Item 12 — neuralPack NaN guard (trust-remediation-plan-2026-06-10 §12).
 *
 * `normalize(vec3f(0))` is undefined behaviour per the WGSL spec and produces
 * NaN on real hardware when the normal buffer holds (0,0,0) at sky pixels,
 * background regions, or un-rendered areas.  The fix replaces the bare
 * `normalize` with a `select`-guarded form that falls back to (0,1,0) when
 * the remapped vector is near-zero.
 *
 * This structural test pins the guard in the WGSL source so a future
 * simplification cannot silently reintroduce the NaN path.
 *
 * The runtime naga compile is the pre-push T1 GPU smoke; this is the
 * CPU-side structural proxy.
 */
import { describe, it, expect } from 'vitest';
import { NEURAL_PACK_WGSL } from '../neuralPack.wgsl.js';

describe('neuralPack — NaN guard for zero-length normals', () => {
  it('does NOT call bare normalize on the remapped normal (would NaN on sky pixels)', () => {
    // Detect the un-guarded pattern: normalize(nd * 2.0 - 1.0) with no length
    // check.  After the fix the remapped vec is in a local variable and guarded
    // with select + a length² threshold before normalize is called.
    const bareNormalize = /normalize\(\s*nd\s*\*\s*2\.0\s*-\s*1\.0\s*\)/;
    expect(NEURAL_PACK_WGSL).not.toMatch(bareNormalize);
  });

  it('uses select to guard against zero-length normals (fallback to (0,1,0))', () => {
    // The guard must use select (WGSL's ternary) with a dot-product length check
    // and a (0,1,0) fallback — the two structural pillars of the fix.
    expect(NEURAL_PACK_WGSL).toContain('select(');
    expect(NEURAL_PACK_WGSL).toContain('0.0, 1.0, 0.0');
    // The dot-product length² threshold replaces the implicit zero-guard.
    expect(NEURAL_PACK_WGSL).toContain('dot(nd_remapped, nd_remapped)');
  });

  it('normalizes the remapped vector after the guard (not before)', () => {
    // normalize must appear INSIDE the select expression, not on the raw input.
    // The pattern: select(normalize(nd_remapped), …) or select(…, normalize(…)).
    expect(NEURAL_PACK_WGSL).toMatch(/select\(\s*normalize\(nd_remapped\)/);
  });
});
