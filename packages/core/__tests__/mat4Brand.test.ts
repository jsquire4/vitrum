/**
 * Mat4 brand test — verifies D6 contract hygiene (2026-05-17 sweep).
 *
 *  - `asMat4()` accepts length-16 Float32Arrays and rejects every other length.
 *  - The brand is at the type level only: unbranded `Float32Array` cannot be
 *    passed where `Mat4` is required. We assert this via expect-error
 *    annotation, not via runtime check (TypeScript erases brands at runtime).
 */

import { describe, expect, it } from 'vitest';
import { asMat4, type Mat4 } from '../src/scene.js';

describe('asMat4 (D6)', () => {
  it('accepts a length-16 Float32Array and returns it branded', () => {
    const raw = new Float32Array(16);
    const m: Mat4 = asMat4(raw);
    // Identity at runtime — brand is a type-only refinement.
    expect(m).toBe(raw);
    expect(m.length).toBe(16);
  });

  it('throws on length 0', () => {
    expect(() => asMat4(new Float32Array(0))).toThrow(/length 0/);
  });

  it('throws on length 15', () => {
    expect(() => asMat4(new Float32Array(15))).toThrow(/length 15/);
  });

  it('throws on length 17', () => {
    expect(() => asMat4(new Float32Array(17))).toThrow(/length 17/);
  });

  it('throws on length 9 (a common upper-3×3 mistake)', () => {
    expect(() => asMat4(new Float32Array(9))).toThrow(/length 9/);
  });

  it('error message names the contract function', () => {
    let caught: unknown = null;
    try { asMat4(new Float32Array(4)); } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toMatch(/^asMat4:/);
  });

  it('rejects unbranded Float32Array at the type level', () => {
    const raw = new Float32Array(16);
    // @ts-expect-error — unbranded Float32Array must not satisfy Mat4 (D6).
    const m: Mat4 = raw;
    void m;
  });
});
