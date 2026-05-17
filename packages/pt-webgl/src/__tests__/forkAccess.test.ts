/**
 * forkAccess.test.ts
 *
 * Verifies the {@link ForkAccess} façade — the single point in pt-webgl that
 * navigates the upstream three-gpu-pathtracer fork's private shape. When the
 * fork eventually exposes an official accessor (or renames `_pathTracer`),
 * only `forkAccess.ts` changes; these tests confirm the façade's contract
 * stays intact in both directions.
 */

import { describe, expect, it } from 'vitest';
import {
  ForkAccess,
  makeForkPathTracerStubForTests,
  type MaterialLike,
} from '../forkAccess.js';

describe('ForkAccess.getMaterial', () => {
  it('returns the inner material when the stub has the fork-private shape', () => {
    const material: MaterialLike = { uniforms: { uTest: { value: 42 } } };
    const tracer = makeForkPathTracerStubForTests({ material });

    const result = ForkAccess.getMaterial(tracer);

    expect(result).toBe(material);
    expect(result?.uniforms?.uTest?.value).toBe(42);
  });

  it('returns null when the stub lacks _pathTracer', () => {
    const tracer = { target: { texture: {} } };
    expect(ForkAccess.getMaterial(tracer)).toBeNull();
  });

  it('returns null when _pathTracer is present but has no material', () => {
    const tracer = { _pathTracer: {} };
    expect(ForkAccess.getMaterial(tracer)).toBeNull();
  });

  it('returns null for null / undefined inputs', () => {
    expect(ForkAccess.getMaterial(null)).toBeNull();
    expect(ForkAccess.getMaterial(undefined)).toBeNull();
  });
});

describe('ForkAccess.getRenderTexture', () => {
  it('returns the inner texture when the stub has target.texture set', () => {
    const fakeTexture = { isTex: true };
    const tracer = makeForkPathTracerStubForTests({ renderTexture: fakeTexture });

    expect(ForkAccess.getRenderTexture(tracer)).toBe(fakeTexture);
  });

  it('returns null when the stub lacks target', () => {
    const tracer = { _pathTracer: { material: {} } };
    expect(ForkAccess.getRenderTexture(tracer)).toBeNull();
  });

  it('returns null when target exists but has no texture', () => {
    const tracer = { target: {} };
    expect(ForkAccess.getRenderTexture(tracer)).toBeNull();
  });

  it('returns null for null / undefined inputs', () => {
    expect(ForkAccess.getRenderTexture(null)).toBeNull();
    expect(ForkAccess.getRenderTexture(undefined)).toBeNull();
  });
});

describe('makeForkPathTracerStubForTests', () => {
  it('produces a shape ForkAccess can navigate', () => {
    const material: MaterialLike = { uniforms: { x: { value: 1 } } };
    const renderTexture = { tag: 'tex' };
    const stub = makeForkPathTracerStubForTests({ material, renderTexture });

    expect(ForkAccess.getMaterial(stub)).toBe(material);
    expect(ForkAccess.getRenderTexture(stub)).toBe(renderTexture);
  });

  it('omitted fields surface as null through the accessors', () => {
    const stub = makeForkPathTracerStubForTests();
    expect(ForkAccess.getMaterial(stub)).toBeNull();
    expect(ForkAccess.getRenderTexture(stub)).toBeNull();
  });
});
