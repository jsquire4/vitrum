import { describe, expect, it } from 'vitest';
import * as PathTracerPkg from 'three-gpu-pathtracer';

describe('three-gpu-pathtracer uniform contract (integration)', () => {
  it('exposes BDPT and spectral bridge uniforms on PhysicalPathTracingMaterial', () => {
    const PhysicalPathTracingMaterial = (PathTracerPkg as unknown as { PhysicalPathTracingMaterial?: new () => unknown })
      .PhysicalPathTracingMaterial;
    expect(typeof PhysicalPathTracingMaterial).toBe('function');
    if (PhysicalPathTracingMaterial == null) return;
    const material = new PhysicalPathTracingMaterial();
    const uniforms = (material as unknown as { uniforms?: Record<string, unknown> }).uniforms ?? {};

    expect(Object.hasOwn(uniforms, 'uBdptEnabled')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uBdptLightPathTex')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uBdptMaxLightBounces')).toBe(true);

    expect(Object.hasOwn(uniforms, 'uXCmfCdf')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uZCmfCdf')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uXCmfIntegral')).toBe(true);
    expect(Object.hasOwn(uniforms, 'uZCmfIntegral')).toBe(true);
  });
});
