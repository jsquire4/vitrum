import { describe, expect, it } from 'vitest';
import { CAMERA_RAYS_WGSL } from '../shaders/cameraRays.wgsl.js';

describe('walkaround camera-ray numeric contract', () => {
  it('equilibrates camera inversion and validates both reciprocal orders', () => {
    expect(CAMERA_RAYS_WGSL).toContain('let scales = vec4f(');
    expect(CAMERA_RAYS_WGSL).toContain(
      'cameraMat4ProductIsIdentity(m, candidate)',
    );
    expect(CAMERA_RAYS_WGSL).toContain(
      'cameraMat4ProductIsIdentity(candidate, m)',
    );
    expect(CAMERA_RAYS_WGSL).toContain(
      'if (!cameraFiniteF32(absoluteTermSum))',
    );
    expect(CAMERA_RAYS_WGSL).not.toContain('abs(det) < 1e-10');
  });

  it('starts at the per-pixel near point and accepts an infinite far plane', () => {
    expect(CAMERA_RAYS_WGSL).toContain('if (farH.w != 0.0)');
    expect(CAMERA_RAYS_WGSL).toContain('ray.origin = nearPoint;');
    expect(CAMERA_RAYS_WGSL).toContain(
      'farH.xyz * nearH.w - nearH.xyz * farH.w',
    );
    expect(CAMERA_RAYS_WGSL).not.toContain(
      'var orientation = sign(nearH.w)',
    );
    expect(CAMERA_RAYS_WGSL).toContain('!(directionScale > 0.0)');
    expect(CAMERA_RAYS_WGSL).not.toContain('far4.xyz  /');
  });
});
