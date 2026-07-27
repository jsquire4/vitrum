import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';

describe('normalDepth output contract', () => {
  it.each([
    ['full', PT_WEBGPU_PATH_TRACE_KERNEL_WGSL],
    ['lite', PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL],
  ])('%s kernel writes packed world normals and the no-hit sentinel', (_name, wgsl) => {
    expect(wgsl).toContain('firstHitNormal * 0.5 + vec3f(0.5)');
    expect(wgsl).toContain('let primaryRayOrigin = ray.origin;');
    expect(wgsl).toContain('firstHitDepth = distance(hitPos, primaryRayOrigin);');
    expect(wgsl).toContain('vec4f(0.5, 1.0, 0.5, 0.0)');
  });
});
