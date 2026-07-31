import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_COMMON_WGSL } from '../wgsl/common.wgsl.js';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL } from '../wgsl/pathTrace/kernelCore.wgsl.js';

describe('pt-webgpu camera-ray WGSL contract', () => {
  it('uses one infinity-safe matrix-only unprojection path for the main tracer', () => {
    expect(PT_WEBGPU_COMMON_WGSL).toContain(
      'fn unproject_ray_common(invViewProjection: mat4x4f, ndc: vec2f) -> Ray',
    );
    expect(PT_WEBGPU_COMMON_WGSL).toContain('if (farH.w != 0.0)');
    expect(PT_WEBGPU_COMMON_WGSL).toContain('ray.origin = nearPoint.xyz;');
    expect(PT_WEBGPU_COMMON_WGSL).toContain(
      'farH.xyz * nearH.w - nearH.xyz * farH.w',
    );
    expect(PT_WEBGPU_COMMON_WGSL).not.toContain(
      'var orientation = sign(nearH.w)',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL).toContain(
      'return unproject_ray_common(params.invViewProj, ndc);',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL).not.toContain(
      'ray.origin = params.cameraPos.xyz',
    );
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL).not.toContain(
      'far4.xyz / far4.w',
    );
  });

  it('keeps exact path replay on the same orthographic/infinite-far semantics', () => {
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('if (farH.w != 0.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('ray.origin = nearPoint;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'farH.xyz * nearH.w - nearH.xyz * farH.w',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).not.toContain(
      'var orientation = sign(nearH.w)',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).not.toContain(
      'ray.origin = params.cameraPos.xyz',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).not.toContain('far4.xyz / far4.w');
  });
});
