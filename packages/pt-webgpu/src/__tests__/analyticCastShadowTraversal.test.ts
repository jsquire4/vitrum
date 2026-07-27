import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { MATERIAL_FLOAT_STRIDE } from '../scene/materialPacking.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

describe('analytic primitive castShadow traversal', () => {
  it('packs the analytic material id and castShadow:false lane together', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'analytic',
        id: 'non-shadowing-sphere',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        castShadow: false,
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.4, metallic: 0 },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene);
    const materialId = packed.analyticHeaders[1] ?? -1;
    const shadowDisabledLane = materialId * MATERIAL_FLOAT_STRIDE + 25 * 4 + 3;

    expect(packed.analyticCount).toBe(1);
    expect(materialId).toBe(0);
    expect(packed.materials[shadowDisabledLane]).toBe(1);
  });

  it('applies the shared shadow predicate after analytic closest-candidate traversal', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialShadowCastDisabled(matId: u32) -> bool');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '_ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('!materialShadowCastDisabled(matId)');
  });

  it('preserves the same centralized analytic and mesh shadow predicate in CWBVH mode', () => {
    const cwbvh = composePtWebgpuTraceWgsl(false, { cwbvhClosest: true });
    expect(cwbvh).toContain('_ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);');
    expect(cwbvh).toContain('!materialShadowCastDisabled(matId)');
    expect(cwbvh).toContain('materialAcceptsSidedHit(matId, hit.frontFace)');
  });

  it('keeps lite mesh shadow filtering without advertising unsupported analytics', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn materialShadowCastDisabled(matId: u32) -> bool');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('return materialShadowCastDisabled(triMaterialIds[triIdx]);');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('fn traceAnalyticShapes(');
  });
});
