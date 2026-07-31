import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  classifyAreaVectorF32,
  classifyTriangleAreaF32,
  normalizeDirectionF32,
} from '../scene/areaEmitterGeometry.js';
import { packEmitterArrays } from '../scene/emitterPacking.js';
import { PT_WEBGPU_COMMON_WGSL } from '../wgsl/common.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL } from '../wgsl/pathTrace/kernelLite.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from '../wgsl/pathTrace/connectLite.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';

function rectScene(
  uAxis: [number, number, number],
  vAxis: [number, number, number],
): Scene {
  return {
    primitives: [],
    emitters: [{
      kind: 'rect-area',
      id: 'scaled-rect',
      position: [0, 0, 0],
      uAxis,
      vAxis,
      color: [1, 1, 1],
      intensity: 1,
    }],
    environment: { kind: 'none' },
  };
}

describe('pt-webgpu area-emitter Float32 scale contract', () => {
  it('retains tiny finite area whose raw squared-cross guard underflows', () => {
    const measured = classifyAreaVectorF32([1e-18, 0, 0], [0, 1e-18, 0], 4);
    expect(measured.valid).toBe(true);
    if (!measured.valid) return;
    expect(measured.area / 4e-36).toBeCloseTo(1, 5);
    expect(measured.normal).toEqual([0, 0, 1]);

    const packed = packEmitterArrays(rectScene([1e-18, 0, 0], [0, 1e-18, 0]));
    expect(packed.rectAreaLightCount).toBe(1);
    expect(packed.rectAreaLightsData[4]).toBeGreaterThan(0);
    expect(packed.rectAreaLightsData[9]).toBeGreaterThan(0);
  });

  it('retains a huge near-parallel area when raw f32 cross products become inf-inf', () => {
    const base = Math.fround(1e20);
    const next = Math.fround(base * (1 + 1e-6));
    const measured = classifyAreaVectorF32(
      [base, base, 0],
      [base, next, 0],
      4,
    );
    expect(measured.valid).toBe(true);
    if (!measured.valid) return;
    expect(measured.area).toBeGreaterThan(1e33);
    expect(measured.area).toBeLessThan(1e36);
    expect(measured.normal).toEqual([0, 0, 1]);
  });

  it('rejects exact degeneracy and area/Jacobian values that f32 cannot represent', () => {
    expect(classifyAreaVectorF32([1, 2, 3], [2, 4, 6], 4)).toEqual({
      valid: false,
      reason: 'degenerate',
    });
    expect(classifyAreaVectorF32([1e20, 0, 0], [0, 1e20, 0], 4)).toEqual({
      valid: false,
      reason: 'unrepresentable-area',
    });
    expect(classifyAreaVectorF32([1e-20, 0, 0], [0, 1e-20, 0], 4)).toEqual({
      valid: false,
      reason: 'unrepresentable-area',
    });
    expect(() => packEmitterArrays(rectScene([1e20, 0, 0], [0, 1e20, 0])))
      .toThrow(/unrepresentable-area/);
  });

  it('measures f32-published triangle edges and normalizes scale-free directions', () => {
    const triangle = classifyTriangleAreaF32(
      [0, 0, 0],
      [1e-18, 0, 0],
      [0, 1e-18, 0],
    );
    expect(triangle.valid).toBe(true);
    if (triangle.valid) expect(triangle.area / 5e-37).toBeCloseTo(1, 5);
    expect(normalizeDirectionF32([1e300, 0, 0])).toEqual([1, 0, 0]);
    expect(normalizeDirectionF32([1e-300, 0, 0])).toEqual([1, 0, 0]);
  });

  it('wires the common equilibrated measure through every production area-light family', () => {
    expect(PT_WEBGPU_COMMON_WGSL).toContain('fn measureAreaVector(');
    expect(PT_WEBGPU_COMMON_WGSL).toContain('fn solveAreaVectorCoordinates(');
    for (const source of [
      PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
      PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL,
      RESTIR_PT_PRODUCER_WGSL,
      PT_WEBGPU_MEDIUM_NEE_WGSL,
      SPPM_PHOTON_PASS_WGSL,
      PT_WEBGPU_PATH_TRACE_CONNECT_WGSL,
      PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL,
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL,
    ]) {
      expect(source).toContain('measureAreaVector(');
    }
  });
});
