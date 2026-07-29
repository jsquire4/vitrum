import { describe, expect, it } from 'vitest';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  bdptDirectionalConePdf,
  bdptDirectionalSourceDirectionWeight,
  bdptEmitterCount,
  distantDirectEmitterCount,
  distantDirectEmitterPower,
  bdptEmitterRejectionThreshold,
  bdptPickEmitterFlat,
  distantDirectSelectionPdf,
  sampleBdptBounce0Cpu,
} from '../bdpt/bdptEmitterPickCpu.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';

function stubScene(partial: Partial<UploadedSceneBuffers>): UploadedSceneBuffers {
  return {
    directionalLightCount: 0,
    directionalLightsData: new Float32Array(0),
    sceneCenter: [0, 0, 0],
    sceneRadius: 1,
    pointLightCount: 0,
    spotLightCount: 0,
    rectAreaLightCount: 0,
    meshAreaLightCount: 0,
    pointLightsData: new Float32Array(0),
    spotLightsData: new Float32Array(0),
    rectAreaLightsData: new Float32Array(0),
    meshAreaLightsData: new Float32Array(0),
    environmentTint: [1, 1, 1],
    environmentSunDirection: [0, 1, 0],
    environmentSunStrength: 0,
    environmentHdriIntensity: 1,
    environmentHdriRotationY: 0,
    environmentMapWidth: 0,
    environmentMapHeight: 0,
    hasEnvironmentMap: false,
    environmentMapTexels: new Float32Array(0),
    environmentMapCdf: new Float32Array(0),
    ...partial,
  } as unknown as UploadedSceneBuffers;
}

function mixedScene(): UploadedSceneBuffers {
  const spot = new Float32Array(16);
  spot.set([7, 8, 9, 0, 0, -1, 0, 0.5, 10, 11, 12, 0.8], 0);
  const rect = new Float32Array(16);
  rect.set([0, 5, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 13, 14, 15, 0], 0);
  const mesh = new Float32Array(16);
  mesh.set([0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 16, 17, 18, 0], 0);
  return stubScene({
    directionalLightCount: 1,
    directionalLightsData: new Float32Array([0, 1, 0, 0, 0.9, 0.8, 0.7, 0]),
    pointLightCount: 1,
    pointLightsData: new Float32Array([1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0]),
    spotLightCount: 1,
    spotLightsData: spot,
    rectAreaLightCount: 1,
    rectAreaLightsData: rect,
    meshAreaLightCount: 1,
    meshAreaLightsData: mesh,
    hasEnvironmentMap: true,
    environmentMapWidth: 1,
    environmentMapHeight: 1,
    environmentMapCdf: new Float32Array([0, 1]),
    environmentMapTexels: new Float32Array([2, 2, 2, 1 / (4 * Math.PI)]),
    sceneRadius: 5,
  });
}

describe('invocation-local BDPT source selection', () => {
  it('ignores the legacy procedural-sun lane without a baked environment map', () => {
    const scene = stubScene({ environmentSunStrength: 1e-12, sceneRadius: 2 });
    expect(bdptEmitterCount(scene)).toBe(0);
    expect(distantDirectEmitterCount(scene)).toBe(0);
    expect(distantDirectEmitterPower(scene, 0)).toBe(0);
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).not.toContain('environmentSun.w');
  });

  it('enumerates every finite and infinite source exactly once', () => {
    const scene = mixedScene();
    expect(bdptEmitterCount(scene)).toBe(6);
    expect([0, 1, 2, 3, 4, 5].map((flat) =>
      sampleBdptBounce0Cpu(scene, flat, 0.37, 0.71)?.lvMatId,
    )).toEqual([-8, -4, -5, -2, -2, -9]);
    for (let flat = 0; flat < 6; flat += 1) {
      expect(sampleBdptBounce0Cpu(scene, flat, 0.37, 0.71)?.selectionPdf)
        .toBeCloseTo(1 / 6, 15);
    }
  });

  it('uses rejection sampling so full-u32 modulo selection is exactly uniform', () => {
    expect(bdptEmitterRejectionThreshold(3)).toBe(1);
    expect(bdptPickEmitterFlat(0, 3)).toBeNull();
    const counts = [0, 0, 0];
    for (let word = 1; word <= 300; word += 1) {
      const selected = bdptPickEmitterFlat(word, 3);
      expect(selected).not.toBeNull();
      counts[selected!] = (counts[selected!] ?? 0) + 1;
    }
    expect(counts).toEqual([100, 100, 100]);
    expect(bdptEmitterRejectionThreshold(0xffffffff)).toBe(1);
    expect(() => bdptEmitterRejectionThreshold(0)).toThrow(/1\.\.4294967295/);
  });

  it('launches hard and soft directional roots with exact endpoint measures', () => {
    const makeDirectional = (diameter: number) => stubScene({
      directionalLightCount: 1,
      directionalLightsData: new Float32Array([0, 1, 0, diameter, 3, 6, 9, 0]),
      sceneCenter: [2, 3, 4],
      sceneRadius: 5,
    });
    const hard = sampleBdptBounce0Cpu(makeDirectional(0), 0, 0.2, 0.7)!;
    const soft = sampleBdptBounce0Cpu(makeDirectional(0.25), 0, 0.2, 0.7)!;
    expect(bdptEmitterCount(makeDirectional(0))).toBe(1);
    expect(hard.directionIsDelta).toBe(true);
    expect(hard.neePdf).toBe(1);
    expect(soft.directionIsDelta).toBe(false);
    expect(soft.directionPdf).toBeCloseTo(bdptDirectionalConePdf(0.25), 13);
    expect(soft.neePdf).toBeCloseTo(soft.directionPdf!, 13);
    expect(bdptDirectionalConePdf(0)).toBe(1);
    expect(bdptDirectionalSourceDirectionWeight(0.25)).toBeCloseTo(
      bdptDirectionalConePdf(0.25), 13,
    );
  });

  it('launches HDRI roots and preserves the authored direction density', () => {
    const scene = stubScene({
      sceneRadius: 2,
      hasEnvironmentMap: true,
      environmentMapWidth: 1,
      environmentMapHeight: 1,
      environmentMapCdf: new Float32Array([0, 1]),
      environmentMapTexels: new Float32Array([2, 4, 6, 0.125]),
      environmentHdriIntensity: 3,
    });
    const sample = sampleBdptBounce0Cpu(scene, 0, 0.4, 0.7)!;
    expect(bdptEmitterCount(scene)).toBe(1);
    expect(sample.directionPdf).toBeCloseTo(0.125, 15);
    expect(sample.neePdf).toBeCloseTo(0.125, 15);
    expect(sample.directionIsDelta).toBe(false);
  });

  it('keeps finite point and area endpoint measures exact', () => {
    const point = stubScene({
      pointLightCount: 1,
      pointLightsData: new Float32Array([1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0]),
    });
    const pointSample = sampleBdptBounce0Cpu(point, 0, 0.3, 0.7)!;
    expect(pointSample.pdfJoint).toBe(1);
    expect(pointSample.positionPdf).toBe(1);
    expect(pointSample.emitRad).toEqual([4, 5, 6]);

    const rectData = new Float32Array(16);
    rectData.set([0, 5, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 2, 4, 6, 0]);
    const rect = stubScene({ rectAreaLightCount: 1, rectAreaLightsData: rectData });
    const rectSample = sampleBdptBounce0Cpu(rect, 0, 0.3, 0.7)!;
    expect(rectSample.positionPdf).toBeCloseTo(1 / 24, 14);
    expect(rectSample.pdfJoint).toBeCloseTo(1 / 24, 14);
  });

  it('preserves the signed concentric-disc quadrant for negative wedge coordinates', () => {
    const discData = new Float32Array(16);
    discData.set([
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
      1, 1, 1, 1,
    ]);
    const disc = stubScene({
      rectAreaLightCount: 1,
      rectAreaLightsData: discData,
    });
    const sample = sampleBdptBounce0Cpu(disc, 0, 0, 0.625)!;
    expect(sample.emitPos[0]).toBeLessThan(0);
    expect(sample.emitPos[1]).toBeGreaterThan(0);
  });

  it('pins full-family uniform roots and power-weighted distant direct selection', () => {
    const countFn = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.slice(
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf('fn bdptEmitterCount()'),
      PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf('fn bdptRandomU32('),
    );
    expect(countFn).toContain('return params.directionalLightCount + params.pointLightCount +');
    expect(countFn).toContain('bdptHasEnvironmentEndpoint()');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let discretePdf = 1.0 / f32(emitterCount);',
    );
    const scene = mixedScene();
    const root = sampleBdptBounce0Cpu(scene, 0, 0.2, 0.7)!;
    const directPdf = distantDirectSelectionPdf(scene, 0);
    expect(root.selectionPdf).toBeCloseTo(1 / 6, 15);
    expect(directPdf).toBeGreaterThan(0);
    expect(directPdf).not.toBeCloseTo(root.selectionPdf!, 6);
    expect(root.neePdf).toBeCloseTo(directPdf * root.directionPdf!, 12);
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let threshold = ((0xffffffffu % emitterCount) + 1u) % emitterCount;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('return vec3f(0.0);');
  });
});
