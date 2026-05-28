import { describe, expect, it } from 'vitest';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  bdptEmitterCount,
  bdptEmitterPower,
  bdptPickEmitterFlat,
  sampleBdptBounce0Cpu,
} from '../bdpt/bdptEmitterPickCpu.js';

function stubScene(partial: Partial<UploadedSceneBuffers>): UploadedSceneBuffers {
  const base = {
    directionalLight: [0, -1, 0],
    directionalIrradiance: [0, 0, 0],
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
    environmentMapWidth: 0,
    environmentMapHeight: 0,
    hasEnvironmentMap: false,
    environmentMapTexels: new Float32Array(0),
    environmentMapCdf: new Float32Array(0),
    triangleCount: 0,
    analyticCount: 0,
    warnings: [],
    tlasNodeCount: 0,
    primitiveTlasBindings: [],
    analyticHeaders: new Float32Array(0),
    analyticParams: new Float32Array(0),
    analyticLocalToWorld: new Float32Array(0),
    analyticWorldToLocal: new Float32Array(0),
    bvhNodesBuffer: null as unknown as GPUBuffer,
    bvhIndexBuffer: null as unknown as GPUBuffer,
    bvhPositionsBuffer: null as unknown as GPUBuffer,
    bvhBeerColorsBuffer: null as unknown as GPUBuffer,
    emittersBuffer: null as unknown as GPUBuffer,
    emitterCdfBuffer: null as unknown as GPUBuffer,
    pointLightsBuffer: null as unknown as GPUBuffer,
    spotLightsBuffer: null as unknown as GPUBuffer,
    rectAreaLightsBuffer: null as unknown as GPUBuffer,
    meshAreaLightsBuffer: null as unknown as GPUBuffer,
    environmentMapTexelsBuffer: null as unknown as GPUBuffer,
    environmentMapCdfBuffer: null as unknown as GPUBuffer,
    tlasNodesBuffer: null as unknown as GPUBuffer,
    tlasInstanceIndicesBuffer: null as unknown as GPUBuffer,
    tlasBlasRootsBuffer: null as unknown as GPUBuffer,
    tlasInstanceWorldToLocalBuffer: null as unknown as GPUBuffer,
    tlasInstanceLocalToWorldBuffer: null as unknown as GPUBuffer,
    bvhNodes: new Uint32Array(0),
    bvhIndex: new Uint32Array(0),
    bvhPositions: new Float32Array(0),
    bvhBeerColors: new Float32Array(0),
    emitters: new Float32Array(0),
    emitterCdf: new Float32Array(0),
    tlasNodes: new Uint32Array(0),
    tlasInstanceIndices: new Uint32Array(0),
    tlasBlasRoots: new Uint32Array(0),
    tlasInstanceWorldToLocal: new Float32Array(0),
    tlasInstanceLocalToWorld: new Float32Array(0),
  } as unknown as UploadedSceneBuffers;
  return { ...base, ...partial } as unknown as UploadedSceneBuffers;
}

describe('bdptEmitterPickCpu', () => {
  it('counts directional + point + spot + rect + mesh + env', () => {
    const sb = stubScene({
      directionalIrradiance: [1, 1, 1],
      pointLightCount: 1,
      spotLightCount: 1,
      rectAreaLightCount: 1,
      meshAreaLightCount: 1,
      pointLightsData: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0]),
      spotLightsData: new Float32Array(12).fill(0),
      rectAreaLightsData: new Float32Array(16).fill(0),
      meshAreaLightsData: new Float32Array(16).fill(0),
      environmentSunStrength: 2,
    });
    expect(bdptEmitterCount(sb)).toBe(6);
  });

  it('power-weighted pick prefers brighter mesh over dim point', () => {
    const mesh = new Float32Array(16);
    mesh[0] = 0;
    mesh[1] = 0;
    mesh[2] = 0;
    mesh[4] = 2;
    mesh[5] = 0;
    mesh[6] = 0;
    mesh[8] = 0;
    mesh[9] = 2;
    mesh[10] = 0;
    mesh[12] = 100;
    mesh[13] = 100;
    mesh[14] = 100;

    const sb = stubScene({
      pointLightCount: 1,
      meshAreaLightCount: 1,
      pointLightsData: new Float32Array([0, 1, 0, 0, 0.01, 0.01, 0.01, 0]),
      meshAreaLightsData: mesh,
    });

    const pointPower = bdptEmitterPower(sb, 0);
    const meshPower = bdptEmitterPower(sb, 1);
    expect(meshPower).toBeGreaterThan(pointPower * 10);

    const flat = bdptPickEmitterFlat(sb, pointPower + meshPower * 0.5, meshPower + pointPower, 2);
    expect(flat).toBe(1);

    const sample = sampleBdptBounce0Cpu(sb, flat, 1, 0.37);
    expect(sample).not.toBeNull();
    expect(sample!.pdfJoint).toBeGreaterThan(0);
  });

  // Regression for the bounce-0 tangent-frame bug: the cosine hemisphere
  // direction must be `wi = t*x + b*y + n*z`. The original code scaled the
  // bitangent `b` by `x` instead of `y` (the `y = r*sin(phi)` local was
  // dead), collapsing the two tangent-plane components to be equal. The
  // mesh triangle a=[0,0,0], b=[2,0,0], c=[0,2,0] has normal +Z, so with
  // uHemi=0.37 the sample gives distinct x=0.241576, y=-0.558248 and the
  // derived hemisphere pdf differs between the buggy and corrected formulas.
  it('builds the bounce-0 direction with the y-scaled bitangent (mesh emitter)', () => {
    const mesh = new Float32Array(16);
    mesh[4] = 2; // b = a + (2,0,0)
    mesh[9] = 2; // c = a + (0,2,0)
    mesh[12] = 100;
    mesh[13] = 100;
    mesh[14] = 100;
    const sb = stubScene({ meshAreaLightCount: 1, meshAreaLightsData: mesh });

    const sample = sampleBdptBounce0Cpu(sb, 0, 1, 0.37);
    expect(sample).not.toBeNull();
    // Triangle normal is +Z.
    expect(sample!.emitNormal[0]).toBeCloseTo(0, 6);
    expect(sample!.emitNormal[1]).toBeCloseTo(0, 6);
    expect(sample!.emitNormal[2]).toBeCloseTo(1, 6);
    // pdfHemi = cosEmit / PI with the CORRECTED (b*y) direction.
    // The buggy (b*x) variant yields 0.2923763342 instead.
    expect(sample!.pdfHemi).toBeCloseTo(0.25265063960867545, 9);
    expect(sample!.pdfJoint).toBeCloseTo(0.25265063960867545, 9);
    expect(sample!.pdfHemi).not.toBeCloseTo(0.2923763342268401, 9);
  });
});
