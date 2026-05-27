import { describe, expect, it } from 'vitest';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import { sampleBdptBounce0Cpu } from '../bdpt/bdptEmitterPickCpu.js';
import { packBdptLightPathColumns } from '../bdpt/fillBdptLightPathCpu.js';

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

describe('bdptLightSubpathOracle', () => {
  it('packBdptLightPathColumns marks unused columns invalid', () => {
    const width = 3;
    const data = packBdptLightPathColumns(width, null);
    expect(data[3]).toBe(3);
    expect(data[7]).toBe(3);
    expect(data[11]).toBe(3);
  });

  it('packBdptLightPathColumns matches sampleBdptBounce0Cpu for point emitter', () => {
    const sb = stubScene({
      pointLightCount: 1,
      pointLightsData: new Float32Array([0, 1, 0, 0, 10, 10, 10, 0]),
    });
    const sample = sampleBdptBounce0Cpu(sb, 0, 1, 0.42);
    expect(sample).not.toBeNull();
    const width = 3;
    const data = packBdptLightPathColumns(width, sample);
    expect(data[3]).toBe(0);
    expect(data[0]).toBeCloseTo(sample!.emitPos[0], 4);
    expect(data[width * 4 + 3]).toBeCloseTo(sample!.pdfJoint, 4);
    expect(data[width * 8 + 0]).toBeGreaterThan(0);
  });
});
