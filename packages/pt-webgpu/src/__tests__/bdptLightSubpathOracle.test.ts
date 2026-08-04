import { describe, expect, it } from 'vitest';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import { sampleBdptBounce0Cpu } from '../bdpt/bdptEmitterPickCpu.js';
import {
  bdptLightPathColumnIndex,
  readBdptMediumCapRow,
  readBdptMediumSideRow,
  writeBdptMediumCapRow,
  writeBdptMediumSideRow,
  packBdptLightPathColumns,
} from '../bdpt/fillBdptLightPathCpu.js';

import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
function stubScene(partial: Partial<UploadedSceneBuffers>): UploadedSceneBuffers {
  const base = {
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
  return { ...base, ...partial };
}

describe('bdptLightSubpathOracle', () => {
  it('packBdptLightPathColumns marks unused columns invalid', () => {
    const width = 3;
    const data = packBdptLightPathColumns(width, null);
    // Flat buffer layout: col*8+row vec4f. Each column's row-0 .w holds the kind.
    expect(data[bdptLightPathColumnIndex(0, 0) + 3]).toBe(3);
    expect(data[bdptLightPathColumnIndex(1, 0) + 3]).toBe(3);
    expect(data[bdptLightPathColumnIndex(2, 0) + 3]).toBe(3);
  });


  it('round-trips exact row-6 medium-side sentinels without row/column overlap', () => {
    const data = packBdptLightPathColumns(2, null);
    const sentinels = [
      {
        incidentMatId: 0x12345678,
        incidentRemainingDistance: 0.375,
        transmittedMatId: 0x9abcdef0,
        transmittedRemainingDistance: 7.25,
      },
      {
        incidentMatId: 0x0badc0de,
        incidentRemainingDistance: 19.5,
        transmittedMatId: 0x76543210,
        transmittedRemainingDistance: 0.0625,
      },
    ] as const;

    const row5Before = [
      ...data.slice(bdptLightPathColumnIndex(0, 5), bdptLightPathColumnIndex(0, 5) + 4),
      ...data.slice(bdptLightPathColumnIndex(1, 5), bdptLightPathColumnIndex(1, 5) + 4),
    ];
    writeBdptMediumSideRow(data, 0, sentinels[0]);
    writeBdptMediumSideRow(data, 1, sentinels[1]);
    writeBdptMediumCapRow(data, 0, {
      incidentInitialDistance: 1.25,
      transmittedInitialDistance: 2.5,
    });
    writeBdptMediumCapRow(data, 1, {
      incidentInitialDistance: 4.75,
      transmittedInitialDistance: 9.5,
    });

    expect(data.length).toBe(2 * 8 * 4);
    expect(bdptLightPathColumnIndex(1, 0) - bdptLightPathColumnIndex(0, 0)).toBe(8 * 4);
    expect(readBdptMediumSideRow(data, 0)).toEqual(sentinels[0]);
    expect(readBdptMediumSideRow(data, 1)).toEqual(sentinels[1]);
    expect(readBdptMediumCapRow(data, 0)).toEqual({
      incidentInitialDistance: 1.25,
      transmittedInitialDistance: 2.5,
    });
    expect(readBdptMediumCapRow(data, 1)).toEqual({
      incidentInitialDistance: 4.75,
      transmittedInitialDistance: 9.5,
    });
    expect([
      ...data.slice(bdptLightPathColumnIndex(0, 5), bdptLightPathColumnIndex(0, 5) + 4),
      ...data.slice(bdptLightPathColumnIndex(1, 5), bdptLightPathColumnIndex(1, 5) + 4),
    ]).toEqual(row5Before);

    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 8u;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bitcast<f32>(incidentMedium.matId), incidentMedium.remainingDistance,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bitcast<f32>(transmittedMedium.matId), transmittedMedium.remainingDistance,',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lightIncidentMediumMatId = bitcast<u32>(lv6.x);');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lightTransmittedMediumMatId = bitcast<u32>(lv6.z);');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lightTransmittedMediumRemainingDistance = lv6.w;');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lightIncidentMediumInitialDistance = lv7.x;');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lightTransmittedMediumInitialDistance = lv7.y;');
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
    expect(data[bdptLightPathColumnIndex(0, 0) + 3]).toBe(0);
    expect(data[bdptLightPathColumnIndex(0, 0) + 0]).toBeCloseTo(sample!.emitPos[0], 4);
    expect(data[bdptLightPathColumnIndex(0, 1) + 3]).toBeCloseTo(sample!.pdfJoint, 4);
    expect(data[bdptLightPathColumnIndex(0, 2) + 0]).toBeGreaterThan(0);
  });

  it('initialises GPU c=0 endpoints with the same vacuum sides as the CPU oracle', () => {
    const sb = stubScene({
      pointLightCount: 1,
      pointLightsData: new Float32Array([0, 1, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0]),
    });
    const sample = sampleBdptBounce0Cpu(sb, 0, 1, 0.42);
    const data = packBdptLightPathColumns(1, sample);
    expect(readBdptMediumSideRow(data, 0)).toEqual({
      incidentMatId: 0xffffffff,
      incidentRemainingDistance: Math.fround(3.402823e38),
      transmittedMatId: 0xffffffff,
      transmittedRemainingDistance: Math.fround(3.402823e38),
    });
    expect(readBdptMediumCapRow(data, 0)).toEqual({
      incidentInitialDistance: Math.fround(3.402823e38),
      transmittedInitialDistance: Math.fround(3.402823e38),
    });

    const endpointStart = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf(
      'fn bdptFinishBounce0Endpoint(',
    );
    const endpointEnd = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.indexOf(
      'fn bdptWriteBounce0(',
      endpointStart,
    );
    const endpointSource = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL.slice(endpointStart, endpointEnd);
    expect(endpointSource).toContain('bdptClearLvMaterialPayload(col);');
    expect(endpointSource).not.toContain('bdptWriteLvInterfaceEta(col, 1.0, 1.0, 1.0);');
  });

  it("stores a point endpoint at c=0 and samples its sphere density only for c>=1", () => {
    const sb = stubScene({
      pointLightCount: 1,
      pointLightsData: new Float32Array([0, 1, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0]),
    });
    const sample = sampleBdptBounce0Cpu(sb, 0, 1, 0.42);
    expect(sample).not.toBeNull();
    expect(sample!.emitNormal).toEqual([0, 1, 0]);
    expect(sample!.pdfHemi).toBe(0);
    expect(sample!.pdfJoint).toBe(1);
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      "scatterDir = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));",
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      "pdfScatter = 0.25 * INV_PI;",
    );
  });

});
