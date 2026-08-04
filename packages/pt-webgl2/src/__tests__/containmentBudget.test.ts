import { describe, expect, it } from 'vitest';
import type { FrameInput, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { DEFAULT_TRACE_FEATURES, type TraceFeatures } from '../featureTypes.js';
import {
  MATERIAL_PIXELS,
  packMaterialsTexture,
} from '../scene/materialsTexture.js';
import {
  containmentWalkQueryLimit,
  programGraphKey,
} from '../gl/glResources.js';
import * as ContainmentModule from '../glsl/shader/bvh/inside_fog_volume_function.glsl.js';
import { FOG_MATERIAL_GLSL } from '../glsl/shader/structs/fog_material.glsl.js';
import {
  PT_WEBGL2_MAX_NESTED_MEDIA,
  PT_WEBGL2_SUPPORT_MANIFEST,
} from '../supportManifest.js';
import { createMockGl } from './mockGl.js';

const inside_fog_volume_function =
  (ContainmentModule as unknown as Record<string, string>)[
    'inside_fog_volume_function'
  ] ?? '';

const SCALAR_RICH_OPAQUE: MaterialSpec = {
  baseColor: [0.7, 0.7, 0.7],
  roughness: 0.6,
  metallic: 0,
  clearcoat: 0,
};

function stackedTrianglesPrimitive(
  id: string,
  triangleCount: number,
  zOffset = 0,
): MeshPrimitive {
  const positions = new Float32Array(triangleCount * 9);
  const normals = new Float32Array(triangleCount * 9);
  const indices = new Uint32Array(triangleCount * 3);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const vertexBase = triangle * 3;
    const floatBase = triangle * 9;
    const z = zOffset + triangle * 0.125;
    positions.set([-2, -2, z, 2, -2, z, 0, 2, z], floatBase);
    normals.set([0, 0, 1, 0, 0, 1, 0, 0, 1], floatBase);
    indices[triangle * 3] = vertexBase;
    indices[triangle * 3 + 1] = vertexBase + 1;
    indices[triangle * 3 + 2] = vertexBase + 2;
  }
  return {
    kind: 'mesh',
    id,
    positions,
    normals,
    indices,
    material: SCALAR_RICH_OPAQUE,
  };
}

function sceneWithTriangles(triangleCount: number): Scene {
  return {
    // A ray through x=y=0 can cross every opaque triangle, so this fixture
    // exercises the historical >63-query failure shape rather than merely
    // compiling a scene that happens to contain many off-ray triangles.
    primitives: [stackedTrianglesPrimitive('stack', triangleCount)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function frame(): FrameInput {
  return {
    viewMatrix: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, -3, 1,
    ]) as never,
    projMatrix: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, -1, -1,
      0, 0, -0.2, 0,
    ]) as never,
    cameraPosition: [0, 0, 3] as never,
    viewport: { width: 4, height: 4, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 1 },
  };
}

function basicFeatures(): TraceFeatures {
  return {
    ...DEFAULT_TRACE_FEATURES,
    basicMaterials: true,
    scalarRichMaterials: false,
    mappedPbrMaterials: false,
    mappedRichMaterials: false,
  };
}

function materialTexel(materialIndex: number, sample: number, channel: number): number {
  return materialIndex * MATERIAL_PIXELS * 4 + sample * 4 + channel;
}

describe('pt-webgl2 exact initial-medium containment budget', () => {
  it('publishes the same explicit eight-medium limit carried by every shader stack', () => {
    expect(PT_WEBGL2_MAX_NESTED_MEDIA).toBe(8);
    expect(PT_WEBGL2_SUPPORT_MANIFEST.opticalMedia).toEqual({
      maxNestedMedia: 8,
      topology: 'closed-oriented-disjoint-or-nested',
      overflowPolicy: 'reject-scene',
    });
    expect(FOG_MATERIAL_GLSL).toContain('const int MEDIUM_STACK_CAPACITY = 8;');
  });

  it('derives N+1 queries from production triangle count without a 63-crossing cutoff', () => {
    expect(containmentWalkQueryLimit(0)).toBe(1);
    expect(containmentWalkQueryLimit(64)).toBe(65);
    expect(containmentWalkQueryLimit(70)).toBe(71);
    expect(inside_fog_volume_function).toContain(
      'uint queryLimit = uSceneTriangleCount + 1u;',
    );
    expect(inside_fog_volume_function).toContain(
      'queryIndex < queryLimit',
    );
    expect(inside_fog_volume_function).not.toContain('FOG_WALK_MAX');
    expect(inside_fog_volume_function).not.toContain('FOG_CHECK_ITERATIONS');
    expect(inside_fog_volume_function).not.toContain('windingMaterialIds');
  });

  it('rejects invalid or uint-overflowing runtime budgets deterministically', () => {
    for (const count of [-1, 0.5, Number.NaN, Number.POSITIVE_INFINITY, 0xffff_ffff]) {
      expect(() => containmentWalkQueryLimit(count)).toThrow(
        /containment triangle count/,
      );
    }
    expect(containmentWalkQueryLimit(123)).toBe(124);
  });

  it('keeps topology size out of the compile identity', () => {
    expect(programGraphKey).toHaveLength(1);
    expect(programGraphKey(basicFeatures())).not.toBe(
      programGraphKey(DEFAULT_TRACE_FEATURES),
    );
  });

  it('updates the runtime bound without relinking after a >63-triangle topology mutation', async () => {
    const record = new Map<string, unknown>();
    const engine = await createPTEngine_WebGL2({ device: createMockGl(record) });
    try {
      engine.setScene(sceneWithTriangles(64));
      expect(engine.renderFrame(frame()).kind).toBe('rendered');
      expect(record.get('uSceneTriangleCount')).toBe(64);
      const firstSources = record.get('__shaderSources') as string[];
      expect(
        firstSources.some(
          (source) =>
            source.includes('bool bvhBuildMediumStack(') &&
            source.includes('uint queryLimit = uSceneTriangleCount + 1u;'),
        ),
      ).toBe(true);

      const beforeMutationSourceCount = firstSources.length;
      engine.addPrimitive?.(stackedTrianglesPrimitive('added', 1, 8.5));
      expect(engine.renderFrame(frame()).kind).toBe('rendered');
      expect(record.get('uSceneTriangleCount')).toBe(65);
      const allSources = record.get('__shaderSources') as string[];
      expect(allSources).toHaveLength(beforeMutationSourceCount);
    } finally {
      engine.dispose();
    }
  });

  it('distinguishes a valid containing target from open or malformed topology', () => {
    const source = inside_fog_volume_function.replace(/\s+/g, ' ');
    // Infinity is accepted only with every outward front/back pair closed.
    expect(source).toContain('if ( pendingCount != 0 ) return false;');
    // Unmatched backs are containing media and are reversed outer-to-inner.
    expect(source).toContain('int sourceIndex = containingCount - 1 - outputIndex;');
    expect(source).toContain('containingMaterialIds[ sourceIndex ]');
    // Reversed, interpenetrating, or out-of-order paired components fail LIFO.
    expect(source).toContain('pendingComponentIds[ top ] != boundaryComponentId');
    // Exhausting N real groups without a miss proof also fails closed.
    expect(source).toContain('if ( queryIndex >= uSceneTriangleCount ) return false;');
    expect(source).toContain('minimumDistanceExclusive = boundaryHit.dist; } return false;');
  });

  it('scans only validated bulk boundaries from one fixed origin and exact t range', () => {
    expect(inside_fog_volume_function).toContain(
      'if ( opticalOnly && ! control.opticalVolume ) continue;',
    );
    expect(inside_fog_volume_function).toContain(
      'bvh, rayOrigin, walkDirection, minimumDistanceExclusive',
    );
    expect(inside_fog_volume_function).toContain(
      'minimumDistanceExclusive = boundaryHit.dist;',
    );
    expect(inside_fog_volume_function).not.toContain('outsideDistance');
    expect(inside_fog_volume_function).not.toContain('nextOrigin');
    expect(inside_fog_volume_function).not.toContain('stepRayOrigin(');
  });

  it('classifies authored bulk independently of a zero transmission-map texel', () => {
    const zeroTransmission = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 0, 0, 255]),
    };
    const packed = packMaterialsTexture([
      {
        ...SCALAR_RICH_OPAQUE,
        transmission: 1,
        thickness: 0.25,
        transmissionMap: { handle: zeroTransmission },
      },
    ], new Map([[zeroTransmission, 0]])).data;

    // MaterialControl derives opticalVolume from the authored scalar
    // transmission lane and the static sheet-vs-bulk bit. The texture remains
    // a real-surface BSDF gate; it cannot punch a synthetic containment hole.
    expect(packed[materialTexel(0, 2, 1)]).toBe(1);
    expect(packed[materialTexel(0, 2, 2)]).toBe(0);
    expect(packed[materialTexel(0, 11, 2)]).toBe(0);
    expect(inside_fog_volume_function).toContain(
      'control.opticalVolume != ( localComponentId != 0u )',
    );
    expect(inside_fog_volume_function).not.toContain('transmissionMap');
  });

  it('forces finite opaque coverage solid despite zero opacity and alpha texels', () => {
    const zeroCoverage = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 0, 0, 0]),
    };
    const packed = packMaterialsTexture([
      {
        ...SCALAR_RICH_OPAQUE,
        transmission: 1,
        thickness: 0.25,
        alphaMode: 'opaque',
        opacity: 0,
        baseColorMap: { handle: zeroCoverage },
        alphaMap: { handle: zeroCoverage },
      },
    ], new Map([[zeroCoverage, 0]])).data;

    expect(packed[materialTexel(0, 13, 1)]).toBe(0);
    expect(packed[materialTexel(0, 13, 2)]).toBe(0);
    expect(Number(packed[materialTexel(0, 14, 3)]) & 1).toBe(0);
    const opaqueSolid = FOG_MATERIAL_GLSL.indexOf(
      'if ( ! transparent ) return SURFACE_COVERAGE_SOLID;',
    );
    const zeroCoverageHole = FOG_MATERIAL_GLSL.indexOf(
      'if ( clampedCoverage <= 0.0 ) return SURFACE_COVERAGE_HOLE;',
    );
    expect(opaqueSolid).toBeGreaterThan(0);
    expect(zeroCoverageHole).toBeGreaterThan(opaqueSolid);
  });
});
