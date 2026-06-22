import { describe, expect, it, vi } from 'vitest';
import type {
  EngineWarning,
  InstancedMeshPrimitive,
  MaterialSpec,
  MeshPrimitive,
  Scene,
} from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { LIGHT_TREE_FLOATS_PER_NODE } from '@vitrum/shared-samplers';
import { buildReSTIRSceneBVHForCoreScene } from '../bvhCore.js';
import {
  collectMeshAreaEmitterTrisFromCore,
  collectRectAreaEmitterTrisFromCore,
  packEmitterTrisForDDGI,
} from '../bvhSceneHelpers.js';

const EMITTER_FLOATS = 20;

interface DecodedEmitter {
  vA: [number, number, number];
  sourceTriIndex: number;
  vB: [number, number, number];
  sourceSubdivLevel: number;
  vC: [number, number, number];
  sourceSubdivOrdinal: number;
  normal: [number, number, number];
  area: number;
  color: [number, number, number];
  castShadowDisabled: number;
  centroid: [number, number, number];
}

function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function hasColor(
  emitters: readonly DecodedEmitter[],
  color: readonly [number, number, number],
): boolean {
  return emitters.some((e) =>
    e.color[0] === color[0] &&
    e.color[1] === color[1] &&
    e.color[2] === color[2]
  );
}

function decodeEmitters(buffer: ArrayBuffer): DecodedEmitter[] {
  const emitterFloats = new Float32Array(buffer);
  const count = Math.floor(emitterFloats.length / EMITTER_FLOATS);
  const out: DecodedEmitter[] = [];
  for (let i = 0; i < count; i += 1) {
    const b = i * EMITTER_FLOATS;
    const vA: [number, number, number] = [
      emitterFloats[b]!,
      emitterFloats[b + 1]!,
      emitterFloats[b + 2]!,
    ];
    const sourceTriIndex = emitterFloats[b + 3]!;
    const vB: [number, number, number] = [
      emitterFloats[b + 4]!,
      emitterFloats[b + 5]!,
      emitterFloats[b + 6]!,
    ];
    const sourceSubdivLevel = emitterFloats[b + 7]!;
    const vC: [number, number, number] = [
      emitterFloats[b + 8]!,
      emitterFloats[b + 9]!,
      emitterFloats[b + 10]!,
    ];
    const sourceSubdivOrdinal = emitterFloats[b + 11]!;
    const normal: [number, number, number] = [
      emitterFloats[b + 12]!,
      emitterFloats[b + 13]!,
      emitterFloats[b + 14]!,
    ];
    const area = emitterFloats[b + 15]!;
    const color: [number, number, number] = [
      emitterFloats[b + 16]!,
      emitterFloats[b + 17]!,
      emitterFloats[b + 18]!,
    ];
    const castShadowDisabled = emitterFloats[b + 19]!;
    out.push({
      vA,
      sourceTriIndex,
      vB,
      sourceSubdivLevel,
      vC,
      sourceSubdivOrdinal,
      normal,
      area,
      color,
      castShadowDisabled,
      centroid: [
        (vA[0] + vB[0] + vC[0]) / 3,
        (vA[1] + vB[1] + vC[1]) / 3,
        (vA[2] + vB[2] + vC[2]) / 3,
      ],
    });
  }
  return out;
}

function decodeLightTreeLeafPowers(buffer: ArrayBuffer, nodeCount: number): number[] {
  const nodes = new Float32Array(buffer);
  const leafPairs: { emitterIndex: number; power: number }[] = [];
  for (let i = 0; i < nodeCount; i += 1) {
    const base = i * LIGHT_TREE_FLOATS_PER_NODE;
    const leftChild = nodes[base + 2]!;
    const rightChild = nodes[base + 3]!;
    const emitterIndex = nodes[base + 0]!;
    if (leftChild < 0 && rightChild < 0 && emitterIndex >= 0) {
      leafPairs.push({ emitterIndex, power: nodes[base + 1]! });
    }
  }
  return leafPairs.sort((a, b) => a.emitterIndex - b.emitterIndex).map((p) => p.power);
}

function stripPlaceholder(es: DecodedEmitter[]): DecodedEmitter[] {
  if (es.length !== 1) return es;
  const e = es[0]!;
  const isPlaceholder =
    e.vA[0] === 0 &&
    e.vA[1] === 10 &&
    e.vA[2] === 0 &&
    e.color[0] === 1 &&
    e.color[1] === 1 &&
    e.color[2] === 1 &&
    e.area === 0.5;
  return isPlaceholder ? [] : es;
}

function opaqueMaterial(): MaterialSpec {
  return { baseColor: [0.2, 0.2, 0.2], roughness: 1, metallic: 0 };
}

function emissiveMaterial(emissive: [number, number, number], intensity = 1): MaterialSpec {
  return {
    baseColor: [0, 0, 0],
    roughness: 1,
    metallic: 0,
    emissive,
    emissiveIntensity: intensity,
  };
}

function supportTriangle(id = 'support'): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
    material: opaqueMaterial(),
  };
}

function translation(x: number, y: number, z: number): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

function scale(x: number, y: number, z: number): Float32Array {
  return new Float32Array([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1]);
}

describe('core ReSTIR direct-light emitter fidelity', () => {
  it('tessellates disc-area emitters as an area-preserving triangle fan', () => {
    const scene: Scene = {
      primitives: [supportTriangle()],
      emitters: [{
        kind: 'disc-area',
        id: 'disc',
        position: [1, 2, 3],
        normal: [0, 0, 1],
        radius: 0.5,
        color: [2, 0.5, 0.25],
        intensity: 3,
      }],
      environment: { kind: 'none' },
    };

    const extra = collectRectAreaEmitterTrisFromCore(scene);
    const expectedArea = Math.PI * 0.5 * 0.5;
    const expectedLe: [number, number, number] = [6, 1.5, 0.75];

    expect(extra).toHaveLength(32);
    expect(extra.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(expectedArea, 5);
    for (const e of extra) {
      expect(e.area).toBeCloseTo(expectedArea / 32, 6);
      expect(e.normal[0]).toBeCloseTo(0, 6);
      expect(e.normal[1]).toBeCloseTo(0, 6);
      expect(e.normal[2]).toBeCloseTo(-1, 6);
      expect(e.Le).toEqual(expectedLe);
    }

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    expect(emitters).toHaveLength(32);
    expect(emitters.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(expectedArea, 5);
    expect(buffers.totalEmissivePower).toBeCloseTo(
      luminance(expectedLe[0], expectedLe[1], expectedLe[2]) * expectedArea,
      4,
    );
  });

  it('includes one world-space emissive triangle per instanced mesh instance', () => {
    const instanced: InstancedMeshPrimitive = {
      kind: 'instanced-mesh',
      id: 'instanced-panel',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: emissiveMaterial([0.25, 0.5, 1], 7),
      instances: [
        asMat4(translation(0, 0, 0)),
        asMat4(translation(2, 0, 0)),
      ],
    };
    const scene: Scene = {
      primitives: [instanced],
      emitters: [],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    const centroidsX = emitters.map((e) => e.centroid[0]).sort((a, b) => a - b);

    expect(emitters).toHaveLength(2);
    expect(centroidsX[0]).toBeCloseTo(1 / 3, 5);
    expect(centroidsX[1]).toBeCloseTo(2 + 1 / 3, 5);
    for (const e of emitters) {
      expect(e.area).toBeCloseTo(0.5, 5);
      expect(e.color).toEqual([0.25, 0.5, 1]);
      expect(e.castShadowDisabled).toBe(0);
      expect(e.sourceTriIndex).toBe(0);
    }
    expect(buffers.totalEmissivePower).toBeCloseTo(luminance(0.25, 0.5, 1) * 1.0, 5);
  });

  it('preserves castShadow:false on mesh-material ReSTIR emitters', () => {
    const panel: MeshPrimitive = {
      ...supportTriangle('shadowless-emissive-panel'),
      castShadow: false,
      material: emissiveMaterial([1.5, 0.75, 0.25], 2),
    };
    const scene: Scene = {
      primitives: [panel, supportTriangle('force-tlas')],
      emitters: [],
      environment: { kind: 'none' },
    };

    for (const bvhMode of ['merged', 'tlas'] as const) {
      const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode });
      const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
      expect(emitters).toHaveLength(1);
      expect(emitters[0]!.color).toEqual([1.5, 0.75, 0.25]);
      expect(emitters[0]!.castShadowDisabled).toBe(1);
    }
  });

  it('packs scalar radiance plus UV-local micro-emitter power for merged emissiveMap emitters', () => {
    const panel: MeshPrimitive = {
      ...supportTriangle('emissive-map-panel'),
      material: {
        ...emissiveMaterial([2, 2, 2], 3),
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Float32Array([
              0.25, 0.5, 1, 1,
              0.75, 0.25, 0.5, 1,
            ]),
            __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
          },
        },
      },
    };
    const scene: Scene = {
      primitives: [panel],
      emitters: [],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    const expectedScalarLe: [number, number, number] = [2, 2, 2];
    const scalarPowerPerMicroEmitter =
      luminance(expectedScalarLe[0], expectedScalarLe[1], expectedScalarLe[2]) * 0.125;

    expect(emitters).toHaveLength(4);
    expect(emitters.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(0.5, 6);
    expect(emitters.map((e) => e.sourceSubdivOrdinal)).toEqual([0, 1, 2, 3]);
    for (const e of emitters) {
      expect(e.sourceTriIndex).toBe(0);
      expect(e.sourceSubdivLevel).toBe(2);
      expect(e.color[0]).toBeCloseTo(expectedScalarLe[0], 6);
      expect(e.color[1]).toBeCloseTo(expectedScalarLe[1], 6);
      expect(e.color[2]).toBeCloseTo(expectedScalarLe[2], 6);
      expect(e.area).toBeCloseTo(0.125, 6);
    }
    expect(buffers.lightTreeEnabled).toBe(true);
    const leafPowers = decodeLightTreeLeafPowers(buffers.lightTree.cpuData, buffers.lightTreeNodeCount);
    expect(leafPowers).toHaveLength(4);
    for (const power of leafPowers) {
      expect(power).toBeGreaterThan(0);
      expect(power).not.toBeCloseTo(scalarPowerPerMicroEmitter, 6);
    }
    expect(leafPowers.reduce((sum, p) => sum + p, 0)).toBeCloseTo(buffers.totalEmissivePower, 6);
    expect(buffers.totalEmissivePower).toBeGreaterThan(0);
    expect(new Float32Array(buffers.emitterCdf.cpuData).at(-1)).toBe(1);
  });

  it('packs TLAS emissiveMap emitters with local material-atlas source triangles', () => {
    const panel: MeshPrimitive = {
      ...supportTriangle('emissive-map-panel'),
      material: {
        ...emissiveMaterial([2, 2, 2], 3),
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Float32Array([
              0.25, 0.5, 1, 1,
              0.75, 0.25, 0.5, 1,
            ]),
            __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
          },
        },
      },
    };
    const blocker = supportTriangle('force-tlas');
    const scene: Scene = {
      primitives: [panel, blocker],
      emitters: [],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    const expectedScalarLe: [number, number, number] = [2, 2, 2];

    expect(emitters).toHaveLength(4);
    expect(emitters.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(0.5, 6);
    expect(emitters.map((e) => e.sourceSubdivOrdinal)).toEqual([0, 1, 2, 3]);
    for (const e of emitters) {
      expect(e.sourceTriIndex).toBe(0);
      expect(e.sourceSubdivLevel).toBe(2);
      expect(e.color[0]).toBeCloseTo(expectedScalarLe[0], 6);
      expect(e.color[1]).toBeCloseTo(expectedScalarLe[1], 6);
      expect(e.color[2]).toBeCloseTo(expectedScalarLe[2], 6);
    }
    expect(buffers.totalEmissivePower).toBeGreaterThan(0);
  });

  it('clips instanced TLAS emissiveMap emitters into exact texel-cell scalar records', () => {
    const instanced: InstancedMeshPrimitive = {
      kind: 'instanced-mesh',
      id: 'instanced-emissive-map',
      castShadow: false,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: {
        ...emissiveMaterial([4, 4, 4], 0.5),
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Float32Array([
              0.5, 0.25, 0.25, 1,
              0.25, 0.75, 0.5, 1,
            ]),
            __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
          },
        },
      },
      instances: [
        asMat4(translation(0, 0, 0)),
        asMat4(translation(3, 0, 0)),
      ],
    };
    const scene: Scene = {
      primitives: [instanced],
      emitters: [],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    expect(emitters).toHaveLength(6);
    for (const e of emitters) {
      expect(e.sourceTriIndex).toBe(-1);
      expect(e.sourceSubdivLevel).toBe(1);
      expect(e.sourceSubdivOrdinal).toBe(0);
      expect(e.castShadowDisabled).toBe(1);
    }
    expect(emitters.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(1.0, 6);
    expect(hasColor(emitters, [2, 1, 1])).toBe(true);
    expect(hasColor(emitters, [1, 3, 2])).toBe(true);
    expect(buffers.totalEmissivePower).toBeGreaterThan(0);
  });

  it('clips mirrored TLAS emissiveMap emitters into exact texel-cell scalar records', () => {
    const instanced: InstancedMeshPrimitive = {
      kind: 'instanced-mesh',
      id: 'mirrored-emissive-map',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: {
        ...emissiveMaterial([3, 3, 3], 1),
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Float32Array([
              0.25, 0.25, 1, 1,
              1, 0.25, 0.25, 1,
            ]),
            __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
          },
        },
      },
      instances: [asMat4(scale(-1, 1, 1))],
    };
    const scene: Scene = {
      primitives: [instanced],
      emitters: [],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    expect(emitters).toHaveLength(3);
    for (const e of emitters) {
      expect(e.sourceTriIndex).toBe(-1);
      expect(e.sourceSubdivLevel).toBe(1);
      expect(e.sourceSubdivOrdinal).toBe(0);
    }
    expect(emitters.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(0.5, 6);
    expect(hasColor(emitters, [0.75, 0.75, 3])).toBe(true);
    expect(hasColor(emitters, [3, 0.75, 0.75])).toBe(true);
    expect(buffers.totalEmissivePower).toBeGreaterThan(0);
  });

  it('does not mark transmissive secondary emitters as emissive-map sources', () => {
    const glassPanel: MeshPrimitive = {
      ...supportTriangle('glass-panel'),
      material: {
        ...opaqueMaterial(),
        transmission: 0.75,
        attenuationColor: [0.8, 0.9, 1],
      },
    };
    const scene: Scene = {
      primitives: [glassPanel],
      emitters: [],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, {
      bvhMode: 'merged',
      primaryLightDir: { x: 0, y: 0, z: 1 },
      primaryLightIntensity: 2,
    });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));

    expect(emitters).toHaveLength(1);
    expect(emitters[0]!.sourceTriIndex).toBe(-1);
    expect(emitters[0]!.color[0]).toBeCloseTo(0.2 * 0.8 * 0.75 * 2, 6);
    expect(emitters[0]!.color[1]).toBeCloseTo(0.2 * 0.9 * 0.75 * 2, 6);
    expect(emitters[0]!.color[2]).toBeCloseTo(0.2 * 1.0 * 0.75 * 2, 6);
  });

  it('does not duplicate material-emissive mesh triangles for mesh-area emitters', () => {
    const panel: MeshPrimitive = {
      ...supportTriangle('panel'),
      material: emissiveMaterial([1, 0.25, 0.1], 1),
    };
    const scene: Scene = {
      primitives: [panel],
      emitters: [{
        kind: 'mesh-area',
        id: 'panel-emitter',
        meshId: 'panel',
        color: [10, 10, 10],
        intensity: 10,
      }],
      environment: { kind: 'none' },
    };

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));

    // H23: mesh-area emitter color*intensity (10*10=100 per channel) overrides the
    // material emissive ([1, 0.25, 0.1]) when a mesh-area emitter references the primitive.
    // The emitter color/intensity is the physical radiance; the material emissive is replaced.
    const expectedLe: [number, number, number] = [100, 100, 100]; // color=[10,10,10] * intensity=10
    expect(emitters).toHaveLength(1);
    expect(emitters[0]!.color[0]).toBeCloseTo(expectedLe[0], 3);
    expect(emitters[0]!.color[1]).toBeCloseTo(expectedLe[1], 3);
    expect(emitters[0]!.color[2]).toBeCloseTo(expectedLe[2], 3);
    expect(buffers.totalEmissivePower).toBeCloseTo(luminance(expectedLe[0], expectedLe[1], expectedLe[2]) * 0.5, 5);
  });

  it('packs castShadow:false on core area emitters into the shared emitter-triangle lane', () => {
    const scene: Scene = {
      primitives: [supportTriangle()],
      emitters: [{
        kind: 'rect-area',
        id: 'rect',
        position: [0, 2, 0],
        uAxis: [1, 0, 0],
        vAxis: [0, 0, 1],
        color: [1, 2, 3],
        intensity: 4,
        castShadow: false,
      }],
      environment: { kind: 'none' },
    };

    const extra = collectRectAreaEmitterTrisFromCore(scene);
    expect(extra).toHaveLength(2);
    expect(extra.every((e) => e.castShadow === false)).toBe(true);

    const ddgiPacked = packEmitterTrisForDDGI(extra);
    expect(ddgiPacked.count).toBe(2);
    expect(ddgiPacked.data[19]).toBe(1);
    expect(ddgiPacked.data[39]).toBe(1);

    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
    expect(emitters).toHaveLength(2);
    expect(emitters[0]!.castShadowDisabled).toBe(1);
    expect(emitters[1]!.castShadowDisabled).toBe(1);
    expect(emitters[0]!.sourceTriIndex).toBe(-1);
    expect(emitters[1]!.sourceTriIndex).toBe(-1);
  });

  it('packs mesh-area castShadow:false into the DDGI/RC emitter-triangle lane', () => {
    const panel: MeshPrimitive = supportTriangle('panel');
    const scene: Scene = {
      primitives: [panel],
      emitters: [{
        kind: 'mesh-area',
        id: 'panel-emitter',
        meshId: 'panel',
        color: [0.5, 0.25, 0.125],
        intensity: 8,
        castShadow: false,
      }],
      environment: { kind: 'none' },
    };

    const extra = collectMeshAreaEmitterTrisFromCore(scene);
    expect(extra).toHaveLength(1);
    expect(extra[0]!.Le).toEqual([4, 2, 1]);
    expect(extra[0]!.castShadow).toBe(false);

    const ddgiPacked = packEmitterTrisForDDGI(extra);
    expect(ddgiPacked.count).toBe(1);
    expect(ddgiPacked.data[3]).toBe(-1);
    expect(ddgiPacked.data[7]).toBe(1);
    expect(ddgiPacked.data[11]).toBe(0);
    expect(ddgiPacked.data[19]).toBe(1);

    for (const bvhMode of ['merged', 'tlas'] as const) {
      const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode });
      const emitters = stripPlaceholder(decodeEmitters(buffers.emitters.cpuData));
      expect(emitters).toHaveLength(1);
      expect(emitters[0]!.color).toEqual([4, 2, 1]);
      expect(emitters[0]!.castShadowDisabled).toBe(1);
    }
  });

  it('routes missing mesh-area DDGI emitter references through structured warnings', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const warnings: EngineWarning[] = [];
    const scene: Scene = {
      primitives: [supportTriangle('panel')],
      emitters: [{
        kind: 'mesh-area',
        id: 'missing-panel-emitter',
        meshId: 'missing-panel',
        color: [1, 1, 1],
        intensity: 2,
      }],
      environment: { kind: 'none' },
    };

    const extra = collectMeshAreaEmitterTrisFromCore(scene, {
      onWarning: (warning) => warnings.push(warning),
      warningPhase: 'lifecycle',
      warningMethod: 'syncDdgiFromCoreScene',
    });

    expect(extra).toEqual([]);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toEqual([
      expect.objectContaining({
        code: 'walkaround-hybrid.mesh-area-emitter-missing-mesh',
        backend: 'walkaround-hybrid',
        phase: 'lifecycle',
        method: 'syncDdgiFromCoreScene',
        details: {
          emitterId: 'missing-panel-emitter',
          meshId: 'missing-panel',
          source: 'ddgi-probe-emitter-tris',
          fallback: 'emitter skipped',
        },
      }),
    ]);
    warnSpy.mockRestore();
  });

  it('packs mesh-area source triangle metadata into the DDGI emitter-triangle lanes when TLAS bindings are available', () => {
    const panel: MeshPrimitive = supportTriangle('panel');
    const scene: Scene = {
      primitives: [panel],
      emitters: [{
        kind: 'mesh-area',
        id: 'panel-emitter',
        meshId: 'panel',
        color: [1, 1, 1],
        intensity: 2,
      }],
      environment: { kind: 'none' },
    };

    const extra = collectMeshAreaEmitterTrisFromCore(scene, {
      tlasPrimitiveBindings: [{
        primitiveId: 'panel',
        primitiveKind: 'mesh',
        blasRoot: 0,
        instanceCount: 1,
        vertexStart: 0,
        vertexCount: 3,
        triStart: 7,
        triCount: 1,
        localAabbMin: [0, 0, 0],
        localAabbMax: [1, 1, 0],
      }],
    });
    expect(extra).toHaveLength(1);
    expect(extra[0]!.sourceTriIndex).toBe(7);

    const ddgiPacked = packEmitterTrisForDDGI(extra);
    expect(ddgiPacked.count).toBe(1);
    expect(ddgiPacked.data[3]).toBe(7);
    expect(ddgiPacked.data[7]).toBe(1);
    expect(ddgiPacked.data[11]).toBe(0);
  });

  it('subdivides explicit mesh-area DDGI triangles through the referenced material emissiveMap', () => {
    const panel: MeshPrimitive = {
      ...supportTriangle('panel'),
      uvs: new Float32Array([0.75, 0, 0.75, 0, 0.75, 0]),
      material: {
        ...opaqueMaterial(),
        emissiveMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Uint8Array([
              255, 0, 0, 255,
              0, 255, 0, 255,
            ]),
          },
        },
      },
    };
    const scene: Scene = {
      primitives: [panel],
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-panel',
        meshId: 'panel',
        color: [2, 2, 2],
        intensity: 1,
      }],
      environment: { kind: 'none' },
    };

    const extra = collectMeshAreaEmitterTrisFromCore(scene, {
      tlasPrimitiveBindings: [{
        primitiveId: 'panel',
        primitiveKind: 'mesh',
        blasRoot: 0,
        instanceCount: 1,
        vertexStart: 0,
        vertexCount: 3,
        triStart: 7,
        triCount: 1,
        localAabbMin: [0, 0, 0],
        localAabbMax: [1, 1, 0],
      }],
    });

    expect(extra).toHaveLength(4);
    expect(extra.reduce((sum, e) => sum + e.area, 0)).toBeCloseTo(0.5, 6);
    expect(extra.map((e) => e.Le)).toEqual([
      [0, 2, 0],
      [0, 2, 0],
      [0, 2, 0],
      [0, 2, 0],
    ]);
    expect(extra.map((e) => e.sourceTriIndex)).toEqual([7, 7, 7, 7]);
    expect(extra.map((e) => e.sourceSubdivLevel)).toEqual([2, 2, 2, 2]);
    expect(extra.map((e) => e.sourceSubdivOrdinal)).toEqual([0, 1, 2, 3]);

    const ddgiPacked = packEmitterTrisForDDGI(extra);
    expect(ddgiPacked.count).toBe(4);
    expect(ddgiPacked.data[3]).toBe(7);
    expect(ddgiPacked.data[7]).toBe(2);
    expect(ddgiPacked.data[11]).toBe(0);
    expect(ddgiPacked.data[16]).toBeCloseTo(0, 6);
    expect(ddgiPacked.data[17]).toBeCloseTo(2, 6);
    expect(ddgiPacked.data[18]).toBeCloseTo(0, 6);
  });
});
