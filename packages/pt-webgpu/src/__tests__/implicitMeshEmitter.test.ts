/**
 * implicitMeshEmitter.test.ts — H14-A implicit mesh-area emitter synthesis tests.
 *
 * Verifies that `packEmitterArrays` synthesizes a mesh-area emitter for any
 * mesh-like primitive with non-zero emissive energy and NO explicit mesh-area
 * emitter backing, so NEE/BDPT can sample its radiance.
 *
 * Tests:
 *   1. Emissive mesh with NO explicit emitter → mesh-area triangle synthesized.
 *   2. Emissive mesh WITH explicit mesh-area emitter → NOT double-counted.
 *   3. Non-emissive mesh → no implicit emitter added.
 *   4. Emissive analytic primitive → no implicit emitter (analytic skip).
 */

import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene, TextureRef } from '@vitrum/core';
import {
  MESH_AREA_LIGHT_FLOAT_STRIDE,
  packEmitterArrays,
  packMeshAreaAdjointReplayArrays,
} from '../scene/emitterPacking.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

function triMesh(
  id: string,
  emissive?: [number, number, number],
  emissiveIntensity?: number,
  materialPatch: Partial<MaterialSpec> = {},
): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: {
      baseColor: [0.5, 0.5, 0.5],
      roughness: 0.5,
      metallic: 0,
      emissive: emissive ?? [0, 0, 0],
      emissiveIntensity: emissiveIntensity ?? 1,
      ...materialPatch,
    },
  };
}

function emissiveMap(data: Float32Array, width: number, height: number): TextureRef {
  return { handle: { width, height, data } };
}

function meshAreaEmitter(id: string, meshId: string): Scene['emitters'][number] {
  return {
    kind: 'mesh-area',
    id,
    meshId,
    color: [2, 0, 0], // intentionally different from the primitive's material emissive
    intensity: 1,
  };
}

// ─── tests ────────────────────────────────────────────────────────────────────

describe('packEmitterArrays — H14-A implicit mesh-area synthesis', () => {
  it('emissive mesh with NO explicit emitter → mesh-area triangle synthesized', () => {
    const scene: Scene = {
      primitives: [triMesh('glow', [1, 0.5, 0.25], 2)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    // One triangle from the implicit emitter.
    expect(packed.meshAreaLightCount).toBeGreaterThan(0);
    // The synthesized radiance = emissive * emissiveIntensity = [2, 1, 0.5].
    // Radiance is packed at float offset 12 (vec4f #3) per triangle.
    const radR = packed.meshAreaLightsData[12]!;
    const radG = packed.meshAreaLightsData[13]!;
    const radB = packed.meshAreaLightsData[14]!;
    expect(radR).toBeCloseTo(2, 5);
    expect(radG).toBeCloseTo(1, 5);
    expect(radB).toBeCloseTo(0.5, 5);
  });

  it('retains full-triangle support and a positive power proxy for mapped implicit emission', () => {
    const scene: Scene = {
      primitives: [
        triMesh('mapped-glow', [2, 2, 2], 1, {
          emissiveMap: emissiveMap(
            new Float32Array([
              1, 0, 0, 1,
              0, 0, 1, 1,
            ]),
            2,
            1,
          ),
        }),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(1);
    // The proposal stores unmodulated base Le so every potentially bright
    // filtered texel has positive sampling support. Exact map Le is evaluated
    // at the sampled barycentric point in WGSL.
    expect(packed.meshAreaLightsData[12]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[13]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[14]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[22]).toBe(1);
    expect(packed.meshAreaLightsData[24]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[25]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[26]).toBeCloseTo(2, 5);
  });

  it('retains proposal support for mapped emission below the former 1e-6 cutoff', () => {
    const scene: Scene = {
      primitives: [
        triMesh('very-dim-mapped-glow', [1, 1, 1], 1, {
          emissiveMap: emissiveMap(new Float32Array([1e-8, 0, 0, 1]), 1, 1),
        }),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(1);
    expect(Array.from(packed.meshAreaLightsData.slice(12, 15))).toEqual([1, 1, 1]);
  });

  it('keeps a black/non-black neighbour and repeat seam in proposal support', () => {
    const scene: Scene = {
      primitives: [{
        ...(triMesh('mapped-glow-exact', [2, 2, 2], 1, {
          emissiveMap: emissiveMap(
            new Float32Array([
              0, 0, 0, 1,
              1, 0, 0, 1,
            ]),
            2,
            1,
          ),
        }) as MeshPrimitive),
        uvs: new Float32Array([-0.01, 0, 0.01, 0, -0.01, 1]),
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(1);
    expect(packed.meshAreaLightsData[12]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[13]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[14]).toBeCloseTo(2, 5);
    const packedUvs = Array.from(packed.meshAreaLightsData.slice(16, 22));
    expect(packedUvs[0]).toBeCloseTo(-0.01, 6);
    expect(packedUvs[2]).toBeCloseTo(0.01, 6);
    expect(packedUvs[4]).toBeCloseTo(-0.01, 6);
    expect([packedUvs[1], packedUvs[3], packedUvs[5]]).toEqual([0, 0, 1]);
  });

  it('packs an arbitrary authored UV set for exact GPU emissive sampling', () => {
    const scene: Scene = {
      primitives: [{
        ...(triMesh('mapped-glow-uv3', [2, 2, 2], 1, {
          emissiveMap: {
            ...emissiveMap(
              new Float32Array([
                1, 0, 0, 1,
                0, 1, 0, 1,
              ]),
              2,
              1,
            ),
            texCoord: 3,
          },
        }) as MeshPrimitive),
        uvSets: [
          undefined,
          undefined,
          undefined,
          new Float32Array([0, 0, 1, 0, 0, 1]),
        ],
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(1);
    expect(Array.from(packed.meshAreaLightsData.slice(16, 22))).toEqual([
      0, 0, 1, 0, 0, 1,
    ]);
    expect(packed.meshAreaLightsData[22]).toBe(1);
  });

  it('packs explicit mapped mesh emission as exact-sample metadata plus a power proxy', () => {
    const primitive = {
      ...triMesh('mapped-panel', [0, 0, 0], 0, {
        emissiveMap: emissiveMap(
          new Float32Array([
            1, 0, 0, 1,
            0, 1, 0, 1,
          ]),
          2,
          1,
        ),
      }),
      uvs: new Float32Array([0.75, 0, 0.75, 0, 0.75, 0]),
    } as Scene['primitives'][number];
    const scene: Scene = {
      primitives: [primitive],
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-explicit',
        meshId: 'mapped-panel',
        color: [2, 2, 2],
        intensity: 1,
      }],
      environment: { kind: 'none' },
    };

    const packed = packEmitterArrays(scene);

    expect(packed.meshAreaLightCount).toBe(1);
    expect(packed.meshAreaLightsData[12]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[13]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[14]).toBeCloseTo(2, 5);
    expect(Array.from(packed.meshAreaLightsData.slice(16, 22))).toEqual([
      0.75, 0, 0.75, 0, 0.75, 0,
    ]);
    expect(packed.meshAreaLightsData[24]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData.length).toBe(MESH_AREA_LIGHT_FLOAT_STRIDE);
  });

  it('packs source factors for mapped explicit mesh-area emitters with zero authored color channels', () => {
    const primitive = {
      ...triMesh('mapped-zero-red-panel', [0, 0, 0], 0, {
        emissiveMap: emissiveMap(new Float32Array([1, 0, 1, 1]), 1, 1),
      }),
      uvs: new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    } as Scene['primitives'][number];
    const scene: Scene = {
      primitives: [primitive],
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-zero-red',
        meshId: 'mapped-zero-red-panel',
        color: [0, 0.5, 1],
        intensity: 4,
      }],
      environment: { kind: 'none' },
    };

    const packed = packEmitterArrays(scene);

    expect(packed.meshAreaLightCount).toBe(1);
    expect(packed.meshAreaLightsData[12]).toBeCloseTo(0, 5);
    expect(packed.meshAreaLightsData[13]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[14]).toBeCloseTo(4, 5);
    expect(packed.meshAreaLightSourceFactorsData.length).toBe(4);
    expect(packed.meshAreaLightSourceFactorsData[0]).toBeCloseTo(1, 5);
    expect(packed.meshAreaLightSourceFactorsData[1]).toBeCloseTo(1, 5);
    expect(packed.meshAreaLightSourceFactorsData[2]).toBeCloseTo(1, 5);
  });

  it('keeps zero-intensity explicit mapped mesh-area emitters only in the adjoint replay stream', () => {
    const primitive = {
      ...triMesh('mapped-zero-intensity-panel', [0, 0, 0], 0, {
        emissiveMap: emissiveMap(new Float32Array([1, 0, 1, 1]), 1, 1),
      }),
      uvs: new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    } as Scene['primitives'][number];
    const scene: Scene = {
      primitives: [primitive],
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-zero-intensity',
        meshId: 'mapped-zero-intensity-panel',
        color: [0.25, 0.5, 1],
        intensity: 0,
      }],
      environment: { kind: 'none' },
    };

    const production = packEmitterArrays(scene);
    const replay = packMeshAreaAdjointReplayArrays(scene);

    expect(production.meshAreaLightCount).toBe(0);
    expect(replay.meshAreaLightCount).toBe(1);
    expect(replay.meshAreaLightsData[12]).toBeCloseTo(0, 5);
    expect(replay.meshAreaLightsData[13]).toBeCloseTo(0, 5);
    expect(replay.meshAreaLightsData[14]).toBeCloseTo(0, 5);
    expect(replay.meshAreaLightSourceFactorsData[0]).toBeCloseTo(1, 5);
    expect(replay.meshAreaLightSourceFactorsData[1]).toBeCloseTo(1, 5);
    expect(replay.meshAreaLightSourceFactorsData[2]).toBeCloseTo(1, 5);
  });

  it('does not synthesize an implicit emitter when a readable emissiveMap averages black', () => {
    const scene: Scene = {
      primitives: [
        triMesh('mapped-dark', [4, 4, 4], 10, {
          emissiveMap: emissiveMap(new Float32Array([0, 0, 0, 1]), 1, 1),
        }),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(0);
  });

  it('fails closed for opaque emissiveMap handles instead of biasing MIS', () => {
    const scene: Scene = {
      primitives: [
        triMesh('opaque-map', [1, 0.25, 0], 2, {
          emissiveMap: { handle: { label: 'gpu-only-texture' } },
        }),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(() => packEmitterArrays(scene)).toThrow(
      /emissiveMap without complete CPU-readable texels/,
    );
  });

  it('emissive mesh WITH explicit mesh-area emitter → NOT double-counted', () => {
    const scene: Scene = {
      primitives: [triMesh('glow', [1, 0.5, 0.25], 2)],
      emitters: [meshAreaEmitter('e1', 'glow')],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    // Only ONE triangle from the explicit emitter — not two (no synthesis).
    // The explicit emitter has color [2,0,0] intensity 1 → radiance [2,0,0].
    // The implicit synthesizer should be suppressed.
    expect(packed.meshAreaLightCount).toBe(1);
    const radR = packed.meshAreaLightsData[12]!;
    expect(radR).toBeCloseTo(2, 5); // explicit emitter's color
    // There should NOT be a second triangle with the material emissive color.
  });

  it('non-emissive mesh → no implicit emitter added', () => {
    const scene: Scene = {
      primitives: [triMesh('dark', [0, 0, 0], 0)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(0);
  });

  it('mesh with emissiveIntensity=0 → no implicit emitter (luminance gate)', () => {
    const scene: Scene = {
      primitives: [triMesh('dimmed', [1, 0.5, 0.25], 0)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(0);
  });

  it('emissive + non-emissive meshes → only emissive gets an implicit emitter', () => {
    const scene: Scene = {
      primitives: [
        triMesh('dark', [0, 0, 0], 0),
        triMesh('bright', [0.8, 0.8, 0.8], 3),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(1);
  });
});
