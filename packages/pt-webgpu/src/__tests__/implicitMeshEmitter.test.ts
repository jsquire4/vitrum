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
import type { MaterialSpec, Scene, TextureRef } from '@vitrum/core';
import { packEmitterArrays } from '../scene/emitterPacking.js';

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
    const STRIDE = 16; // MESH_AREA_LIGHT_FLOAT_STRIDE
    const radR = packed.meshAreaLightsData[12]!;
    const radG = packed.meshAreaLightsData[13]!;
    const radB = packed.meshAreaLightsData[14]!;
    expect(radR).toBeCloseTo(2, 5);
    expect(radG).toBeCloseTo(1, 5);
    expect(radB).toBeCloseTo(0.5, 5);
    void STRIDE;
  });

  it('modulates implicit mesh-area radiance by UV-local CPU-readable emissiveMap energy', () => {
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
    expect(packed.meshAreaLightsData[12]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[13]).toBeCloseTo(0, 5);
    expect(packed.meshAreaLightsData[14]).toBeCloseTo(0, 5);
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

  it('warns and falls back to scalar emissive radiance for opaque emissiveMap handles', () => {
    const scene: Scene = {
      primitives: [
        triMesh('opaque-map', [1, 0.25, 0], 2, {
          emissiveMap: { handle: { label: 'gpu-only-texture' } },
        }),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packEmitterArrays(scene);
    expect(packed.meshAreaLightCount).toBe(1);
    expect(packed.meshAreaLightsData[12]).toBeCloseTo(2, 5);
    expect(packed.meshAreaLightsData[13]).toBeCloseTo(0.5, 5);
    expect(packed.warnings).toContain(
      '@vitrum/pt-webgpu: primitive "opaque-map" has an emissiveMap without CPU-readable texels; implicit mesh-area NEE uses scalar emissive radiance only.',
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
