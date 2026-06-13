/**
 * Regression tests for the DDGI emitter single-count fix and mesh-area probe NEE.
 *
 * Context (2026-06-10):
 *
 * DOUBLE-COUNT BUG — rect/disc area emitters were being counted TWICE in DDGI:
 *   1. `coreEmittersToDDGILights` mapped them to `kind:'fixture'` point-proxy
 *      DDGILights → evaluated as `direct_analytic` in probeUpdateRays.wgsl.
 *   2. `collectRectAreaEmitterTrisFromCore` also tessellated the same emitters
 *      into triangles → evaluated as `direct_emitter` via `ddgiEmitterNEE`.
 *   Both paths are summed on line 587: `let direct = direct_analytic + direct_emitter`.
 *
 * FIX — removed the rect-area and disc-area → fixture point-proxy mapping from
 * `coreEmittersToDDGILights`. The NEE triangle path is the physically-correct
 * one and is already present. Point and spot fixtures are unaffected.
 *
 * MESH-AREA FIX — mesh-area emitter triangles were excluded from
 * `collectRectAreaEmitterTrisFromCore`, so emissive-mesh-lit scenes produced
 * zero DDGI probe direct lighting. Extended to include them (2026-06-10).
 *
 * COMMENT FIX — the `mesh-area → null` case comment falsely claimed emissive
 * geometry reached probes via "probe rays hit mat.emissive"; probeUpdateRays.wgsl
 * does not read mat.emissive on BVH hits. Corrected to reference the H18 NEE path.
 */

import { describe, it, expect, vi } from 'vitest';
import type { Scene, RectAreaEmitter, DiscAreaEmitter, PointEmitter } from '@vitrum/core';
import { coreEmitterToDDGILight, coreEmittersToDDGILights } from '../src/coreEmittersToDDGILights.js';
import {
  collectRectAreaEmitterTrisFromCore,
  collectMeshAreaEmitterTrisFromCore,
  packEmitterTrisForDDGI,
} from '../src/restir/bvhSceneHelpers.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

function makeScene(...emitters: Scene['emitters']): Scene {
  return { primitives: [], emitters, environment: { kind: 'none' } };
}

const RECT_AREA: RectAreaEmitter = {
  kind: 'rect-area',
  id: 'rect1',
  color: [1, 0, 0],
  intensity: 5,
  position: [0, 2, 0],
  uAxis: [1, 0, 0],
  vAxis: [0, 0, 1],
};

const DISC_AREA: DiscAreaEmitter = {
  kind: 'disc-area',
  id: 'disc1',
  color: [0, 1, 0],
  intensity: 3,
  position: [0, 2, 0],
  normal: [0, 1, 0],
  radius: 0.5,
};

const POINT: PointEmitter = {
  kind: 'point',
  id: 'pt1',
  color: [1, 1, 1],
  intensity: 10,
  position: [1, 2, 3],
};

// ─── Single-count: rect/disc must NOT produce a fixture in coreEmittersToDDGILights ──

describe('coreEmittersToDDGILights — no fixture for rect/disc area emitters', () => {
  it('rect-area emitter produces null from coreEmitterToDDGILight (no fixture proxy)', () => {
    // rect/disc fixture-proxy removed: was double-counted with H18 NEE, 2026-06-10
    const result = coreEmitterToDDGILight(RECT_AREA);
    expect(result).toBeNull();
  });

  it('disc-area emitter produces null from coreEmitterToDDGILight (no fixture proxy)', () => {
    // rect/disc fixture-proxy removed: was double-counted with H18 NEE, 2026-06-10
    const result = coreEmitterToDDGILight(DISC_AREA);
    expect(result).toBeNull();
  });

  it('coreEmittersToDDGILights contains no fixture for rect-area kinds', () => {
    // This is the pinned regression test: a scene with one rect-area emitter must
    // produce ZERO fixtures from coreEmittersToDDGILights. The emitter reaches
    // DDGI through ddgiEmitterNEE (H18 NEE tris), not as an analytic fixture.
    const lights = coreEmittersToDDGILights(makeScene(RECT_AREA));
    const fixtures = lights.filter((l) => l.kind === 'fixture');
    expect(fixtures).toHaveLength(0);
  });

  it('coreEmittersToDDGILights contains no fixture for disc-area kinds', () => {
    const lights = coreEmittersToDDGILights(makeScene(DISC_AREA));
    const fixtures = lights.filter((l) => l.kind === 'fixture');
    expect(fixtures).toHaveLength(0);
  });

  it('point emitters still produce a fixture (unaffected by the fix)', () => {
    const lights = coreEmittersToDDGILights(makeScene(POINT));
    const fixtures = lights.filter((l) => l.kind === 'fixture');
    expect(fixtures).toHaveLength(1);
    expect(fixtures[0]!.intensity).toBe(10);
  });

  it('preserves castShadow:false on directional, point, and spot analytic lights', () => {
    const lights = coreEmittersToDDGILights(makeScene(
      {
        kind: 'directional',
        id: 'sun-no-shadow',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 1,
        castShadow: false,
      },
      {
        ...POINT,
        id: 'point-no-shadow',
        castShadow: false,
      },
      {
        kind: 'spot',
        id: 'spot-no-shadow',
        position: [0, 1, 0],
        direction: [0, -1, 0],
        angle: Math.PI / 6,
        penumbra: 0.25,
        color: [1, 1, 1],
        intensity: 2,
        castShadow: false,
      },
    ));
    expect(lights.map((l) => l.castShadow)).toEqual([false, false, false]);
  });

  it('mixed scene: rect-area + point produces exactly one fixture (the point)', () => {
    const lights = coreEmittersToDDGILights(makeScene(RECT_AREA, POINT));
    const fixtures = lights.filter((l) => l.kind === 'fixture');
    expect(fixtures).toHaveLength(1);
    expect(fixtures[0]!.id).toBe('pt1');
  });
});

// ─── mesh-area tris appear in the packed emitter-tri list ────────────────────

describe('collectMeshAreaEmitterTrisFromCore — mesh-area tris added to probe NEE', () => {
  it('mesh-area emitter triangles appear in the packed emitter-tri list', () => {
    // mesh-area tris added to probe NEE, 2026-06-10
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'panel',
          // Two triangles forming a 2×1 rectangle at z=0.
          positions: new Float32Array([
            0, 0, 0,
            2, 0, 0,
            2, 1, 0,
            0, 1, 0,
          ]),
          normals: new Float32Array([
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
          ]),
          indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
          material: { baseColor: [0, 0, 0], roughness: 1, metallic: 0 },
        },
      ],
      emitters: [
        {
          kind: 'mesh-area',
          id: 'panel-emitter',
          meshId: 'panel',
          color: [2, 0.5, 0.1],
          intensity: 4,
        },
      ],
      environment: { kind: 'none' },
    };

    const tris = collectMeshAreaEmitterTrisFromCore(scene);
    // The panel has 2 triangles indexed; both should appear.
    expect(tris).toHaveLength(2);

    // Le = color * intensity = [8, 2, 0.4]
    const expectedLe: [number, number, number] = [8, 2, 0.4];
    for (const tri of tris) {
      expect(tri.Le[0]).toBeCloseTo(expectedLe[0], 5);
      expect(tri.Le[1]).toBeCloseTo(expectedLe[1], 5);
      expect(tri.Le[2]).toBeCloseTo(expectedLe[2], 5);
    }

    // Also verify packEmitterTrisForDDGI returns count=2.
    const packed = packEmitterTrisForDDGI(tris);
    expect(packed.count).toBe(2);
  });

  it('mesh-area emitter with identity transform: world vertices equal local vertices', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'tri',
          positions: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          indices: new Uint32Array([0, 1, 2]),
          material: { baseColor: [0, 0, 0], roughness: 1, metallic: 0 },
        },
      ],
      emitters: [
        {
          kind: 'mesh-area',
          id: 'e',
          meshId: 'tri',
          color: [1, 1, 1],
          intensity: 1,
        },
      ],
      environment: { kind: 'none' },
    };

    const tris = collectMeshAreaEmitterTrisFromCore(scene);
    expect(tris).toHaveLength(1);
    const tri = tris[0]!;
    // Without a transform, world vertices should match local positions.
    expect(tri.vA[0]).toBeCloseTo(1, 5);
    expect(tri.vA[1]).toBeCloseTo(0, 5);
    expect(tri.vB[1]).toBeCloseTo(1, 5);
    expect(tri.vC[2]).toBeCloseTo(1, 5);
  });

  it('mesh-area emitter not in scene.primitives: emits a console.warn and produces no tris', () => {
    const warnMock = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const scene: Scene = {
      primitives: [],
      emitters: [
        {
          kind: 'mesh-area',
          id: 'ghost',
          meshId: 'nonexistent',
          color: [1, 1, 1],
          intensity: 1,
        },
      ],
      environment: { kind: 'none' },
    };

    const tris = collectMeshAreaEmitterTrisFromCore(scene);
    expect(tris).toHaveLength(0);
    expect(warnMock).toHaveBeenCalledOnce();
    expect(warnMock.mock.calls[0]![0]).toContain('nonexistent');
    warnMock.mockRestore();
  });

  it('collectRectAreaEmitterTrisFromCore does NOT include mesh-area tris (ReSTIR safety)', () => {
    // Pin that the ReSTIR function does not expand mesh-area (would double-count
    // with the merged geometry stream). The DDGI path uses collectMeshAreaEmitterTrisFromCore
    // separately. (2026-06-10)
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'panel',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          indices: new Uint32Array([0, 1, 2]),
          material: { baseColor: [0, 0, 0], roughness: 1, metallic: 0 },
        },
      ],
      emitters: [
        {
          kind: 'mesh-area',
          id: 'e',
          meshId: 'panel',
          color: [1, 1, 1],
          intensity: 1,
        },
      ],
      environment: { kind: 'none' },
    };
    const tris = collectRectAreaEmitterTrisFromCore(scene);
    expect(tris).toHaveLength(0); // mesh-area must NOT appear in the ReSTIR stream
  });

  it('rect-area emitter still produces 2 tris from collectRectAreaEmitterTrisFromCore', () => {
    const tris = collectRectAreaEmitterTrisFromCore(makeScene(RECT_AREA));
    // rect-area → 2 triangles (LL-LR-UR and LL-UR-UL)
    expect(tris).toHaveLength(2);
    // Le = color * intensity = [5, 0, 0]
    for (const tri of tris) {
      expect(tri.Le[0]).toBeCloseTo(5, 5);
      expect(tri.Le[1]).toBeCloseTo(0, 5);
      expect(tri.Le[2]).toBeCloseTo(0, 5);
    }
  });
});
