import { describe, expect, it } from 'vitest';
import type { Mat4, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  packSceneFromCore,
  rebuildTlasReuseBlas,
  refitTlasTransforms,
  type ScenePackResult,
  type TlasGpuSnapshot,
} from '../scenePack.js';

// ─────────────────────────────────────────────────────────────────────────────
// R5 / V2-3 characterization: the initial pack SKIPS non-invertible (singular)
// TLAS instances (scenePack.ts H34-e), but the instance-count rebuild path used
// to INSERT them at identity. The two build paths must agree: a singular instance
// is skipped on BOTH paths, and the per-primitive `instanceCount` bookkeeping must
// reflect actual TLAS membership (invertible instances only), not the raw
// transform count.
// ─────────────────────────────────────────────────────────────────────────────

function translate(x: number): Mat4 {
  return asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1]));
}

/** A zero-first-column matrix is singular (det=0) → invertMat4 returns null. */
function singular(x = 0): Mat4 {
  return asMat4(new Float32Array([
    0, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, 0, 0, 1,
  ]));
}

function instancedMesh(id: string, instances: Mat4[]): Scene['primitives'][number] {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    instances,
  };
}

function sceneWithInstances(instances: Mat4[]): Scene {
  return {
    primitives: [instancedMesh('inst', instances)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function tlasSnapshot(pack: ScenePackResult): TlasGpuSnapshot {
  return {
    tlasNodes: pack.tlasNodes,
    tlasInstanceIndices: pack.tlasInstanceIndices,
    tlasBlasRoots: pack.tlasBlasRoots,
    tlasInstanceWorldToLocal: pack.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: pack.tlasInstanceLocalToWorld,
  };
}

describe('rebuildTlasReuseBlas — singular-instance skip parity (R5 / V2-3)', () => {
  it('initial pack skips singular instances and counts only TLAS-inserted instances', () => {
    // Two invertible + one singular → 2 real TLAS instances.
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), singular(), translate(4)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });

    // Only the two invertible instances end up in the TLAS.
    expect(packed.tlasBlasRoots.length).toBe(2);
    // instanceCount reflects actual TLAS membership, NOT the raw 3-transform count.
    expect(packed.primitiveTlasBindings[0]?.instanceCount).toBe(2);
    expect(packed.primitiveTlasBindings[0]?.instanceSourceIndices).toEqual([0, 2]);
  });

  it('world bounds union only the exact instances admitted to the TLAS', () => {
    const scene = sceneWithInstances([
      translate(0),
      singular(100),
      translate(4),
    ]);
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });

    const bounds = computeWorldAabbForBindings(scene, packed.primitiveTlasBindings);
    expect(bounds).not.toBeNull();
    expect(bounds?.min[0]).toBe(0);
    expect(bounds?.max[0]).toBe(5);
  });

  it('rebuild path MIRRORS the skip — a singular instance never appears at identity', () => {
    // Start: two invertible instances (both real TLAS members).
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    expect(packed.primitiveTlasBindings[0]?.instanceCount).toBe(2);

    // Membership change: grow to THREE invertible instances AND append a singular
    // one. A full repack skips the singular → 3 real TLAS members. The rebuild
    // path must agree: 3 members, singular never inserted at identity.
    const next: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(2), translate(4), singular()])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const full = packSceneFromCore(next, { tlas: true, resolveMaterialId: () => 0 });
    // Full repack: singular skipped → 3 real TLAS instances (not 4).
    expect(full.tlasBlasRoots.length).toBe(3);
    expect(full.primitiveTlasBindings[0]?.instanceCount).toBe(3);

    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;

    // The rebuild path must NOT insert the singular instance at identity.
    // (Old behavior: 4 blasRoots with an identity fallback for the singular one.)
    expect(rebuilt.pack.tlasBlasRoots.length).toBe(full.tlasBlasRoots.length);
    expect(rebuilt.pack.tlasBlasRoots.length).toBe(3);
    // instanceCount matches actual membership on both paths.
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceCount).toBe(3);
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceCount).toBe(
      full.primitiveTlasBindings[0]?.instanceCount,
    );
    // No identity-fallback transform snuck in: the three inserted local→world
    // matrices are exactly the pure translations at x=0, x=2, x=4 (the singular
    // instance, whose identity fallback would translate to x=0, is absent).
    const xs = [
      rebuilt.pack.tlasInstanceLocalToWorld[0 * 16 + 12],
      rebuilt.pack.tlasInstanceLocalToWorld[1 * 16 + 12],
      rebuilt.pack.tlasInstanceLocalToWorld[2 * 16 + 12],
    ].sort((a, b) => (a ?? 0) - (b ?? 0));
    expect(xs).toEqual([0, 2, 4]);
  });

  it('rebuild removing a singular instance stays consistent with a full repack', () => {
    // Start with a singular instance already present in the mix.
    const scene: Scene = {
      primitives: [instancedMesh('inst', [translate(0), singular(), translate(4)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const packed = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    // 2 real TLAS members despite 3 transforms.
    expect(packed.primitiveTlasBindings[0]?.instanceCount).toBe(2);

    // Drop the singular instance → 2 transforms, both invertible.
    const next: Scene = {
      primitives: [instancedMesh('inst', [translate(0), translate(4)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    // The count remains 2, but raw membership changes from [0,2] to [0,1].
    // A membership-aware TLAS rebuild must not mistake that for a stable refit.
    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceCount).toBe(2);
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceSourceIndices).toEqual([0, 1]);
    expect(rebuilt.pack.bvhNodes).toBe(packed.bvhNodes);
    expect(rebuilt.pack.positions).toBe(packed.positions);
  });
});

describe('refitTlasTransforms — exact singular-instance membership', () => {
  it('refits a stable skipped-singular set without inserting an identity fallback', () => {
    const initial = sceneWithInstances([translate(0), singular(), translate(4)]);
    const packed = packSceneFromCore(initial, { tlas: true, resolveMaterialId: () => 0 });
    const baselineNodes = Array.from(packed.tlasNodes);
    const baselineWorldToLocal = Array.from(packed.tlasInstanceWorldToLocal);

    const next = sceneWithInstances([translate(1), singular(), translate(5)]);
    const refit = refitTlasTransforms(
      next,
      packed.primitiveTlasBindings,
      tlasSnapshot(packed),
      { primitiveId: 'inst' },
    );

    expect(refit.ok).toBe(true);
    if (!refit.ok) return;
    expect(refit.tlasBlasRoots).toHaveLength(2);
    expect([
      refit.tlasInstanceLocalToWorld[12],
      refit.tlasInstanceLocalToWorld[16 + 12],
    ]).toEqual([1, 5]);
    expect(Array.from(refit.dirtyTlasTransformInstanceIndices ?? [])).toEqual([0, 1]);
    // Partial refit treats the supplied snapshot as a rollback baseline.
    expect(Array.from(packed.tlasNodes)).toEqual(baselineNodes);
    expect(Array.from(packed.tlasInstanceWorldToLocal)).toEqual(baselineWorldToLocal);
  });

  it('rejects singular-to-invertible membership growth', () => {
    const next = sceneWithInstances([translate(0), translate(2), translate(4)]);
    const packed = packSceneFromCore(
      sceneWithInstances([translate(0), singular(), translate(4)]),
      { tlas: true, resolveMaterialId: () => 0 },
    );
    const refit = refitTlasTransforms(
      next,
      packed.primitiveTlasBindings,
      tlasSnapshot(packed),
      { primitiveId: 'inst' },
    );

    expect(refit.ok).toBe(false);
    if (refit.ok) return;
    expect(refit.reason).toContain('membership changed from [0,2] to [0,1,2]');

    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceSourceIndices).toEqual([0, 1, 2]);
    expect(rebuilt.pack.tlasBlasRoots).toHaveLength(3);
    expect(rebuilt.pack.bvhNodes).toBe(packed.bvhNodes);
  });

  it('rejects invertible-to-singular membership shrinkage', () => {
    const next = sceneWithInstances([translate(0), singular(), translate(4)]);
    const packed = packSceneFromCore(
      sceneWithInstances([translate(0), translate(2), translate(4)]),
      { tlas: true, resolveMaterialId: () => 0 },
    );
    const refit = refitTlasTransforms(
      next,
      packed.primitiveTlasBindings,
      tlasSnapshot(packed),
      { primitiveId: 'inst' },
    );

    expect(refit.ok).toBe(false);
    if (refit.ok) return;
    expect(refit.reason).toContain('membership changed from [0,1,2] to [0,2]');

    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceSourceIndices).toEqual([0, 2]);
    expect(rebuilt.pack.tlasBlasRoots).toHaveLength(2);
    expect(rebuilt.pack.bvhNodes).toBe(packed.bvhNodes);
  });

  it('rejects a same-count membership swap, then rebuilds TLAS while reusing BLAS', () => {
    const initial = sceneWithInstances([translate(0), singular(), translate(4)]);
    const packed = packSceneFromCore(initial, { tlas: true, resolveMaterialId: () => 0 });
    const next = sceneWithInstances([singular(), translate(2), translate(4)]);

    const refit = refitTlasTransforms(
      next,
      packed.primitiveTlasBindings,
      tlasSnapshot(packed),
      { primitiveId: 'inst' },
    );
    expect(refit.ok).toBe(false);
    if (refit.ok) return;
    expect(refit.reason).toContain('membership changed from [0,2] to [1,2]');

    const rebuilt = rebuildTlasReuseBlas(next, packed);
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceCount).toBe(2);
    expect(rebuilt.pack.primitiveTlasBindings[0]?.instanceSourceIndices).toEqual([1, 2]);
    expect(rebuilt.pack.bvhNodes).toBe(packed.bvhNodes);
    expect(rebuilt.pack.positions).toBe(packed.positions);
    expect([
      rebuilt.pack.tlasInstanceLocalToWorld[12],
      rebuilt.pack.tlasInstanceLocalToWorld[16 + 12],
    ]).toEqual([2, 4]);
  });
});
