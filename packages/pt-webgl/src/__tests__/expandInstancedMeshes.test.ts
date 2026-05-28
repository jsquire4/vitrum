/**
 * Unit tests for pt-webgl's instanced-mesh expansion.
 *
 * The shared `vitrumSceneToThree` builds ONE `THREE.InstancedMesh` per
 * `instanced-mesh` primitive (right for walkaround's TLAS path). pt-webgl's
 * `setScene` then expands it into N baked `THREE.Mesh` instances, because the
 * absorbed three-gpu-pathtracer fork's geometry generator bakes only
 * `mesh.matrixWorld` and ignores `instanceMatrix`. These tests pin:
 *   1. an N-instance instanced-mesh expands to N standalone meshes at the
 *      correct per-instance world transforms (real THREE objects — no GPU);
 *   2. the InstancedMesh is removed from the scene and replaced by the N
 *      children, which share the single geometry + material;
 *   3. plain meshes / scenes with no instanced-mesh are untouched;
 *   4. the capability getter now declares `instanced-mesh` and matches the
 *      core promiseLedger pt-webgl row.
 */
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { InstancedMesh as TInstancedMesh, Matrix4, Mesh, Scene } from 'three';
import type { Mesh as TMesh } from 'three';
import { asMat4, BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import type { InstancedMeshPrimitive, Scene as VitrumScene } from '@vitrum/core';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import {
  expandInstancedMesh,
  expandInstancedMeshesInScene,
  findAllMeshesByPrimitiveId,
} from '../expandInstancedMeshes.js';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    readonly target = { texture: {} };
    samples = 0;
    bounces = 0;
    transmissiveBounces = 0;
    filterGlossyFactor = 0;
    renderDelay = 0;
    minSamples = 0;
    dynamicLowRes = false;
    multipleImportanceSampling = false;
    tileRepeatFactors: Uint8Array | null = null;
    configureAdditiveAccumulation = vi.fn();
    readonly tiles = { set: vi.fn() };
    readonly _pathTracer = { material: { uniforms: {} } };
    setScene(): void {}
    setCamera(): void {}
    renderSample(): void { this.samples += 1; }
    reset(): void {}
    dispose(): void {}
    updateEnvironment(): void {}
  }
  return { WebGLPathTracer };
});

/** A single unit triangle in object space (so a world transform is observable
 *  as a pure translation/scale of the centroid). */
const TRI_POSITIONS = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const TRI_NORMALS = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);

/** Column-major translation matrix `[x,y,z]` (THREE / vitrum Mat4 layout). */
function translation(x: number, y: number, z: number): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

/** Column-major uniform-scale matrix. */
function scale(s: number): Float32Array {
  // prettier-ignore
  return new Float32Array([
    s, 0, 0, 0,
    0, s, 0, 0,
    0, 0, s, 0,
    0, 0, 0, 1,
  ]);
}

function instancedPrim(
  id: string,
  instances: Float32Array[],
): InstancedMeshPrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: TRI_POSITIONS,
    normals: TRI_NORMALS,
    material: { baseColor: [0.4, 0.5, 0.6], roughness: 0.5, metallic: 0 },
    instances: instances.map((m) => asMat4(m)),
  };
}

function collectMeshes(scene: Scene): TMesh[] {
  const meshes: TMesh[] = [];
  scene.traverse((o) => {
    if ((o as TMesh).isMesh === true && (o as TInstancedMesh).isInstancedMesh !== true) {
      meshes.push(o as TMesh);
    }
  });
  return meshes;
}

function decompose(mesh: TMesh): { x: number; y: number; z: number; sx: number } {
  const pos = new Matrix4().copy(mesh.matrixWorld);
  const e = pos.elements;
  // Column-major: translation = elements[12..14]; uniform scale from |col 0|.
  const sx = Math.hypot(e[0]!, e[1]!, e[2]!);
  return { x: e[12]!, y: e[13]!, z: e[14]!, sx };
}

describe('expandInstancedMesh (unit, no GPU)', () => {
  it('expands an N-instance THREE.InstancedMesh into N meshes at the per-instance world transforms', () => {
    const instances = [translation(0, 0, 0), translation(2, 0, 0), translation(0, 3, 0)];
    const vscene: VitrumScene = {
      primitives: [instancedPrim('inst-a', instances)],
      emitters: [],
      environment: { kind: 'none' },
    };

    const threeScene = vitrumSceneToThree(vscene);
    // Exactly one InstancedMesh before expansion.
    let instancedCount = 0;
    threeScene.traverse((o) => {
      if ((o as TInstancedMesh).isInstancedMesh === true) instancedCount += 1;
    });
    expect(instancedCount).toBe(1);

    expandInstancedMeshesInScene(threeScene);

    // No InstancedMesh remains; N standalone meshes were added.
    let instancedAfter = 0;
    threeScene.traverse((o) => {
      if ((o as TInstancedMesh).isInstancedMesh === true) instancedAfter += 1;
    });
    expect(instancedAfter).toBe(0);

    const meshes = collectMeshes(threeScene);
    expect(meshes).toHaveLength(3);

    const worlds = meshes
      .map(decompose)
      .map((d) => `${d.x},${d.y},${d.z}`)
      .sort();
    expect(worlds).toEqual(['0,0,0', '0,3,0', '2,0,0'].sort());
  });

  it('bakes a non-translation (scaled) instance matrix into matrixWorld', () => {
    const im = new TInstancedMesh(undefined as never, undefined as never, 1);
    const s = new Matrix4().fromArray(Array.from(scale(2.5)));
    im.setMatrixAt(0, s);
    im.instanceMatrix.needsUpdate = true;
    im.updateMatrixWorld(true);

    const children = expandInstancedMesh(im);
    expect(children).toHaveLength(1);
    const d = decompose(children[0]!);
    expect(d.sx).toBeCloseTo(2.5, 5);
  });

  it('composes the InstancedMesh node transform with each instance matrix', () => {
    // InstancedMesh placed at world (10,0,0); two instances at local +0 and +5 in x.
    const im = new TInstancedMesh(undefined as never, undefined as never, 2);
    im.matrixAutoUpdate = false;
    im.matrix.copy(new Matrix4().fromArray(Array.from(translation(10, 0, 0))));
    im.matrixWorld.copy(im.matrix);
    im.setMatrixAt(0, new Matrix4().fromArray(Array.from(translation(0, 0, 0))));
    im.setMatrixAt(1, new Matrix4().fromArray(Array.from(translation(5, 0, 0))));
    im.instanceMatrix.needsUpdate = true;

    const children = expandInstancedMesh(im);
    const xs = children.map((c) => decompose(c).x).sort((a, b) => a - b);
    expect(xs[0]).toBeCloseTo(10, 5);
    expect(xs[1]).toBeCloseTo(15, 5);
  });

  it('shares the single geometry + material across expanded children', () => {
    const im = new TInstancedMesh(undefined as never, undefined as never, 3);
    for (let i = 0; i < 3; i += 1) im.setMatrixAt(i, new Matrix4());
    im.instanceMatrix.needsUpdate = true;
    im.updateMatrixWorld(true);

    const children = expandInstancedMesh(im);
    expect(children).toHaveLength(3);
    for (const c of children) {
      expect(c.geometry).toBe(im.geometry);
      expect(c.material).toBe(im.material);
      expect(c.name).toBe(im.name);
      expect(c.matrixAutoUpdate).toBe(false);
    }
  });

  it('findAllMeshesByPrimitiveId returns all N expanded children sharing the primitive id', () => {
    const instances = [translation(0, 0, 0), translation(1, 0, 0), translation(2, 0, 0)];
    const vscene: VitrumScene = {
      primitives: [instancedPrim('inst-mat', instances)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const threeScene = vitrumSceneToThree(vscene);
    expandInstancedMeshesInScene(threeScene);

    const matched = findAllMeshesByPrimitiveId(threeScene, 'inst-mat');
    expect(matched).toHaveLength(3);
    // Re-pointing the material on every child is what the engine's material
    // fast path relies on (each holds its own mesh.material reference).
    for (const m of matched) {
      m.material = new Mesh().material;
    }
    // All three remain resolvable by id (no child was lost).
    expect(findAllMeshesByPrimitiveId(threeScene, 'inst-mat')).toHaveLength(3);
  });

  it('leaves a scene with no InstancedMesh untouched', () => {
    const scene = new Scene();
    const plain = new Mesh();
    plain.name = 'plain';
    scene.add(plain);
    expandInstancedMeshesInScene(scene);
    const meshes = collectMeshes(scene);
    expect(meshes).toHaveLength(1);
    expect(meshes[0]!.name).toBe('plain');
  });
});

describe('pt-webgl instanced-mesh capability + ledger', () => {
  let teardown: (() => void) | null = null;
  beforeAll(() => {
    teardown = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardown?.();
    teardown = null;
  });

  it('declares instanced-mesh in supportedPrimitiveKinds', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    expect(engine.capabilities.supportedPrimitiveKinds!.has('instanced-mesh')).toBe(true);
  });

  it('matches the core promiseLedger pt-webgl row (instanced-mesh present)', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    const declared = Array.from(engine.capabilities.supportedPrimitiveKinds ?? []).sort();
    const ledger = [...BACKEND_PROMISE_LEDGER['pt-webgl'].supportedPrimitiveKinds].sort();
    expect(declared).toEqual(ledger);
    expect(ledger).toContain('instanced-mesh');
  });
});
