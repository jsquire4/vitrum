import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';
import {
  isGeometryOnlyPrimitivePatch,
  isMaterialOnlyPrimitivePatch,
  applyGeometryPatchToMesh,
} from '../scenePatch.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

const setScene = vi.fn();
const reset = vi.fn();
const updateMaterials = vi.fn();
// The fork's targeted geometry+BVH regen entry. `regenerateSceneGeometry`
// invokes `_generator.generate()`; observing this proves the incremental path
// ran instead of a full `setScene` teardown.
const generate = vi.fn(() => ({
  bvhChanged: true,
  bvh: { mock: true },
  needsMaterialIndexUpdate: false,
  geometry: { attributes: { normal: { array: [] } } },
}));

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

// A geometry stub that records the buffers written to it so we can assert the
// new (longer) vertex array landed on the THREE mesh. `getAttribute('position')`
// reflects whatever was last `setAttribute`-d so the same-count fast-path guard
// (`applyPositionsPatchToMesh`) reads a real count if it ever runs.
function makeGeometryStub() {
  const attrs: Record<string, { count: number; array: ArrayLike<number> }> = {
    position: { count: 3, array: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) },
  };
  let index: { count: number; array: ArrayLike<number> } | null = null;
  return {
    attrs,
    get index() {
      return index;
    },
    getAttribute: (name: string) => attrs[name],
    setAttribute: vi.fn((name: string, attr: { count: number; array: ArrayLike<number> }) => {
      attrs[name] = attr;
    }),
    setIndex: vi.fn((attr: { count: number; array: ArrayLike<number> }) => {
      index = attr;
    }),
  };
}

const meshGeometry = makeGeometryStub();
const meshStub = {
  isMesh: true,
  name: 'mesh-a',
  uuid: 'mesh-a',
  geometry: meshGeometry,
  material: {},
};

function makeStubRoot() {
  return {
    updateMatrixWorld: vi.fn(),
    traverse: (cb: (o: unknown) => void) => cb(meshStub),
    traverseVisible: (cb: (o: unknown) => void) => cb(meshStub),
  };
}

vi.mock('@vitrum/three-bindings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/three-bindings')>();
  return {
    ...actual,
    vitrumSceneToThree: vi.fn(() => makeStubRoot()),
    findMeshByPrimitiveId: vi.fn(() => meshStub),
    applyVitrumMaterialToMesh: vi.fn(),
  };
});

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
    readonly _pathTracer = {
      material: {
        bvh: { updateFrom: vi.fn() },
        attributesArray: { updateFrom: vi.fn() },
        materialIndexAttribute: { updateFrom: vi.fn() },
        uniforms: {},
      },
    };
    readonly _generator = { initialized: true, generate };

    setScene = setScene;
    setCamera(): void {}
    renderSample(): void {
      this.samples += 1;
    }
    reset = reset;
    dispose(): void {}
    updateEnvironment(): void {}
    updateMaterials = updateMaterials;
  }

  return { WebGLPathTracer };
});

describe('scenePatch geometry-topology classifier + mutator', () => {
  it('classifies a vertex-count change (positions only) as geometry-only', () => {
    expect(
      isGeometryOnlyPrimitivePatch({
        positions: new Float32Array(12), // 4 verts vs prior 3 — a COUNT change
      }),
    ).toBe(true);
  });

  it('classifies an index-only change as geometry-only', () => {
    expect(isGeometryOnlyPrimitivePatch({ indices: new Uint32Array([0, 1, 2, 0, 2, 3]) })).toBe(true);
  });

  it('classifies a positions+normals+uvs+indices surgery as geometry-only', () => {
    expect(
      isGeometryOnlyPrimitivePatch({
        positions: new Float32Array(12),
        normals: new Float32Array(12),
        uvs: new Float32Array(8),
        indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
      }),
    ).toBe(true);
  });

  it('does NOT classify a patch that also touches material as geometry-only', () => {
    const patch = {
      positions: new Float32Array(12),
      material: { baseColor: [1, 0, 0] as [number, number, number], roughness: 0.5, metallic: 0 },
    };
    expect(isGeometryOnlyPrimitivePatch(patch)).toBe(false);
    // ...and it's not material-only either (carries geometry), so the engine
    // routes it to the full-rebuild fallthrough.
    expect(isMaterialOnlyPrimitivePatch(patch)).toBe(false);
  });

  it('does NOT classify a patch that also touches transform as geometry-only', () => {
    expect(
      isGeometryOnlyPrimitivePatch({
        positions: new Float32Array(12),
        transform: new Float32Array(16) as never,
      }),
    ).toBe(false);
  });

  it('rejects an empty/id-only patch (nothing geometric)', () => {
    expect(isGeometryOnlyPrimitivePatch({})).toBe(false);
    expect(isGeometryOnlyPrimitivePatch({ id: 'x' })).toBe(false);
  });

  it('applyGeometryPatchToMesh writes a larger position buffer (count change) + index', () => {
    const geom = makeGeometryStub();
    const mesh = { geometry: geom } as never;
    const applied = applyGeometryPatchToMesh(mesh, {
      positions: new Float32Array(12), // 4 verts
      normals: new Float32Array(12),
      uvs: new Float32Array(8),
      tangents: new Float32Array(16),
      indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
    });
    expect(applied).toBe(true);
    expect(geom.attrs.position!.count).toBe(4);
    expect(geom.attrs.normal!.array.length).toBe(12);
    expect(geom.attrs.uv!.array.length).toBe(8);
    expect(geom.attrs.tangent!.array.length).toBe(16);
    expect(geom.index?.array).toBeInstanceOf(Uint16Array);
    // buffers are COPIED, not aliased: the position array written is a fresh
    // typed array, distinct from the source buffer handed to the mutator.
    const src = new Float32Array([9, 9, 9, 9, 9, 9]);
    const aliasGeom = makeGeometryStub();
    applyGeometryPatchToMesh({ geometry: aliasGeom } as never, { positions: src });
    expect(aliasGeom.attrs.position!.array).not.toBe(src);
  });
});

describe('PTEngineWebGL2.updatePrimitive vertex-count topology (PR-9)', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('takes the incremental fork regen path for a same-material vertex-COUNT change', async () => {
    setScene.mockClear();
    generate.mockClear();
    reset.mockClear();
    updateMaterials.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({
      primitives: [
        {
          id: 'mesh-a',
          kind: 'mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    });

    setScene.mockClear();
    generate.mockClear();
    meshGeometry.setAttribute.mockClear();
    meshGeometry.setIndex.mockClear();

    // Grow the mesh from 3 verts (1 tri) to 4 verts (2 tris via an index).
    engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });

    // Incremental fork regen ran (NOT a full setScene teardown).
    expect(generate).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    // The new geometry landed on the mesh.
    expect(meshGeometry.setAttribute).toHaveBeenCalled();
    expect(meshGeometry.setIndex).toHaveBeenCalledTimes(1);
    expect(meshGeometry.attrs.position!.count).toBe(4);
    // The accumulator was cleared via the regen's reset callback.
    expect(reset).toHaveBeenCalled();
    // No material re-pack on the geometry-only path.
    expect(updateMaterials).not.toHaveBeenCalled();
  });

  it('falls back to full setScene when the patch ALSO changes the material', async () => {
    setScene.mockClear();
    generate.mockClear();
    reset.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({
      primitives: [
        {
          id: 'mesh-a',
          kind: 'mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    });

    setScene.mockClear();
    generate.mockClear();

    // Geometry COUNT change + a NEW material in the same patch: the
    // geometry-only regen skips updateMaterials(), so this MUST full-rebuild.
    engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
      material: { baseColor: [0.9, 0.1, 0.1], roughness: 0.2, metallic: 1 },
    });

    expect(setScene).toHaveBeenCalledTimes(1);
    // The incremental regen path did NOT run for this patch.
    expect(generate).not.toHaveBeenCalled();
  });
});
