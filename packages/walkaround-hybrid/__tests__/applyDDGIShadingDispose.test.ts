/**
 * Bug regression — `disposeApplyDDGIShadingCache()` must reset `_injectedMaterials`.
 *
 * Prior to the fix `_injectedMaterials` was declared `const`, so dispose() left
 * the WeakMap intact. On multi-engine / test-reuse scenarios where the same
 * THREE.Mesh object survived across engine cycles, `_injectedMaterials.has(mesh)`
 * short-circuited and skipped re-injection on the next cycle.
 *
 * The fix: `let _injectedMaterials = new WeakMap(...)` + reset in dispose().
 * This test pins that behavioral contract.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';

// upgradeSpy must be hoisted so it is initialised BEFORE vi.mock factories run
// (vi.mock calls are hoisted to the top of the file by vitest — any non-hoisted
// variable referenced inside a factory causes a "Cannot access before init" error).
const { upgradeSpy } = vi.hoisted(() => ({
  upgradeSpy: vi.fn((mat: unknown) => ({
    isNodeMaterial: true as const,
    outputNode: null as unknown,
    needsUpdate: false,
    dispose: vi.fn(),
    _originalMat: mat,
  })),
}));

// ---------------------------------------------------------------------------
// Mock three/webgpu — StorageTexture is the only class used by the module
// under test at module scope (via `slotToStorageTexture`). We provide a
// minimal stand-in with a `dispose` spy so tests can verify teardown.
// ---------------------------------------------------------------------------
vi.mock('three/webgpu', () => {
  class MockStorageTexture {
    format = 0;
    type = 0;
    minFilter = 0;
    magFilter = 0;
    needsUpdate = false;
    dispose = vi.fn();
    constructor(public width: number, public height: number) {}
  }
  return {
    StorageTexture: MockStorageTexture,
    MeshPhysicalNodeMaterial: class MockMeshPhysicalNodeMaterial {
      isNodeMaterial = true;
      outputNode: unknown = null;
      needsUpdate = false;
      dispose = vi.fn();
    },
    MeshStandardNodeMaterial: class MockMeshStandardNodeMaterial {
      isNodeMaterial = true;
      outputNode: unknown = null;
      needsUpdate = false;
      dispose = vi.fn();
    },
  };
});

// ---------------------------------------------------------------------------
// Mock three/tsl — all TSL node-builder functions used in applyDDGIShading.
// None of them need to do real work; they just need to return something
// chain-able so the material injection path doesn't throw.
// ---------------------------------------------------------------------------
vi.mock('three/tsl', () => {
  const noop = () => ({ __node: true });
  return {
    add:           noop,
    vec4:          noop,
    mul:           noop,
    output:        { __node: true },
    materialColor: { __node: true },
    uniform:       noop,
    texture:       noop,
    positionWorld: { __node: true },
    normalWorld:   { __node: true },
    wgslFn:        () => noop,
    renderOutput:  noop,
  };
});

// ---------------------------------------------------------------------------
// Mock the WGSL string (not meaningful in unit tests — just needs to exist).
// ---------------------------------------------------------------------------
vi.mock('../src/ddgi/ddgiSampleWgsl.js', () => ({
  DDGI_SAMPLE_WGSL: '/* mock */',
}));

// ---------------------------------------------------------------------------
// Mock upgradeToNodeMaterial — uses the hoisted upgradeSpy (declared above).
// ---------------------------------------------------------------------------
vi.mock('../src/lib/nodeMaterialUpgrade.js', () => ({
  upgradeToNodeMaterial: upgradeSpy,
}));

// ---------------------------------------------------------------------------
// Import SUT AFTER mocks are set up (vitest hoists vi.mock calls, but the
// dynamic import makes the ordering explicit and readable).
// ---------------------------------------------------------------------------
import { applyDDGIShading, disposeApplyDDGIShadingCache } from '../src/ddgi/applyDDGIShading.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal ProbeGrid stub that satisfies applyDDGIShading's guard checks. */
function makeProbeGrid() {
  return {
    irradianceA: {},   // truthy — satisfies `if (!probeGrid.irradianceA || ...)` guard
    irradianceB: {},
    irradianceReadTex: { width: 64, height: 64 },
    visibilityReadTex: { width: 64, height: 64 },
    params: {
      origin: { x: 0, y: 0, z: 0 },
      spacing: 1,
      dims:    { x: 4, y: 4, z: 4 },
      irradianceAtlasW: 256, irradianceAtlasH: 256,
      visibilityAtlasW: 256, visibilityAtlasH: 256,
    },
  } as unknown as import('../src/ddgi/probeGrid.js').ProbeGrid;
}

/** Create a minimal THREE.Scene containing a single MeshStandardMaterial mesh. */
function makeScene(): { scene: THREE.Scene; mesh: THREE.Mesh } {
  const scene = new THREE.Scene();
  const geom  = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  const mat  = new THREE.MeshStandardMaterial({ color: 0xffffff });
  const mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);
  return { scene, mesh };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

afterEach(() => {
  // Guarantee a clean module state between tests.
  disposeApplyDDGIShadingCache();
  upgradeSpy.mockClear();
});

describe('disposeApplyDDGIShadingCache — _injectedMaterials reset', () => {
  it('calls upgradeToNodeMaterial on first injection', () => {
    const { scene } = makeScene();
    applyDDGIShading(scene, makeProbeGrid());
    expect(upgradeSpy).toHaveBeenCalledTimes(1);
  });

  it('does NOT call upgradeToNodeMaterial on second applyDDGIShading for the same mesh (skip guard active)', () => {
    const { scene } = makeScene();
    const grid = makeProbeGrid();

    applyDDGIShading(scene, grid);
    upgradeSpy.mockClear();

    // Second call — same mesh, same engine cycle.
    applyDDGIShading(scene, grid);
    // The `_injectedMaterials.has(obj) && enabled` guard must have fired.
    expect(upgradeSpy).not.toHaveBeenCalled();
  });

  it('after disposeApplyDDGIShadingCache(), the SAME mesh is treated as NOT-yet-injected (re-injection fires)', () => {
    const { scene, mesh } = makeScene();
    const grid = makeProbeGrid();

    // Inject once — mesh is now recorded in _injectedMaterials.
    applyDDGIShading(scene, grid);
    expect(upgradeSpy).toHaveBeenCalledTimes(1);

    upgradeSpy.mockClear();

    // Capture the material currently on the mesh (the mock node-mat from cycle 1).
    const matBeforeReinjection = mesh.material;

    // Dispose resets the module state.
    disposeApplyDDGIShadingCache();

    // The mesh object itself still exists (simulating multi-engine reuse).
    // After dispose, applyDDGIShading must re-inject it.
    applyDDGIShading(scene, grid);

    // upgradeToNodeMaterial MUST have been called again — re-injection happened.
    expect(upgradeSpy).toHaveBeenCalledTimes(1);

    // The material passed to upgradeToNodeMaterial was whatever was on the mesh
    // BEFORE the re-injection (the mock node-mat from the first cycle).
    const calledWithMat = upgradeSpy.mock.calls[0]![0];
    expect(calledWithMat).toBe(matBeforeReinjection);
  });

  it('dispose is idempotent — calling it twice does not throw and leaves state clean', () => {
    const { scene } = makeScene();
    applyDDGIShading(scene, makeProbeGrid());

    expect(() => {
      disposeApplyDDGIShadingCache();
      disposeApplyDDGIShadingCache();
    }).not.toThrow();

    // After double-dispose a fresh injection still works.
    upgradeSpy.mockClear();
    applyDDGIShading(scene, makeProbeGrid());
    expect(upgradeSpy).toHaveBeenCalledTimes(1);
  });

  it('two distinct mesh objects are both re-injected after dispose', () => {
    const sceneA = new THREE.Scene();
    const sceneB = new THREE.Scene();

    function addMesh(s: THREE.Scene): THREE.Mesh {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0,0,0,1,0,0,0,1,0]), 3));
      const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial());
      s.add(m);
      return m;
    }
    addMesh(sceneA);
    addMesh(sceneB);

    const grid = makeProbeGrid();
    applyDDGIShading(sceneA, grid);
    applyDDGIShading(sceneB, grid);
    expect(upgradeSpy).toHaveBeenCalledTimes(2);

    upgradeSpy.mockClear();
    disposeApplyDDGIShadingCache();

    // Both meshes must be re-injected after dispose.
    applyDDGIShading(sceneA, grid);
    applyDDGIShading(sceneB, grid);
    expect(upgradeSpy).toHaveBeenCalledTimes(2);
  });
});
