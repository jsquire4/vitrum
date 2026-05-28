/**
 * Focused coverage for the dedup helpers extracted in Theme T15:
 *   - buildRectAreaLight (shared by the rect-area emitter arm + disc-area path)
 *   - applyTransform / buildGeometry (shared mesh/skinned/instanced geometry tail)
 *   - disposeMaterialTextures (single dispose list, derived from applyTextureMaps)
 *
 * These paths had no prior direct test, so they pin the behaviour the refactor
 * must preserve.
 */
import { describe, expect, it, vi } from 'vitest';
import {
  Mesh,
  RectAreaLight,
  Scene,
  Texture,
  MeshPhysicalMaterial,
  Vector3,
} from 'three';
import {
  vitrumSceneToThree,
  disposeVitrumThreeSceneRoot,
} from '../vitrumSceneToThree.js';
import { luminance } from '../math.js';

function rectLightOf(scene: import('three').Scene): RectAreaLight | undefined {
  return scene.children.find((c) => c instanceof RectAreaLight) as RectAreaLight | undefined;
}

describe('rect-area emitter → RectAreaLight (buildRectAreaLight)', () => {
  it('derives width/height = 2·|axis| and cellPower = lum·4·|u×v|', () => {
    const color: [number, number, number] = [1, 0.5, 0.25];
    const intensity = 3;
    const uAxis: [number, number, number] = [2, 0, 0]; // half-width 2 → width 4
    const vAxis: [number, number, number] = [0, 1.5, 0]; // half-height 1.5 → height 3
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        { kind: 'rect-area', id: 'rl', color, intensity, position: [5, 6, 7], uAxis, vAxis },
      ],
      environment: { kind: 'none' },
    } as never);

    const L = rectLightOf(scene);
    expect(L).toBeDefined();
    expect(L!.width).toBeCloseTo(4);
    expect(L!.height).toBeCloseTo(3);
    // Position baked into matrix (matrixAutoUpdate disabled).
    expect(L!.matrixAutoUpdate).toBe(false);
    expect(L!.position.x).toBeCloseTo(5);
    expect(L!.position.y).toBeCloseTo(6);
    expect(L!.position.z).toBeCloseTo(7);
    // rectArea = 4·|u×v| = 4·(2·1.5) = 12; cellPower = lum·12.
    const expectedArea = 4 * new Vector3(...uAxis).cross(new Vector3(...vAxis)).length();
    const expectedCell = luminance(color[0], color[1], color[2], intensity) * expectedArea;
    expect(L!.userData['cellPower']).toBeCloseTo(expectedCell);
  });

  it('preserves the historical contract that the rect-area arm leaves .name empty', () => {
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        {
          kind: 'rect-area',
          id: 'rl-no-name',
          color: [1, 1, 1],
          intensity: 1,
          position: [0, 0, 0],
          uAxis: [1, 0, 0],
          vAxis: [0, 1, 0],
        },
      ],
      environment: { kind: 'none' },
    } as never);
    const L = rectLightOf(scene);
    expect(L).toBeDefined();
    expect(L!.name).toBe('');
  });

  it('skips a degenerate (parallel-axes) rect-area emitter with a warning', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        {
          kind: 'rect-area',
          id: 'rl-degen',
          color: [1, 1, 1],
          intensity: 1,
          position: [0, 0, 0],
          uAxis: [1, 0, 0],
          vAxis: [2, 0, 0], // parallel to uAxis → zero cross
        },
      ],
      environment: { kind: 'none' },
    } as never);
    expect(rectLightOf(scene)).toBeUndefined();
    expect(warn).toHaveBeenCalled();
    expect(warn.mock.calls.some((c) => /degenerate u\/v axes/.test(String(c[0])))).toBe(true);
    warn.mockRestore();
  });
});

describe('disc-area emitter → RectAreaLight (buildRectAreaLight shared path)', () => {
  it('produces an area-preserving square (half-span √π·r/2) and names the light', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const radius = 2;
    const color: [number, number, number] = [0.8, 0.9, 1];
    const intensity = 4;
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        {
          kind: 'disc-area',
          id: 'disc',
          color,
          intensity,
          position: [1, 2, 3],
          normal: [0, 0, 1],
          radius,
        },
      ],
      environment: { kind: 'none' },
    } as never);
    const L = rectLightOf(scene);
    expect(L).toBeDefined();
    // half-span s = √π·r/2, width = height = 2s = √π·r.
    const side = Math.sqrt(Math.PI) * radius;
    expect(L!.width).toBeCloseTo(side);
    expect(L!.height).toBeCloseTo(side);
    // disc path DOES set the name.
    expect(L!.name).toBe('disc');
    // Area equals π·r² (area-preserving), so cellPower = lum·π·r².
    const expectedCell = luminance(color[0], color[1], color[2], intensity) * Math.PI * radius * radius;
    expect(L!.userData['cellPower']).toBeCloseTo(expectedCell, 4);
    warn.mockRestore();
  });

  it('skips a disc-area emitter with a degenerate normal', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const scene = vitrumSceneToThree({
      primitives: [],
      emitters: [
        {
          kind: 'disc-area',
          id: 'disc-degen',
          color: [1, 1, 1],
          intensity: 1,
          position: [0, 0, 0],
          normal: [0, 0, 0],
          radius: 1,
        },
      ],
      environment: { kind: 'none' },
    } as never);
    expect(rectLightOf(scene)).toBeUndefined();
    warn.mockRestore();
  });
});

describe('mesh geometry/transform tail (buildGeometry + applyTransform)', () => {
  it('bakes the transform into matrix/matrixWorld with auto-update disabled', () => {
    const transform = new Float32Array([
      2, 0, 0, 0,
      0, 2, 0, 0,
      0, 0, 2, 0,
      10, 20, 30, 1,
    ]);
    const scene = vitrumSceneToThree({
      primitives: [
        {
          kind: 'mesh',
          id: 'm',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
          indices: new Uint32Array([0, 1, 2]),
          transform: transform as never,
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    } as never);
    const mesh = scene.children.find((c) => c instanceof Mesh) as Mesh;
    expect(mesh).toBeDefined();
    expect(mesh.matrixAutoUpdate).toBe(false);
    expect(mesh.matrix.elements[12]).toBeCloseTo(10);
    expect(mesh.matrix.elements[13]).toBeCloseTo(20);
    expect(mesh.matrix.elements[14]).toBeCloseTo(30);
    expect(mesh.matrixWorld.elements[0]).toBeCloseTo(2);
    expect(mesh.name).toBe('m');
    // uv + index round-tripped via buildGeometry.
    expect(mesh.geometry.getAttribute('uv')).toBeDefined();
    expect(mesh.geometry.getIndex()).not.toBeNull();
  });

  it('defaults to identity when no transform is supplied', () => {
    const scene = vitrumSceneToThree({
      primitives: [
        {
          kind: 'mesh',
          id: 'm-noxform',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    } as never);
    const mesh = scene.children.find((c) => c instanceof Mesh) as Mesh;
    expect(mesh.matrix.elements).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  });
});

describe('disposeVitrumThreeSceneRoot (disposeMaterialTextures shared list)', () => {
  it('disposes every applied texture-map field exactly once across meshes', () => {
    // Bare THREE.Scene (background/environment stay null) so the test isolates
    // the per-material texture dispose list, not the env/background dispose.
    const scene = new Scene();

    // Hand-build two meshes sharing one texture in two map slots so we can
    // assert (a) all seven slots are visited and (b) dedup prevents a double
    // dispose of the shared texture.
    const shared = new Texture();
    const baseColor = new Texture();
    const normal = new Texture();
    const rough = new Texture();
    const metal = new Texture();
    const emissive = new Texture();
    const alpha = new Texture();

    const matA = new MeshPhysicalMaterial();
    matA.map = baseColor;
    matA.normalMap = normal;
    matA.roughnessMap = rough;
    matA.metalnessMap = metal;
    matA.emissiveMap = emissive;
    matA.alphaMap = alpha;
    matA.transmissionMap = shared;

    const matB = new MeshPhysicalMaterial();
    matB.map = shared; // same texture as matA.transmissionMap → dedup target

    const meshA = new Mesh(undefined, matA);
    const meshB = new Mesh(undefined, matB);
    scene.add(meshA);
    scene.add(meshB);

    const all = [baseColor, normal, rough, metal, emissive, alpha, shared];
    const spies = all.map((t) => vi.spyOn(t, 'dispose'));

    disposeVitrumThreeSceneRoot(scene);

    // Each texture disposed exactly once (shared one only once despite 2 refs).
    for (const s of spies) expect(s).toHaveBeenCalledTimes(1);
  });

  it('handles array-material meshes via the same dispose list', () => {
    const scene = new Scene();
    const tex = new Texture();
    const mat = new MeshPhysicalMaterial();
    mat.emissiveMap = tex;
    const mesh = new Mesh(undefined, [mat]);
    scene.add(mesh);
    const spy = vi.spyOn(tex, 'dispose');
    disposeVitrumThreeSceneRoot(scene);
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
