// glTF / GLB loader → vitrum Scene + optional embedded camera.
//
// Wraps three.js's GLTFLoader (the de-facto loader for the format on web)
// and threads the result through sceneFromThreeJS so the host gets a
// backend-agnostic vitrum Scene back. Lights and emissive meshes already
// flow through sceneFromThreeJS into `scene.emitters`; the loader only
// adds explicit camera surfacing on top.
//
// Input forms:
//   - `string` (URL)             — fetched + decoded
//   - `Blob` / `File`            — read as ArrayBuffer
//   - `ArrayBuffer`              — parsed in-place
//
// The loader does not own any GPU resources; it just calls GLTFLoader.parse,
// converts the result, and returns plain data. Hosts that need DRACO /
// KTX2 decompression should install the relevant extensions on the loader
// instance via the `configure` hook.

import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Mat4, Vec3, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';

import { sceneFromThreeJS } from './index.js';

export interface LoadedGltf {
  /** vitrum Scene converted from the glTF's scene graph. */
  readonly scene: Scene;
  /** First embedded camera, if any. View + projection matrices are computed
   *  from the world-space transform of the camera node. */
  readonly camera?: GltfCamera;
}

export interface GltfCamera {
  readonly viewMatrix: Mat4;
  readonly projMatrix: Mat4;
  readonly cameraPosition: Vec3;
}

export interface LoadGltfSceneOptions {
  /** Hook to configure the GLTFLoader before parsing — wire DRACO, KTX2,
   *  or Meshopt decoders here if the host needs them. */
  configure?: (loader: GLTFLoader) => void;
  /** Base path passed to `GLTFLoader.parse()`. Defaults to '' (relative).
   *  Only meaningful for glTF files with external .bin / texture URIs. */
  path?: string;
}

export async function loadGltfScene(
  source: string | Blob | File | ArrayBuffer,
  opts?: LoadGltfSceneOptions,
): Promise<LoadedGltf> {
  const arrayBuffer = await toArrayBuffer(source);
  const loader = new GLTFLoader();
  if (opts?.configure) opts.configure(loader);

  const gltf = await new Promise<GLTF>((resolve, reject) => {
    loader.parse(arrayBuffer, opts?.path ?? '', resolve, (err) => reject(err));
  });

  const threeScene = ensureThreeScene(gltf);
  // Auto-compute vertex normals for any mesh missing them. Many glTFs in
  // the wild include normals, but baked exporters and procedural builds
  // often omit them; sceneFromThreeJS requires NORMAL, so compute here
  // rather than surfacing a deep "no normal attribute" error to the host.
  ensureMeshNormals(threeScene);
  const vitrumScene = sceneFromThreeJS(threeScene);

  const camera = extractFirstCamera(gltf);
  return camera ? { scene: vitrumScene, camera } : { scene: vitrumScene };
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

async function toArrayBuffer(source: string | Blob | File | ArrayBuffer): Promise<ArrayBuffer> {
  if (source instanceof ArrayBuffer) return source;
  if (typeof Blob !== 'undefined' && source instanceof Blob) return await source.arrayBuffer();
  if (typeof source === 'string') {
    const resp = await fetch(source);
    if (!resp.ok) {
      throw new Error(`loadGltfScene: failed to fetch ${source}: ${resp.status} ${resp.statusText}`);
    }
    return await resp.arrayBuffer();
  }
  throw new TypeError('loadGltfScene: source must be a URL string, Blob/File, or ArrayBuffer');
}

/** Walk the loaded scene; for any Mesh whose BufferGeometry lacks a
 *  `normal` attribute, compute flat vertex normals in-place. */
function ensureMeshNormals(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    const geom = mesh.geometry as THREE.BufferGeometry;
    if (geom?.attributes?.['normal'] == null && typeof geom?.computeVertexNormals === 'function') {
      geom.computeVertexNormals();
    }
  });
}

/** The GLTFLoader callback gives `gltf.scene` as a THREE.Group rooting the
 *  primary scene. sceneFromThreeJS expects a THREE.Scene; wrap when needed. */
function ensureThreeScene(gltf: GLTF): THREE.Scene {
  const root = gltf.scene as unknown as THREE.Object3D & { isScene?: boolean };
  if (root.isScene) return root as THREE.Scene;
  const wrapper = new THREE.Scene();
  wrapper.add(root as THREE.Object3D);
  return wrapper;
}

function extractFirstCamera(gltf: GLTF): GltfCamera | undefined {
  const cameras = gltf.cameras;
  if (!cameras || cameras.length === 0) return undefined;
  const cam = cameras[0]!;
  cam.updateMatrixWorld(true);

  // Camera world matrix: glTF stores model-space; matrixWorld is the
  // accumulated transform after parent matrices are applied. The view
  // matrix is its inverse.
  const matWorld = new THREE.Matrix4().copy(cam.matrixWorld);
  const matWorldInverse = matWorld.clone().invert();

  // Projection matrix: PerspectiveCamera.projectionMatrix is populated by
  // three.js when the camera was constructed from glTF. Same for orthographic.
  const proj = (cam as THREE.PerspectiveCamera | THREE.OrthographicCamera).projectionMatrix;

  const viewMatrix = asMat4(new Float32Array(matWorldInverse.elements));
  const projMatrix = asMat4(new Float32Array(proj.elements));
  const cameraPosition: Vec3 = [
    matWorld.elements[12]!, matWorld.elements[13]!, matWorld.elements[14]!,
  ];

  return { viewMatrix, projMatrix, cameraPosition };
}
