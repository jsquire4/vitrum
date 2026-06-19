import { asMat4, type Mat4 } from '@vitrum/core';
import type { GltfCamera, GltfJson } from './gltfTypes.js';

export interface GltfPerspectiveCameraProjection {
  readonly yfov?: number;
  readonly znear?: number;
  readonly zfar?: number;
  readonly aspectRatio?: number;
}

export interface GltfOrthographicCameraProjection {
  readonly xmag?: number;
  readonly ymag?: number;
  readonly znear?: number;
  readonly zfar?: number;
}

export interface GltfSceneCamera {
  readonly cameraIndex: number;
  readonly nodeIndex: number;
  readonly path: string;
  readonly nodePath: string;
  readonly type: 'perspective' | 'orthographic' | 'unknown';
  readonly worldMatrix: Mat4;
  readonly name?: string;
  readonly nodeName?: string;
  readonly perspective?: GltfPerspectiveCameraProjection;
  readonly orthographic?: GltfOrthographicCameraProjection;
}

export function collectSceneCameras(
  gltf: GltfJson,
  worldTransforms: ReadonlyMap<number, Mat4>,
): GltfSceneCamera[] {
  const cameras = gltf.cameras ?? [];
  const nodes = gltf.nodes ?? [];
  const result: GltfSceneCamera[] = [];
  for (const [nodeIndex, worldMatrix] of worldTransforms) {
    const node = nodes[nodeIndex];
    if (!node) continue;
    const candidateCameraIndex = node.camera;
    if (
      candidateCameraIndex === undefined ||
      !Number.isInteger(candidateCameraIndex) ||
      candidateCameraIndex < 0
    ) {
      continue;
    }
    const cameraIndex = candidateCameraIndex;
    const camera = cameras[cameraIndex];
    if (!camera) continue;
    const type = camera.type === 'perspective'
      ? 'perspective'
      : camera.type === 'orthographic'
        ? 'orthographic'
        : 'unknown';
    result.push({
      cameraIndex,
      nodeIndex,
      path: `cameras[${cameraIndex}]`,
      nodePath: `nodes[${nodeIndex}]`,
      type,
      worldMatrix: asMat4(new Float32Array(worldMatrix)),
      ...(typeof camera.name === 'string' ? { name: camera.name } : {}),
      ...(typeof node.name === 'string' ? { nodeName: node.name } : {}),
      ...(type === 'perspective'
        ? { perspective: extractPerspectiveCameraProjection(camera) }
        : {}),
      ...(type === 'orthographic'
        ? { orthographic: extractOrthographicCameraProjection(camera) }
        : {}),
    });
  }
  return result;
}

function extractPerspectiveCameraProjection(
  camera: GltfCamera,
): GltfPerspectiveCameraProjection {
  const source = isRecord(camera.perspective) ? camera.perspective : {};
  return {
    ...finiteField(source, 'yfov'),
    ...finiteField(source, 'znear'),
    ...finiteField(source, 'zfar'),
    ...finiteField(source, 'aspectRatio'),
  };
}

function extractOrthographicCameraProjection(
  camera: GltfCamera,
): GltfOrthographicCameraProjection {
  const source = isRecord(camera.orthographic) ? camera.orthographic : {};
  return {
    ...finiteField(source, 'xmag'),
    ...finiteField(source, 'ymag'),
    ...finiteField(source, 'znear'),
    ...finiteField(source, 'zfar'),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function finiteField<TName extends string>(
  source: Record<string, unknown>,
  name: TName,
): Partial<Record<TName, number>> {
  const value = source[name];
  return typeof value === 'number' && Number.isFinite(value)
    ? { [name]: value } as Partial<Record<TName, number>>
    : {};
}
