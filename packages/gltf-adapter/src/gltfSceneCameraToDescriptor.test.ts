import { describe, expect, it } from 'vitest';
import { asMat4, isCameraDescriptor } from '@vitrum/core';
import { gltfSceneCameraToDescriptor, type GltfSceneCamera } from './cameraMetadata.js';

describe('gltfSceneCameraToDescriptor', () => {
  it('flattens a perspective glTF camera into a core CameraDescriptor', () => {
    const camera: GltfSceneCamera = {
      cameraIndex: 0,
      nodeIndex: 1,
      path: 'cameras[0]',
      nodePath: 'nodes[1]',
      type: 'perspective',
      worldMatrix: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ])),
      perspective: { yfov: 0.8, znear: 0.05, zfar: 200, aspectRatio: 1.5 },
    };
    const descriptor = gltfSceneCameraToDescriptor(camera);
    expect(isCameraDescriptor(descriptor)).toBe(true);
    expect(descriptor).toMatchObject({
      type: 'perspective',
      yfov: 0.8,
      znear: 0.05,
      zfar: 200,
      aspectRatio: 1.5,
    });
  });
});
