import type { FrameInput } from '@vitrum/core';
import { Matrix4, PerspectiveCamera } from 'three';

function copyMat4ArrayLike(dst: Matrix4, src: ArrayLike<number>): Matrix4 {
  const e = dst.elements;
  for (let i = 0; i < 16; i += 1) {
    e[i] = src[i] ?? 0;
  }
  return dst;
}

/**
 * Applies pre-multiplied view / projection matrices from {@link FrameInput} to a
 * Three.js camera so three-gpu-pathtracer's `setCamera` sees the same rays the host used.
 */
export function applyFrameToPerspectiveCamera(
  cam: PerspectiveCamera,
  input: FrameInput,
): void {
  cam.matrixAutoUpdate = false;
  copyMat4ArrayLike(cam.matrixWorld, input.viewMatrix).invert();
  cam.matrix.copy(cam.matrixWorld);
  copyMat4ArrayLike(cam.projectionMatrix, input.projMatrix);
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
}
