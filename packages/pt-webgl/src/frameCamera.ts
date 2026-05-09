import type { FrameInput } from '@vitrum/core';
import { Matrix4, PerspectiveCamera } from 'three';

/**
 * Applies pre-multiplied view / projection matrices from {@link FrameInput} to a
 * Three.js camera so three-gpu-pathtracer's `setCamera` sees the same rays the host used.
 */
export function applyFrameToPerspectiveCamera(
  cam: PerspectiveCamera,
  input: FrameInput,
): void {
  const view = new Matrix4().fromArray(Array.from(input.viewMatrix));
  const world = view.clone().invert();
  cam.matrixAutoUpdate = false;
  cam.matrixWorld.copy(world);
  cam.matrix.copy(world);
  cam.projectionMatrix.fromArray(Array.from(input.projMatrix));
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
}
