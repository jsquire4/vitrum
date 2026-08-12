/**
 * Host camera descriptors. Cameras are frame-owned, not Scene-owned: the host
 * (or attachVitrum's glTF loop) converts a descriptor into `FrameInput`
 * view/projection matrices each tick.
 *
 * Pose is a column-major world matrix (glTF / three.js convention). Projection
 * matches the glTF 2.0 / OpenGL clip-space matrix so THREE cameras and authored
 * glTF cameras land in the same FrameInput space.
 */

import { asMat4, isMat4, type Mat4 } from './scene/math.js';
import { deriveCameraPositionFromViewMatrix } from './frame.js';
import type { Vec3 } from './scene/math.js';

export interface PerspectiveCameraDescriptor {
  readonly type: 'perspective';
  readonly worldMatrix: Mat4;
  /** Vertical field of view in radians. Must be > 0. */
  readonly yfov: number;
  readonly znear: number;
  readonly zfar?: number;
  /** Authored aspect (width/height). Viewport aspect overrides when supplied. */
  readonly aspectRatio?: number;
}

export interface OrthographicCameraDescriptor {
  readonly type: 'orthographic';
  readonly worldMatrix: Mat4;
  readonly xmag: number;
  readonly ymag: number;
  readonly znear: number;
  readonly zfar: number;
}

export type CameraDescriptor = PerspectiveCameraDescriptor | OrthographicCameraDescriptor;

export interface CameraFrameMatrices {
  readonly viewMatrix: Mat4;
  readonly projMatrix: Mat4;
  readonly cameraPosition: Vec3;
}

export function isCameraDescriptor(value: unknown): value is CameraDescriptor {
  if (value == null || typeof value !== 'object') return false;
  const cam = value as Partial<CameraDescriptor> & { readonly type?: unknown };
  if (!isMat4(cam.worldMatrix) && !isMat4Length(cam.worldMatrix)) return false;
  if (cam.type === 'perspective') {
    const p = cam as Partial<PerspectiveCameraDescriptor>;
    return isPositiveFinite(p.yfov) && isPositiveFinite(p.znear);
  }
  if (cam.type === 'orthographic') {
    const o = cam as Partial<OrthographicCameraDescriptor>;
    return (
      isPositiveFinite(o.xmag) &&
      isPositiveFinite(o.ymag) &&
      Number.isFinite(o.znear) &&
      Number.isFinite(o.zfar) &&
      (o.zfar as number) > (o.znear as number)
    );
  }
  return false;
}

/**
 * Convert a host camera descriptor into FrameInput matrices.
 *
 * @param aspect - Viewport width/height. Overrides perspective `aspectRatio`
 *   when finite and > 0. For orthographic cameras, keeps `ymag` and scales
 *   `xmag` to this aspect so a resized canvas does not stretch.
 */
export function cameraToFrameMatrices(
  camera: CameraDescriptor,
  aspect?: number,
): CameraFrameMatrices {
  const world = asMat4(
    camera.worldMatrix instanceof Float32Array
      ? camera.worldMatrix
      : new Float32Array(camera.worldMatrix as unknown as ArrayLike<number>),
  );
  const inverse = invertAffineMat4(world);
  if (inverse == null) {
    throw new RangeError('CameraDescriptor.worldMatrix must be invertible.');
  }
  const viewMatrix = asMat4(inverse);
  const projMatrix = camera.type === 'perspective'
    ? perspectiveProjection(camera, aspect)
    : orthographicProjection(camera, aspect);
  return {
    viewMatrix,
    projMatrix,
    cameraPosition: deriveCameraPositionFromViewMatrix(viewMatrix, 'CameraDescriptor.worldMatrix'),
  };
}

function isMat4Length(value: unknown): value is ArrayLike<number> {
  return (
    value != null &&
    typeof value === 'object' &&
    typeof (value as ArrayLike<number>).length === 'number' &&
    (value as ArrayLike<number>).length === 16
  );
}

function isPositiveFinite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function perspectiveProjection(
  camera: PerspectiveCameraDescriptor,
  aspect?: number,
): Mat4 {
  const yfov = camera.yfov;
  const znear = camera.znear;
  if (!(yfov > 0) || !(znear > 0)) {
    throw new RangeError('PerspectiveCameraDescriptor requires yfov > 0 and znear > 0.');
  }
  const authoredAspect = camera.aspectRatio;
  const a =
    aspect != null && Number.isFinite(aspect) && aspect > 0
      ? aspect
      : authoredAspect != null && Number.isFinite(authoredAspect) && authoredAspect > 0
        ? authoredAspect
        : 1;
  const tanHalf = Math.tan(yfov * 0.5);
  if (!(tanHalf > 0) || !Number.isFinite(tanHalf)) {
    throw new RangeError('PerspectiveCameraDescriptor.yfov must produce a finite tan(yfov/2).');
  }
  const xScale = 1 / (a * tanHalf);
  const yScale = 1 / tanHalf;
  const zfar = camera.zfar;
  const out = new Float32Array(16);
  out[0] = xScale;
  out[5] = yScale;
  out[11] = -1;
  if (zfar != null && Number.isFinite(zfar) && zfar > znear) {
    out[10] = (zfar + znear) / (znear - zfar);
    out[14] = (2 * zfar * znear) / (znear - zfar);
  } else {
    out[10] = -1;
    out[14] = -2 * znear;
  }
  return asMat4(out);
}

function orthographicProjection(
  camera: OrthographicCameraDescriptor,
  aspect?: number,
): Mat4 {
  let xmag = camera.xmag;
  const ymag = camera.ymag;
  const znear = camera.znear;
  const zfar = camera.zfar;
  if (!(xmag > 0) || !(ymag > 0) || !Number.isFinite(znear) || !(zfar > znear)) {
    throw new RangeError(
      'OrthographicCameraDescriptor requires xmag > 0, ymag > 0, and zfar > znear.',
    );
  }
  if (aspect != null && Number.isFinite(aspect) && aspect > 0) {
    xmag = ymag * aspect;
  }
  const out = new Float32Array(16);
  out[0] = 1 / xmag;
  out[5] = 1 / ymag;
  out[10] = 2 / (znear - zfar);
  out[14] = (zfar + znear) / (znear - zfar);
  out[15] = 1;
  return asMat4(out);
}

/** Invert a column-major affine 4×4. Returns null when singular or non-finite. */
function invertAffineMat4(m: Mat4): Float32Array | null {
  const a00 = m[0]!, a01 = m[1]!, a02 = m[2]!, a03 = m[3]!;
  const a10 = m[4]!, a11 = m[5]!, a12 = m[6]!, a13 = m[7]!;
  const a20 = m[8]!, a21 = m[9]!, a22 = m[10]!, a23 = m[11]!;
  const a30 = m[12]!, a31 = m[13]!, a32 = m[14]!, a33 = m[15]!;
  if (
    ![a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33].every(
      Number.isFinite,
    )
  ) {
    return null;
  }
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || det === 0) return null;
  const invDet = 1 / det;
  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return Array.from(out).every(Number.isFinite) ? out : null;
}
