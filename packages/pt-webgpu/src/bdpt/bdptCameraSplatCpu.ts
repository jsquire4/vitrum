/**
 * Deterministic CPU algebra for pt-webgpu's BDPT t=1 camera-splat strategy.
 *
 * The production shader uses the same pinhole-perspective measurement as
 * PBRT's `PerspectiveCamera::{We,Pdf_We,Sample_Wi}`:
 *
 *   camera directional pdf = 1 / (A cos^3(theta))
 *   Sample_Wi importance / position pdf =
 *     1 / (A cos^3(theta) distance^2)
 *
 * `A` is the area of the camera's raster-support rectangle projected onto the
 * plane one unit along the camera's forward direction. It is derived from the
 * inverse view-projection matrix, so asymmetric perspective projections are
 * handled without a separately-authored FOV or aspect ratio.
 *
 * References:
 * - Pharr, Jakob & Humphreys, *Physically Based Rendering*, 3e,
 *   Perspective Camera `We`, `Pdf_We`, and `Sample_Wi`.
 * - Veach 1997 §10.3, the s=n-1/t=1 bidirectional technique.
 */

import type { Vec3 } from './bdptConnectionMisFull.js';

export type Mat4Like = ArrayLike<number>;

export interface BdptCameraSplatProjection {
  readonly pixelX: number;
  readonly pixelY: number;
  readonly pixelIndex: number;
  /** Unit direction from the camera endpoint toward the light vertex. */
  readonly cameraToVertex: Vec3;
  /** Unit pinhole-camera forward direction. */
  readonly cameraForward: Vec3;
  readonly distanceSquared: number;
  readonly cosTheta: number;
  readonly imagePlaneArea: number;
  /** `PerspectiveCamera::Pdf_We`'s directional component. */
  readonly cameraDirectionalPdf: number;
  /** `PerspectiveCamera::Sample_Wi` return value divided by its position pdf. */
  readonly sampleWiOverPdf: number;
}

function finiteVec3(v: Vec3): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v: Vec3): Vec3 | null {
  const l = length(v);
  if (!Number.isFinite(l) || l <= 1e-12) return null;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function requireMat4(matrix: Mat4Like, label: string): void {
  if (matrix.length !== 16) {
    throw new RangeError(`${label} must contain exactly 16 column-major values`);
  }
  for (let i = 0; i < 16; i += 1) {
    if (!Number.isFinite(matrix[i])) {
      throw new RangeError(`${label}[${i}] must be finite`);
    }
  }
}

function transformPoint4(
  matrix: Mat4Like,
  point: readonly [number, number, number, number],
): readonly [number, number, number, number] {
  const [x, y, z, w] = point;
  return [
    matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]! * w,
    matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]! * w,
    matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]! * w,
    matrix[3]! * x + matrix[7]! * y + matrix[11]! * z + matrix[15]! * w,
  ];
}

function cameraDirectionForNdc(
  inverseViewProjection: Mat4Like,
  ndcX: number,
  ndcY: number,
): Vec3 | null {
  const near = cameraPointForNdc(inverseViewProjection, ndcX, ndcY, -1);
  const far = cameraPointForNdc(inverseViewProjection, ndcX, ndcY, 1);
  if (near == null || far == null) return null;
  return normalize(sub(far, near));
}

function cameraPointForNdc(
  inverseViewProjection: Mat4Like,
  ndcX: number,
  ndcY: number,
  ndcZ: number,
): Vec3 | null {
  const point4 = transformPoint4(
    inverseViewProjection,
    [ndcX, ndcY, ndcZ, 1],
  );
  if (!Number.isFinite(point4[3]) || Math.abs(point4[3]) <= 1e-12) {
    return null;
  }
  const point: Vec3 = [
    point4[0] / point4[3],
    point4[1] / point4[3],
    point4[2] / point4[3],
  ];
  return finiteVec3(point) ? point : null;
}

/**
 * Project one light-subpath vertex onto the pinhole sensor and return the exact
 * camera-measure factors used by the production WGSL. `null` means the vertex
 * is outside raster support, behind the camera, or the matrices are degenerate.
 */
export function projectBdptCameraSplat(args: {
  readonly viewProjection: Mat4Like;
  readonly inverseViewProjection: Mat4Like;
  readonly cameraPosition: Vec3;
  readonly vertexPosition: Vec3;
  readonly width: number;
  readonly height: number;
}): BdptCameraSplatProjection | null {
  const {
    viewProjection,
    inverseViewProjection,
    cameraPosition,
    vertexPosition,
    width,
    height,
  } = args;
  requireMat4(viewProjection, 'projectBdptCameraSplat.viewProjection');
  requireMat4(
    inverseViewProjection,
    'projectBdptCameraSplat.inverseViewProjection',
  );
  if (!finiteVec3(cameraPosition) || !finiteVec3(vertexPosition)) {
    throw new RangeError(
      'projectBdptCameraSplat cameraPosition and vertexPosition must be finite',
    );
  }
  if (!Number.isSafeInteger(width) || width <= 0 ||
      !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError(
      'projectBdptCameraSplat width and height must be positive safe integers',
    );
  }

  const clip = transformPoint4(
    viewProjection,
    [vertexPosition[0], vertexPosition[1], vertexPosition[2], 1],
  );
  if (!Number.isFinite(clip[3]) || clip[3] <= 1e-8) return null;
  const ndcX = clip[0] / clip[3];
  const ndcY = clip[1] / clip[3];
  if (
    !Number.isFinite(ndcX) ||
    !Number.isFinite(ndcY) ||
    ndcX < -1 ||
    ndcX >= 1 ||
    ndcY <= -1 ||
    ndcY > 1
  ) {
    return null;
  }

  const rasterCenterDirection = cameraDirectionForNdc(
    inverseViewProjection,
    0,
    0,
  );
  const near00 = cameraPointForNdc(inverseViewProjection, -1, 1, -1);
  const near10 = cameraPointForNdc(inverseViewProjection, 1, 1, -1);
  const near01 = cameraPointForNdc(inverseViewProjection, -1, -1, -1);
  let cameraForward =
    near00 == null || near10 == null || near01 == null
      ? null
      : normalize(cross(sub(near10, near00), sub(near01, near00)));
  if (
    cameraForward != null &&
    rasterCenterDirection != null &&
    dot(cameraForward, rasterCenterDirection) < 0
  ) {
    cameraForward = [
      -cameraForward[0],
      -cameraForward[1],
      -cameraForward[2],
    ];
  }
  const corner00 = cameraDirectionForNdc(
    inverseViewProjection,
    -1,
    1,
  );
  const corner10 = cameraDirectionForNdc(
    inverseViewProjection,
    1,
    1,
  );
  const corner01 = cameraDirectionForNdc(
    inverseViewProjection,
    -1,
    -1,
  );
  if (
    rasterCenterDirection == null ||
    cameraForward == null ||
    corner00 == null ||
    corner10 == null ||
    corner01 == null
  ) {
    return null;
  }

  const c00 = dot(corner00, cameraForward);
  const c10 = dot(corner10, cameraForward);
  const c01 = dot(corner01, cameraForward);
  if (c00 <= 1e-8 || c10 <= 1e-8 || c01 <= 1e-8) return null;
  const plane00: Vec3 = [
    corner00[0] / c00,
    corner00[1] / c00,
    corner00[2] / c00,
  ];
  const plane10: Vec3 = [
    corner10[0] / c10,
    corner10[1] / c10,
    corner10[2] / c10,
  ];
  const plane01: Vec3 = [
    corner01[0] / c01,
    corner01[1] / c01,
    corner01[2] / c01,
  ];
  const imagePlaneArea = length(
    cross(sub(plane10, plane00), sub(plane01, plane00)),
  );
  if (!Number.isFinite(imagePlaneArea) || imagePlaneArea <= 1e-12) return null;

  const cameraToVertexVector = sub(vertexPosition, cameraPosition);
  const distanceSquared = dot(cameraToVertexVector, cameraToVertexVector);
  const cameraToVertex = normalize(cameraToVertexVector);
  if (
    cameraToVertex == null ||
    !Number.isFinite(distanceSquared) ||
    distanceSquared <= 1e-12
  ) {
    return null;
  }
  const cosTheta = dot(cameraToVertex, cameraForward);
  if (!Number.isFinite(cosTheta) || cosTheta <= 1e-8) return null;
  const cameraDirectionalPdf =
    1 / (imagePlaneArea * cosTheta * cosTheta * cosTheta);
  const sampleWiOverPdf = cameraDirectionalPdf / distanceSquared;
  if (
    !Number.isFinite(cameraDirectionalPdf) ||
    !Number.isFinite(sampleWiOverPdf) ||
    cameraDirectionalPdf <= 0 ||
    sampleWiOverPdf <= 0
  ) {
    return null;
  }

  const pixelX = Math.floor((ndcX * 0.5 + 0.5) * width);
  const pixelY = Math.floor((1 - (ndcY * 0.5 + 0.5)) * height);
  if (pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height) {
    return null;
  }
  return {
    pixelX,
    pixelY,
    pixelIndex: pixelY * width + pixelX,
    cameraToVertex,
    cameraForward,
    distanceSquared,
    cosTheta,
    imagePlaneArea,
    cameraDirectionalPdf,
    sampleWiOverPdf,
  };
}

/**
 * Algebraic t=1 contribution at one connectible light vertex. Production
 * traversal has already required an unoccluded vacuum edge; a non-vacuum
 * camera edge is rejected because the camera endpoint carries no medium state.
 */
export function evaluateBdptCameraSplatContribution(args: {
  readonly lightThroughput: Vec3;
  readonly lightScatter: Vec3;
  readonly surfaceCosine: number;
  readonly sampleWiOverPdf: number;
  readonly misWeight: number;
}): Vec3 {
  const {
    lightThroughput,
    lightScatter,
    surfaceCosine,
    sampleWiOverPdf,
    misWeight,
  } = args;
  for (const [label, value] of [
    ['lightThroughput', lightThroughput],
    ['lightScatter', lightScatter],
  ] as const) {
    if (!finiteVec3(value) || value.some((x) => x < 0)) {
      throw new RangeError(
        `evaluateBdptCameraSplatContribution.${label} must be finite and non-negative`,
      );
    }
  }
  for (const [label, value] of [
    ['surfaceCosine', surfaceCosine],
    ['sampleWiOverPdf', sampleWiOverPdf],
    ['misWeight', misWeight],
  ] as const) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(
        `evaluateBdptCameraSplatContribution.${label} must be finite and non-negative`,
      );
    }
  }
  const scale = surfaceCosine * sampleWiOverPdf * misWeight;
  return [
    lightThroughput[0] * lightScatter[0] * scale,
    lightThroughput[1] * lightScatter[1] * scale,
    lightThroughput[2] * lightScatter[2] * scale,
  ];
}
