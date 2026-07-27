import { describe, expect, it } from 'vitest';
import {
  CAMERA_POSITION_ABSOLUTE_TOLERANCE,
  CAMERA_POSITION_RELATIVE_TOLERANCE,
  asMat4,
  canonicalizeFrameCamera,
  deriveCameraPositionFromViewMatrix,
  resolveFrameCameraPosition,
  type FrameInput,
  type Vec3,
} from '../index.js';

function translatedView(position: Vec3) {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    -position[0], -position[1], -position[2], 1,
  ]));
}

function frame(
  position: Vec3,
  cameraPosition?: Vec3,
): FrameInput {
  return {
    viewMatrix: translatedView(position),
    // Deliberately asymmetric/off-axis. Projection does not move the eye.
    projMatrix: asMat4(new Float32Array([
      1.2, 0, 0, 0,
      0, 1.5, 0, 0,
      0.17, -0.08, -1, -1,
      0, 0, -0.2, 0,
    ])),
    ...(cameraPosition === undefined ? {} : { cameraPosition }),
    viewport: { width: 16, height: 9, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
  };
}

describe('canonical frame camera position', () => {
  it('derives inverse(viewMatrix) translation and accepts an exact legacy value', () => {
    const input = frame([3, -2, 7], [3, -2, 7]);
    expect(resolveFrameCameraPosition(input)).toEqual([3, -2, 7]);
    expect(canonicalizeFrameCamera(input).cameraPosition).toEqual([3, -2, 7]);
  });

  it('derives the position when the deprecated field is omitted', () => {
    const input = frame([-4, 5, 9]);
    expect(resolveFrameCameraPosition(input)).toEqual([-4, 5, 9]);
  });

  it('canonicalizes an identity-view eye to positive zero in every component', () => {
    const eye = deriveCameraPositionFromViewMatrix(translatedView([0, 0, 0]));
    expect(eye).toEqual([0, 0, 0]);
    expect(eye.every((component) => !Object.is(component, -0))).toBe(true);
  });

  it('accepts float32-scale rounding but returns the matrix-derived value', () => {
    const provided: Vec3 = [
      2 + CAMERA_POSITION_ABSOLUTE_TOLERANCE * 0.5,
      3,
      4,
    ];
    expect(resolveFrameCameraPosition(frame([2, 3, 4], provided))).toEqual([2, 3, 4]);
  });

  it('uses a relative f32 tolerance at large world coordinates', () => {
    const position: Vec3 = [1_000_000, -2_000_000, 3_000_000];
    const tolerance =
      CAMERA_POSITION_RELATIVE_TOLERANCE * Math.abs(position[0]);
    expect(resolveFrameCameraPosition(frame(position, [
      position[0] + tolerance * 0.5,
      position[1],
      position[2],
    ]))).toEqual(position);
  });

  it('rejects a meaningful disagreement with an actionable error', () => {
    expect(() => resolveFrameCameraPosition(
      frame([1, 2, 3], [1.25, 2, 3]),
      'renderFrame',
    )).toThrow(/cameraPosition disagrees with inverse\(viewMatrix\).*Projection jitter/s);
  });

  it('derives the eye through a rotated affine view transform', () => {
    // Camera world rotation is +90° around Y; view uses its transpose and
    // translation -R^T·eye for eye=(3,4,5).
    const view = asMat4(new Float32Array([
      0, 0, -1, 0,
      0, 1, 0, 0,
      1, 0, 0, 0,
      -5, -4, 3, 1,
    ]));
    expect(deriveCameraPositionFromViewMatrix(view)).toEqual([3, 4, 5]);
  });

  it('rejects singular and non-finite view matrices', () => {
    const singular = asMat4(new Float32Array(16));
    singular[15] = 1;
    expect(() => deriveCameraPositionFromViewMatrix(singular)).toThrow(/singular/);
    const nearSingular = translatedView([0, 0, 0]);
    nearSingular[10] = 1e-8;
    expect(() => deriveCameraPositionFromViewMatrix(nearSingular)).toThrow(/singular/);
    const nonFinite = translatedView([0, 0, 0]);
    nonFinite[6] = Number.NaN;
    expect(() => deriveCameraPositionFromViewMatrix(nonFinite)).toThrow(/viewMatrix\[6\].*finite/);
  });

  it('rejects a projective matrix in the view slot', () => {
    const projectiveView = translatedView([0, 0, 0]);
    projectiveView[3] = 0.01;
    expect(() => deriveCameraPositionFromViewMatrix(projectiveView)).toThrow(
      /affine camera view matrix/,
    );
  });

  it('does not let large translation hide a projective last-row term', () => {
    const projectiveView = translatedView([
      1_000_000_000,
      -2_000_000_000,
      3_000_000_000,
    ]);
    projectiveView[11] = 0.01;
    expect(() => deriveCameraPositionFromViewMatrix(projectiveView)).toThrow(
      /affine camera view matrix/,
    );
  });
});
