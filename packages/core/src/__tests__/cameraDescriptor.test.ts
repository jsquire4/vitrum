import { describe, expect, it } from 'vitest';
import {
  asMat4,
  cameraToFrameMatrices,
  isCameraDescriptor,
  type PerspectiveCameraDescriptor,
  type OrthographicCameraDescriptor,
} from '../index.js';

const IDENTITY = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

describe('cameraToFrameMatrices', () => {
  it('inverts a translated world matrix into view + eye', () => {
    const world = asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      2, -3, 5, 1,
    ]));
    const camera: PerspectiveCameraDescriptor = {
      type: 'perspective',
      worldMatrix: world,
      yfov: Math.PI / 2,
      znear: 0.1,
      zfar: 100,
      aspectRatio: 1,
    };
    const { viewMatrix, cameraPosition } = cameraToFrameMatrices(camera);
    expect(cameraPosition[0]).toBeCloseTo(2);
    expect(cameraPosition[1]).toBeCloseTo(-3);
    expect(cameraPosition[2]).toBeCloseTo(5);
    expect(viewMatrix[12]).toBeCloseTo(-2);
    expect(viewMatrix[13]).toBeCloseTo(3);
    expect(viewMatrix[14]).toBeCloseTo(-5);
  });

  it('builds the glTF/OpenGL finite perspective matrix', () => {
    const camera: PerspectiveCameraDescriptor = {
      type: 'perspective',
      worldMatrix: IDENTITY,
      yfov: Math.PI / 2,
      znear: 1,
      zfar: 100,
      aspectRatio: 1,
    };
    const { projMatrix } = cameraToFrameMatrices(camera);
    expect(projMatrix[0]).toBeCloseTo(1);
    expect(projMatrix[5]).toBeCloseTo(1);
    expect(projMatrix[10]).toBeCloseTo(-101 / 99);
    expect(projMatrix[11]).toBe(-1);
    expect(projMatrix[14]).toBeCloseTo(-200 / 99);
    expect(projMatrix[15]).toBe(0);
  });

  it('overrides authored aspect with the viewport aspect', () => {
    const camera: PerspectiveCameraDescriptor = {
      type: 'perspective',
      worldMatrix: IDENTITY,
      yfov: Math.PI / 2,
      znear: 1,
      aspectRatio: 1,
    };
    const square = cameraToFrameMatrices(camera, 1);
    const wide = cameraToFrameMatrices(camera, 2);
    expect(wide.projMatrix[0]).toBeCloseTo(square.projMatrix[0]! * 0.5);
    expect(wide.projMatrix[5]).toBeCloseTo(square.projMatrix[5]!);
  });

  it('builds a symmetric orthographic matrix and scales xmag to viewport aspect', () => {
    const camera: OrthographicCameraDescriptor = {
      type: 'orthographic',
      worldMatrix: IDENTITY,
      xmag: 2,
      ymag: 2,
      znear: 0.1,
      zfar: 10,
    };
    const { projMatrix } = cameraToFrameMatrices(camera, 2);
    expect(projMatrix[0]).toBeCloseTo(1 / 4);
    expect(projMatrix[5]).toBeCloseTo(0.5);
    expect(projMatrix[15]).toBe(1);
  });

  it('rejects a singular world matrix', () => {
    const camera: PerspectiveCameraDescriptor = {
      type: 'perspective',
      worldMatrix: asMat4(new Float32Array(16)),
      yfov: 1,
      znear: 0.1,
    };
    expect(() => cameraToFrameMatrices(camera)).toThrow(/invertible/);
  });
});

describe('isCameraDescriptor', () => {
  it('accepts a well-formed perspective descriptor', () => {
    expect(isCameraDescriptor({
      type: 'perspective',
      worldMatrix: IDENTITY,
      yfov: 1,
      znear: 0.1,
    })).toBe(true);
  });

  it('rejects a THREE-shaped camera object', () => {
    expect(isCameraDescriptor({
      updateMatrixWorld() { /* noop */ },
      matrixWorldInverse: { elements: IDENTITY },
      projectionMatrix: { elements: IDENTITY },
      position: { x: 0, y: 0, z: 0 },
    })).toBe(false);
  });
});
