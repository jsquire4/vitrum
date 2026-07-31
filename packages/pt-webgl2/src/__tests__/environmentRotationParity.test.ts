import { describe, expect, it } from 'vitest';
import type { FrameInput, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { canonicalizeEnvironmentRotationF32 } from '@vitrum/shared-samplers';
import {
  preflightFrameUniforms,
  type FrameUniformsConfig,
} from '../gl/frameUniformsPacker.js';
import { makeRotationYMat4 } from '../mat4.js';

const IDENTITY = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]));

function environmentMatrix(rotationY: number): Float32Array {
  const scene: Scene = {
    primitives: [],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: { opaque: true },
      rotationY,
    },
  };
  const input = {
    viewMatrix: IDENTITY,
    projMatrix: IDENTITY,
    viewport: { width: 4, height: 4 },
    frameSeed: 1,
  } as FrameInput;
  const config: FrameUniformsConfig = {
    scene,
    hasEnvMap: true,
    materialLodDepth: 0,
    backgroundBlur: 0,
    spectralEnabled: false,
    backgroundAlpha: 1,
    bdpt: false,
    bdptMaxLightBounces: 1,
    cameraType: 1,
    transportBounds: {
      center: [0, 0, 0],
      radius: 10,
      min: [-10, -10, -10],
      max: [10, 10, 10],
    },
    dof: undefined,
  };
  return preflightFrameUniforms(input, 1, 4, 4, config)
    .uniforms.environmentRotation;
}

describe('pt-webgl2 environment rotation parity', () => {
  it.each([Math.PI / 3, 1e300, 0, -0, 2 ** -150])(
    'uses the shared wrap-then-fround policy for %s',
    (rotationY) => {
      const packed = canonicalizeEnvironmentRotationF32(rotationY);
      const expected = packed === 0
        ? IDENTITY
        : makeRotationYMat4(-packed);
      expect(environmentMatrix(rotationY)).toEqual(expected);
    },
  );
});
