import { describe, expect, it } from 'vitest';
import type { FrameInput, Scene } from '@vitrum/core';
import {
  packFrameUniforms,
  type FrameUniformsConfig,
} from './frameUniformsPacker.js';
import { invertMat4 } from '../mat4.js';

const IDENTITY = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);
const PERSPECTIVE = new Float32Array([
  1.5, 0, 0, 0,
  0, 1.5, 0, 0,
  0, 0, -1.002, -1,
  0, 0, -0.2, 0,
]);

const FRAME: FrameInput = {
  viewMatrix: IDENTITY as never,
  projMatrix: PERSPECTIVE as never,
  cameraPosition: [0, 0, 0],
  viewport: { width: 4, height: 4, devicePixelRatio: 1 },
  frameIndex: 0,
  frameSeed: 1,
};

function config(environment: Scene['environment']): FrameUniformsConfig {
  return {
    scene: { primitives: [], emitters: [], environment },
    hasEnvMap: environment.kind === 'hdri',
    materialLodDepth: 0,
    backgroundBlur: 0,
    spectralEnabled: false,
    backgroundAlpha: 1,
    bdpt: false,
    bdptMaxLightBounces: 1,
    cameraType: 0,
    transportBounds: {
      center: [0, 0, 0],
      radius: 1,
      min: [0, 0, 0],
      max: [0, 0, 0],
    },
    dof: undefined,
  };
}

function hdri(intensity: number): Scene['environment'] {
  return {
    kind: 'hdri',
    intensity,
    hdri: {
      width: 1,
      height: 1,
      data: new Float32Array([1, 1, 1]),
    },
  };
}

function translatedMatrix(x: number): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, 0, 0, 1,
  ]);
}

function pointTransport(x: number): FrameUniformsConfig['transportBounds'] {
  return {
    center: [x, 0, 0],
    radius: 1,
    min: [x, 0, 0],
    max: [x, 0, 0],
  };
}

describe('packFrameUniforms HDRI float32 boundary', () => {
  it('uploads the realized finite float32 environment intensity', () => {
    const packed = packFrameUniforms(
      FRAME,
      1,
      4,
      4,
      config(hdri(1 / 3)),
    );
    expect(packed.environmentIntensity).toBe(Math.fround(1 / 3));
  });

  it('rejects HDRI intensity overflow and positive underflow before uniform upload', () => {
    expect(() =>
      packFrameUniforms(FRAME, 1, 4, 4, config(hdri(1e39))),
    ).toThrow(/HDRI environment intensity overflows WebGL float32 storage/);
    expect(() =>
      packFrameUniforms(
        FRAME,
        1,
        4,
        4,
        config(hdri(Number.MIN_VALUE)),
      ),
    ).toThrow(/HDRI environment intensity underflows WebGL float32 storage/);
  });
});

describe('packFrameUniforms public float uniform boundary', () => {
  it('packs every public floating-point option and frame dial to its realized float32 value', () => {
    const raw = 1 / 3;
    const cfg: FrameUniformsConfig = {
      ...config({ kind: 'none' }),
      backgroundBlur: raw,
      backgroundAlpha: raw,
      dof: {
        focusDistance: raw,
        bokehSize: raw,
        apertureRotation: raw,
        anamorphicRatio: raw,
      },
    };
    const packed = packFrameUniforms(
      {
        ...FRAME,
        quality: {
          filteredGlossyFactor: raw,
          exposure: raw,
        },
      },
      1,
      4,
      4,
      cfg,
    );

    expect(packed.filterGlossyFactor).toBe(Math.fround(raw));
    expect(packed.backgroundBlur).toBe(Math.fround(raw));
    expect(packed.backgroundAlpha).toBe(Math.fround(raw));
    expect(packed.exposure).toBe(Math.fround(raw));
    expect(packed.dof).toMatchObject({
      focusDistance: Math.fround(raw),
      bokehSize: Math.fround(raw),
      apertureRotation: Math.fround(raw),
      anamorphicRatio: Math.fround(raw),
    });
  });

  it('rejects direct packer bypasses that would upload infinity', () => {
    expect(() =>
      packFrameUniforms(
        { ...FRAME, quality: { exposure: 1e300 } },
        1,
        4,
        4,
        config({ kind: 'none' }),
      ),
    ).toThrow(/exposure overflows WebGL float32 storage/);

    expect(() =>
      packFrameUniforms(
        FRAME,
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          dof: {
            focusDistance: 1,
            bokehSize: 1,
            apertureRotation: 1e300,
          },
        },
      ),
    ).toThrow(/dof.apertureRotation overflows WebGL float32 storage/);
  });

  it('accepts the complete positive-float32 anamorphic-ratio domain', () => {
    const ratios = [
      Math.fround(2 ** -149),
      Math.fround(3.4028234663852886e38),
    ];
    for (const anamorphicRatio of ratios) {
      const packed = packFrameUniforms(
        FRAME,
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          dof: {
            focusDistance: 1,
            bokehSize: 1,
            anamorphicRatio,
          },
        },
      );
      expect(packed.dof?.anamorphicRatio).toBe(anamorphicRatio);
    }
  });

  it('accepts large-aperture/small-focus and both focus-distance f32 extremes', () => {
    for (const focusDistance of [
      Math.fround(2 ** -149),
      Math.fround(1e-4),
      Math.fround(3.4028234663852886e38),
    ]) {
      const packed = packFrameUniforms(
        FRAME,
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          dof: {
            focusDistance,
            bokehSize: 1,
          },
        },
      );
      expect(packed.dof?.focusDistance).toBe(focusDistance);
    }
  });

  it('rejects only active equirectangular DOF at the direct packer boundary', () => {
    const pinhole = packFrameUniforms(
      FRAME,
      1,
      4,
      4,
      {
        ...config({ kind: 'none' }),
        cameraType: 2,
        dof: {
          focusDistance: 1,
          bokehSize: 0,
        },
      },
    );
    expect(pinhole.dof?.bokehSize).toBe(0);

    expect(() =>
      packFrameUniforms(
        FRAME,
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          cameraType: 2,
          dof: {
            focusDistance: 1,
            bokehSize: 1,
          },
        },
      ),
    ).toThrow(/active dof is unsupported for an equirectangular camera/);
  });

  it('requires an active perspective DOF direction to stay off the lens plane', () => {
    const lensCrossingInverseProjection = new Float32Array([
      0, 0, 1, 0,
      0, 1, 0, 0,
      0, 0, 0, 1,
      1, 0, 0, 0,
    ]);
    const lensCrossingProjection = invertMat4(
      lensCrossingInverseProjection,
    );
    expect(lensCrossingProjection).not.toBeNull();
    const input: FrameInput = {
      ...FRAME,
      projMatrix: lensCrossingProjection as never,
    };

    // The pinhole direction has a stable world X component and is valid.
    expect(() =>
      packFrameUniforms(
        input,
        1,
        4,
        4,
        config({ kind: 'none' }),
      ),
    ).not.toThrow();
    expect(() =>
      packFrameUniforms(
        input,
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          dof: {
            focusDistance: 1,
            bokehSize: 1,
          },
        },
      ),
    ).toThrow(/perspective direction domain crosses the camera lens plane/);
  });
});

describe('packFrameUniforms primary-ray transport preflight', () => {
  it('rejects every actual primary-origin path when its scene subtraction overflows float32', () => {
    const positiveEye = Math.fround(3e38);
    const negativeTransport = Math.fround(-3e38);
    const common = {
      ...FRAME,
      cameraPosition: [positiveEye, 0, 0] as const,
    };
    const cases = [
      {
        label: 'perspective inverse-view origin',
        input: {
          ...common,
          viewMatrix: translatedMatrix(-positiveEye) as never,
        },
        cfg: {
          ...config({ kind: 'none' }),
          cameraType: 0 as const,
          transportBounds: pointTransport(negativeTransport),
        },
      },
      {
        label: 'orthographic transformed near-plane origin',
        input: {
          ...FRAME,
          projMatrix: translatedMatrix(-positiveEye) as never,
          cameraPosition: [0, 0, 0] as const,
        },
        cfg: {
          ...config({ kind: 'none' }),
          cameraType: 1 as const,
          transportBounds: pointTransport(negativeTransport),
        },
      },
      {
        label: 'equirectangular inverse-view origin',
        input: {
          ...common,
          viewMatrix: translatedMatrix(-positiveEye) as never,
        },
        cfg: {
          ...config({ kind: 'none' }),
          cameraType: 2 as const,
          transportBounds: pointTransport(negativeTransport),
        },
      },
    ];
    for (const testCase of cases) {
      expect(
        () => packFrameUniforms(testCase.input, 1, 4, 4, testCase.cfg),
        testCase.label,
      ).toThrow(
        /world-space camera origin|camera-to-transport separation|camera-to-transport distance/,
      );
    }
  });

  it('includes the transformed DOF lens disk in the primary-origin domain', () => {
    const cameraX = Math.fround(3.402e38);
    const cfg: FrameUniformsConfig = {
      ...config({ kind: 'none' }),
      cameraType: 1,
      transportBounds: pointTransport(cameraX),
      dof: {
        focusDistance: 1,
        bokehSize: 2e38,
      },
    };
    expect(() =>
      packFrameUniforms(
        {
          ...FRAME,
          viewMatrix: translatedMatrix(-cameraX) as never,
          projMatrix: IDENTITY as never,
          cameraPosition: [cameraX, 0, 0],
        },
        1,
        4,
        4,
        cfg,
      ),
    ).toThrow(/DOF lens-origin maximum/);
  });

  it('mirrors sequential shader aperture scaling at finite-float headroom', () => {
    const bokehSize = Math.fround(1.3486323790845438e35);
    const cameraScale = Math.fround(682936);
    const cameraX = Math.fround(2.942308719621129e38);
    const cameraWorldMatrix = new Float32Array([
      cameraScale, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      cameraX, 0, 0, 1,
    ]);
    const viewMatrix = invertMat4(cameraWorldMatrix);
    expect(viewMatrix).not.toBeNull();
    const realizedCameraWorld = invertMat4(viewMatrix!);
    expect(realizedCameraWorld).not.toBeNull();

    // The prior host expression rounded only once. The emitted GLSL evaluates
    // two float multiplications in source order, and its one-ulp-larger scale
    // is enough to overflow this transformed lens sample.
    const onceFoldedScale = Math.fround(bokehSize * 0.5 * 1e-3);
    const shaderOrderedScale = Math.fround(
      Math.fround(bokehSize * 0.5) * Math.fround(1e-3),
    );
    expect(shaderOrderedScale).toBeGreaterThan(onceFoldedScale);
    const onceFoldedExtent = Math.fround(
      onceFoldedScale * realizedCameraWorld![0]!,
    );
    const shaderOrderedExtent = Math.fround(
      shaderOrderedScale * realizedCameraWorld![0]!,
    );
    expect(
      Number.isFinite(
        Math.fround(realizedCameraWorld![12]! + onceFoldedExtent),
      ),
    ).toBe(true);
    expect(
      Number.isFinite(
        Math.fround(realizedCameraWorld![12]! + shaderOrderedExtent),
      ),
    ).toBe(false);

    expect(() =>
      packFrameUniforms(
        {
          ...FRAME,
          viewMatrix: viewMatrix as never,
          projMatrix: IDENTITY as never,
          cameraPosition: [cameraX, 0, 0],
        },
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          cameraType: 1,
          transportBounds: pointTransport(cameraX),
          dof: {
            focusDistance: 1.5e38,
            bokehSize,
          },
        },
      ),
    ).toThrow(/DOF lens-origin maximum/);
  });

  it('rejects a finite invertible camera transform whose direction domain overflows', () => {
    const xyInverse = 2e-39;
    const zInverse = 1 / 3e38;
    const extremeView = new Float32Array([
      xyInverse, xyInverse, 0, 0,
      xyInverse, -xyInverse, 0, 0,
      0, 0, zInverse, 0,
      0, 0, 0, 1,
    ]);
    expect(() =>
      packFrameUniforms(
        {
          ...FRAME,
          viewMatrix: extremeView as never,
          cameraPosition: [0, 0, 0],
        },
        1,
        4,
        4,
        {
          ...config({ kind: 'none' }),
          cameraType: 2,
        },
      ),
    ).toThrow(/camera direction transform row .*overflows|cannot preserve every unit direction/);
  });
});
