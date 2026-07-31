import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import {
  WEBGL2_F32_MIN_NORMAL,
  computeWebgl2TransportBounds,
  resolveWebgl2SceneRadius,
  validateWebgl2CameraTransportDomain,
  webgl2RayOriginBias,
} from './sceneScalePolicy.js';

function geometryPack(positions: readonly number[]): WorldSpaceMergeResult {
  return {
    positions: new Float32Array(positions),
    positionStrideFloats: 3,
  } as unknown as WorldSpaceMergeResult;
}

function scene(
  emitters: Scene['emitters'] = [],
  environment: Scene['environment'] = { kind: 'none' },
): Scene {
  return { primitives: [], emitters, environment };
}

describe('pt-webgl2 scene-scale ray policy', () => {
  it('preserves finite positive radii at tiny and large scales', () => {
    for (const radius of [1e-30, 1, 1e30]) {
      expect(resolveWebgl2SceneRadius([0, 0, 0], radius)).toBe(radius);
    }
  });

  it('scales the ray offset homogeneously', () => {
    const ordinary = webgl2RayOriginBias(1);
    for (const scale of [1e-30, 1e30]) {
      expect(webgl2RayOriginBias(scale) / ordinary).toBeCloseTo(scale, 12);
    }
  });

  it('uses only an f32-safe coordinate-relative fallback for a degenerate root', () => {
    expect(resolveWebgl2SceneRadius([0, 0, 0], 0)).toBe(
      WEBGL2_F32_MIN_NORMAL,
    );
    expect(resolveWebgl2SceneRadius([1024, 0, 0], 0)).toBe(2 ** -10);
  });

  it('rejects invalid inputs', () => {
    expect(() => resolveWebgl2SceneRadius([0, 0, 0], -1)).toThrow(RangeError);
    expect(() => resolveWebgl2SceneRadius([0, Number.NaN, 0], 1)).toThrow(
      RangeError,
    );
    expect(() => webgl2RayOriginBias(0)).toThrow(RangeError);
  });

  it('rejects scene centers and radii that a WebGL float uniform cannot represent', () => {
    expect(() => resolveWebgl2SceneRadius([0, 0, 0], 3.5e38)).toThrow(
      /scene radius overflows WebGL float32 storage/,
    );
    expect(() => resolveWebgl2SceneRadius([3.5e38, 0, 0], 1)).toThrow(
      /scene center\[0\] overflows WebGL float32 storage/,
    );
    expect(() => resolveWebgl2SceneRadius([0, 0, 0], Number.MIN_VALUE)).toThrow(
      /scene radius underflows WebGL float32 storage/,
    );
    expect(() => webgl2RayOriginBias(3.5e38)).toThrow(
      /ray-bias scene radius overflows WebGL float32 storage/,
    );
  });

  it('rejects a finite light/geometry pair whose shader displacement overflows', () => {
    expect(() =>
      computeWebgl2TransportBounds(
        geometryPack([-3e38, 0, 0]),
        scene([
          {
            id: 'point',
            kind: 'point',
            position: [3e38, 0, 0],
            color: [1, 1, 1],
            intensity: 1,
          },
        ]),
      ),
    ).toThrow(/transport span|bounding-box diagonal/);
  });

  it('reserves the max-float sentinel instead of accepting an ambiguous finite distance', () => {
    expect(() =>
      computeWebgl2TransportBounds(
        geometryPack([0, 0, 0, 3.4e38, 0, 0]),
        scene(),
      ),
    ).toThrow(/guarded transport diameter|max-float distance sentinel/);
  });

  it('rejects analytic area endpoints that overflow compound shader addition', () => {
    expect(() =>
      computeWebgl2TransportBounds(
        geometryPack([0, 0, 0]),
        scene([
          {
            id: 'rect',
            kind: 'rect-area',
            position: [3e38, 0, 0],
            uAxis: [5e37, 0, 0],
            vAxis: [0, 1e-6, 0],
            color: [1, 1, 1],
            intensity: 1,
          },
        ]),
      ),
    ).toThrow(/sampled endpoint/);
  });

  it('keeps non-BDPT scales but rejects unrepresentable distant launch disks', () => {
    const tiny = geometryPack([-1e-30, 0, 0, 1e-30, 0, 0]);
    const huge = geometryPack([-2e19, 0, 0, 2e19, 0, 0]);
    const directional = scene([
      {
        id: 'sun',
        kind: 'directional',
        direction: [0, 1, 0],
        color: [1, 1, 1],
        intensity: 1,
      },
    ]);

    expect(computeWebgl2TransportBounds(tiny, directional).radius).toBeGreaterThan(0);
    expect(computeWebgl2TransportBounds(huge, directional).radius).toBeGreaterThan(0);
    expect(() =>
      computeWebgl2TransportBounds(tiny, directional, { bdpt: true }),
    ).toThrow(/BDPT distant-emitter launch disk/);
    expect(() =>
      computeWebgl2TransportBounds(huge, directional, { bdpt: true }),
    ).toThrow(/BDPT distant-emitter launch disk/);
  });

  it('applies the BDPT launch-disk policy to baked procedural skies', () => {
    const tiny = geometryPack([-1e-30, 0, 0, 1e-30, 0, 0]);
    const proceduralSky: Scene['environment'] = {
      kind: 'procedural-sky',
      sunDirection: [0, 1, 0],
      turbidity: 2,
      rayleigh: 1,
      mieCoefficient: 0.005,
      mieDirectionalG: 0.8,
    };
    expect(() =>
      computeWebgl2TransportBounds(
        tiny,
        scene([], proceduralSky),
        { bdpt: true },
      ),
    ).toThrow(/BDPT distant-emitter launch disk/);
  });

  it('rejects a nondegenerate scene smaller than the minimum ray offset', () => {
    expect(() =>
      computeWebgl2TransportBounds(
        geometryPack([-1e-40, 0, 0, 1e-40, 0, 0]),
        scene(),
      ),
    ).toThrow(/smaller than its minimum representable ray-origin offset/);
  });

  it('rejects geometry and finite analytic endpoints without outward stepRayOrigin headroom', () => {
    const nextFloatBelowMaximum = Math.fround(
      3.4028234663852886e38 * (1 - 2 ** -24),
    );
    expect(() =>
      computeWebgl2TransportBounds(
        geometryPack([nextFloatBelowMaximum, 0, 0]),
        scene(),
      ),
    ).toThrow(/insufficient absolute-coordinate headroom for stepRayOrigin/);
    expect(() =>
      computeWebgl2TransportBounds(
        geometryPack([]),
        scene([
          {
            id: 'point-near-f32-max',
            kind: 'point',
            position: [nextFloatBelowMaximum, 0, 0],
            color: [1, 1, 1],
            intensity: 1,
          },
        ]),
      ),
    ).toThrow(/insufficient absolute-coordinate headroom for stepRayOrigin/);
  });

  it('includes outward-stepped origins in camera-to-transport subtraction safety', () => {
    const cameraX = Math.fround(1.7e38);
    const transportX = Math.fround(-1.702823e38);
    const unsteppedDistance = Math.fround(cameraX - transportX);
    expect(Number.isFinite(unsteppedDistance)).toBe(true);
    expect(unsteppedDistance).toBeLessThan(3.4028234663852886e38);

    expect(() =>
      validateWebgl2CameraTransportDomain(
        { min: [cameraX, 0, 0], max: [cameraX, 0, 0] },
        {
          center: [transportX, 0, 0],
          radius: 1,
          min: [transportX, 0, 0],
          max: [transportX, 0, 0],
        },
      ),
    ).toThrow(/camera-to-transport separation.*overflows/);
  });
});
