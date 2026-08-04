// Cross-backend emitter packing parity.
//
// The two path-tracing backends and walkaround renderer pack `scene.emitters`
// into their own byte layouts
// (`@vitrum/pt-webgl2` `packLightsTexture` — a flat 6-texel RGBA32F grid;
// `@vitrum/pt-webgpu` `packEmitterArrays` — per-kind storage streams;
// `@vitrum/walkaround-hybrid` area emitters — packed EmitterTri streams).
// The byte layouts differ and stay per-backend. This test pins their shared
// core-contract semantics for a canonical fixture scene:
//   - analytic light count,
//   - per-light power (luminance·intensity·area),
//   - spot cone cosines (outer from `angle`, inner from `angle·(1−penumbra)`).
//
// It also pins two deliberate contract/stream boundaries: spots are
// delta-position sources in the core contract, and mesh-area emitters use the
// dedicated triangle-light stream rather than either backend's analytic list.

import { describe, expect, it } from 'vitest';
import type { Scene, SceneEmitter } from '@vitrum/core';
// Deep source imports — the packers are internal to each backend (not on the
// public entry). Relative imports keep this explicitly monorepo-test-only.
import { packLightsTexture, LIGHT_PIXELS } from '../../../pt-webgl2/src/scene/lightsTexture.js';
import { packEmitterArrays } from '../../../pt-webgpu/src/scene/emitterPacking.js';
import {
  collectRectAreaEmitterTrisFromCore,
  packEmitterTrisForDDGI,
} from '../../../walkaround-hybrid/src/restir/emitterHelpers.js';

const REC709 = (rgb: readonly [number, number, number]): number =>
  0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

function emittedPower(
  emitter: Pick<SceneEmitter, 'color' | 'intensity'>,
  area: number,
): number {
  return REC709(emitter.color) * emitter.intensity * area;
}

function fixtureEmitters(): SceneEmitter[] {
  return [
    {
      kind: 'directional',
      id: 'sun',
      color: [1, 0.95, 0.9],
      intensity: 3,
      direction: [0.3, -1, 0.2],
    },
    {
      kind: 'point',
      id: 'lamp',
      color: [1, 1, 1],
      intensity: 5,
      position: [1, 2, 3],
      distance: 10,
      decay: 2,
    },
    {
      kind: 'spot',
      id: 'spot',
      color: [0.8, 0.9, 1],
      intensity: 4,
      position: [0, 4, 0],
      direction: [0, -1, 0],
      angle: 0.6,
      penumbra: 0.3,
    },
    {
      kind: 'rect-area',
      id: 'rect',
      color: [1, 0.8, 0.6],
      intensity: 2,
      position: [-2, 3, 0],
      uAxis: [0.5, 0, 0],
      vAxis: [0, 0, 0.5],
    },
    {
      kind: 'disc-area',
      id: 'disc',
      color: [0.6, 1, 0.7],
      intensity: 6,
      position: [2, 3, 1],
      normal: [0, -1, 0],
      radius: 0.4,
    },
  ];
}

function fixtureScene(emitters: SceneEmitter[]): Scene {
  return {
    primitives: [],
    emitters,
    environment: { kind: 'none' },
  } as unknown as Scene;
}

describe('cross-backend emitter packing parity', () => {
  const emitters = fixtureEmitters();
  const scene = fixtureScene(emitters);
  const expectedAnalyticCount = emitters.filter((emitter) => emitter.kind !== 'mesh-area').length;
  const spot = emitters.find(
    (emitter): emitter is Extract<SceneEmitter, { kind: 'spot' }> => emitter.kind === 'spot',
  )!;

  it('agrees on analytic light count between both backends', () => {
    const webgl2 = packLightsTexture(emitters);
    const webgpu = packEmitterArrays(scene);
    const webgpuAnalyticCount =
      webgpu.directionalLightCount +
      webgpu.pointLightCount +
      webgpu.spotLightCount +
      webgpu.rectAreaLightCount; // disc packed into the rect stream
    expect(webgl2.lightCount).toBe(expectedAnalyticCount);
    expect(webgpuAnalyticCount).toBe(expectedAnalyticCount);
  });

  it('agrees on the spot cone cosines (outer + inner) across both backends', () => {
    // pt-webgl2: spot s4.coneCos = cos(angle); s5.r penumbraCos = cos(angle·(1−penumbra)).
    const webgl2 = packLightsTexture(emitters);
    const spotIndex = emitters.findIndex((e) => e.kind === 'spot');
    const base = spotIndex * LIGHT_PIXELS * 4;
    const webgl2CoseOuter = webgl2.data[base + 4 * 4 + 3]!; // s4.a (channel 19)
    const webgl2CosInner = webgl2.data[base + 5 * 4 + 0]!; // s5.r (channel 20)

    // pt-webgpu: spot vec4[1].w = cosOuter, vec4[2].w = cosInner.
    const webgpu = packEmitterArrays(scene);
    const webgpuCosOuter = webgpu.spotLightsData[1 * 4 + 3]!;
    const webgpuCosInner = webgpu.spotLightsData[2 * 4 + 3]!;
    const expectedCosOuter = Math.cos(spot.angle);
    const expectedCosInner = Math.cos(spot.angle * (1 - (spot.penumbra ?? 0)));

    // Packed streams are Float32Array — compare at f32 precision (~6 digits).
    expect(webgl2CoseOuter).toBeCloseTo(expectedCosOuter, 6);
    expect(webgpuCosOuter).toBeCloseTo(expectedCosOuter, 6);
    expect(webgl2CosInner).toBeCloseTo(expectedCosInner, 6);
    expect(webgpuCosInner).toBeCloseTo(expectedCosInner, 6);
  });

  it('pins the core spot emitter as a delta-position source in both backend packers', () => {
    const webgl2 = packLightsTexture(emitters);
    const spotIndex = emitters.findIndex((e) => e.kind === 'spot');
    const base = spotIndex * LIGHT_PIXELS * 4;

    expect('radius' in spot).toBe(false);
    expect(webgl2.data[base + 3 * 4 + 3]).toBe(0); // s3.a: source area
    // s4.r is the represented light-selection PMF, not a geometric radius.
    expect(webgl2.data[base + 4 * 4 + 0]).toBeGreaterThan(0);

    const webgpu = packEmitterArrays(scene);
    expect(webgpu.spotLightCount).toBe(1);
    // The WebGPU spot record has no radius/area payload; one spot is exactly
    // the canonical fixed-size delta-light record.
    expect(webgpu.spotLightsData).toHaveLength(16);
  });

  it('A/B/C agrees on the core rect surface, area, Le/power, and emitting side', () => {
    // pt-webgl2 stores the rect/disc "power" in s2.a; disc power carries the
    // π/4 rectangle→disc correction on a (2r)² axis span, i.e. luminance·I·π·r².
    const webgl2 = packLightsTexture(emitters);
    const webgpu = packEmitterArrays(scene);
    const walkaround = packEmitterTrisForDDGI(
      collectRectAreaEmitterTrisFromCore(scene),
    );
    const rectIdx = emitters.findIndex((e) => e.kind === 'rect-area');
    const discIdx = emitters.findIndex((e) => e.kind === 'disc-area');
    const rectBase = rectIdx * LIGHT_PIXELS * 4;
    const rectPower = webgl2.data[rectIdx * LIGHT_PIXELS * 4 + 2 * 4 + 3]!;
    const discPower = webgl2.data[discIdx * LIGHT_PIXELS * 4 + 2 * 4 + 3]!;

    const rect = emitters.find(
      (emitter): emitter is Extract<SceneEmitter, { kind: 'rect-area' }> =>
        emitter.kind === 'rect-area',
    )!;
    const disc = emitters.find(
      (emitter): emitter is Extract<SceneEmitter, { kind: 'disc-area' }> =>
        emitter.kind === 'disc-area',
    )!;
    const coreHalfExtentCross = Math.hypot(
      rect.uAxis[1] * rect.vAxis[2] - rect.uAxis[2] * rect.vAxis[1],
      rect.uAxis[2] * rect.vAxis[0] - rect.uAxis[0] * rect.vAxis[2],
      rect.uAxis[0] * rect.vAxis[1] - rect.uAxis[1] * rect.vAxis[0],
    );
    const rectArea = 4 * coreHalfExtentCross;
    const glFullU = Array.from(webgl2.data.slice(rectBase + 8, rectBase + 11));
    const glFullV = Array.from(webgl2.data.slice(rectBase + 12, rectBase + 15));
    const gpuHalfU = Array.from(webgpu.rectAreaLightsData.slice(4, 7));
    const gpuHalfV = Array.from(webgpu.rectAreaLightsData.slice(8, 11));
    const webglArea = webgl2.data[rectBase + 15]!;
    const webgpuArea = 4 * Math.hypot(
      gpuHalfU[1]! * gpuHalfV[2]! - gpuHalfU[2]! * gpuHalfV[1]!,
      gpuHalfU[2]! * gpuHalfV[0]! - gpuHalfU[0]! * gpuHalfV[2]!,
      gpuHalfU[0]! * gpuHalfV[1]! - gpuHalfU[1]! * gpuHalfV[0]!,
    );
    const webgpuRadiance = Array.from(webgpu.rectAreaLightsData.slice(12, 15)) as [
      number,
      number,
      number,
    ];
    const webgpuPower = REC709(webgpuRadiance) * webgpuArea;
    const walkaroundArea = walkaround.data[15]! + walkaround.data[20 + 15]!;
    const walkaroundRadiance = Array.from(walkaround.data.slice(16, 19)) as [
      number,
      number,
      number,
    ];
    const walkaroundPower = REC709(walkaroundRadiance) * walkaroundArea;

    expect(glFullU).toEqual(gpuHalfU.map((component) => 2 * component));
    expect(glFullV).toEqual(gpuHalfV.map((component) => 2 * component));
    // The canonical fixture packs two rect triangles followed by the disc's
    // equal-area 32-triangle fan. This comparison intentionally consumes only
    // the exact rect representation; the fan is area-equivalent but its
    // adjusted-radius polygon is not pointwise identical to an analytic disc.
    expect(walkaround.count).toBe(34);
    expect(webglArea).toBeCloseTo(rectArea, 6);
    expect(webgpuArea).toBeCloseTo(rectArea, 6);
    expect(walkaroundArea).toBeCloseTo(rectArea, 6);
    expect(rectPower).toBeCloseTo(emittedPower(rect, rectArea), 6);
    expect(rectPower).toBeCloseTo(webgpuPower, 6);
    expect(walkaroundRadiance).toEqual(webgpuRadiance);
    expect(rectPower).toBeCloseTo(walkaroundPower, 6);
    expect(discPower).toBeCloseTo(emittedPower(disc, Math.PI * disc.radius * disc.radius), 6);

    // The analytic wire formats use different axis conventions, but every
    // random pair lands on the same authored core rectangle.
    for (const [xi, yi] of [[0, 0], [0.17, 0.73], [0.5, 0.5], [1, 1]] as const) {
      const glPoint = rect.position.map(
        (center, axis) =>
          center + glFullU[axis]! * (xi - 0.5) + glFullV[axis]! * (yi - 0.5),
      );
      const gpuPoint = rect.position.map(
        (center, axis) =>
          center + gpuHalfU[axis]! * (2 * xi - 1) + gpuHalfV[axis]! * (2 * yi - 1),
      );
      glPoint.forEach((component, axis) => {
        expect(component).toBeCloseTo(gpuPoint[axis]!, 7);
      });
    }

    // Walkaround tessellates the same rect into two production EmitterTri
    // records. Its packed vertex set must be exactly the four analytic corners.
    const cornerKey = (v: ArrayLike<number>): string =>
      Array.from(v, (component) => component.toFixed(7)).join(',');
    const analyticCorners = new Set(
      [[0, 0], [1, 0], [1, 1], [0, 1]].map(([xi, yi]) =>
        cornerKey(rect.position.map(
          (center, axis) =>
            center + gpuHalfU[axis]! * (2 * xi! - 1) + gpuHalfV[axis]! * (2 * yi! - 1),
        )),
      ),
    );
    const walkaroundVertices = new Set<string>();
    for (let tri = 0; tri < 2; tri += 1) {
      const base = tri * 20;
      walkaroundVertices.add(cornerKey(walkaround.data.slice(base, base + 3)));
      walkaroundVertices.add(cornerKey(walkaround.data.slice(base + 4, base + 7)));
      walkaroundVertices.add(cornerKey(walkaround.data.slice(base + 8, base + 11)));
    }
    expect(walkaroundVertices).toEqual(analyticCorners);

    const normalizeCross = (u: readonly number[], v: readonly number[]): number[] => {
      const cross = [
        u[1]! * v[2]! - u[2]! * v[1]!,
        u[2]! * v[0]! - u[0]! * v[2]!,
        u[0]! * v[1]! - u[1]! * v[0]!,
      ];
      const length = Math.hypot(...cross);
      return cross.map((component) => component / length);
    };
    const webglFront = normalizeCross(glFullU, glFullV);
    const webgpuFront = normalizeCross(gpuHalfU, gpuHalfV);
    const walkaroundFront = Array.from(walkaround.data.slice(12, 15));
    webglFront.forEach((component, axis) => {
      expect(component).toBeCloseTo(webgpuFront[axis]!, 7);
      expect(component).toBeCloseTo(walkaroundFront[axis]!, 7);
      expect(walkaround.data[20 + 12 + axis]).toBeCloseTo(component, 7);
    });
  });

  it('resolves the directional toward-light vector identically in both backends', () => {
    // pt-webgpu directional vec4[0].xyz = -dir/|dir| (toward the light).
    const webgpu = packEmitterArrays(scene);
    const directional = emitters.find(
      (emitter): emitter is Extract<SceneEmitter, { kind: 'directional' }> =>
        emitter.kind === 'directional',
    )!;
    const directionLength = Math.hypot(...directional.direction);
    // Packed stream is Float32Array — compare at f32 precision (~6 digits).
    expect(webgpu.directionalLightsData[0]).toBeCloseTo(
      -directional.direction[0] / directionLength,
      6,
    );
    expect(webgpu.directionalLightsData[1]).toBeCloseTo(
      -directional.direction[1] / directionLength,
      6,
    );
    expect(webgpu.directionalLightsData[2]).toBeCloseTo(
      -directional.direction[2] / directionLength,
      6,
    );
  });

  it('resolves SHADOW-01 castShadow:false identically', () => {
    const shadowed: SceneEmitter = {
      kind: 'point',
      id: 'noShadow',
      color: [1, 1, 1],
      intensity: 1,
      position: [0, 0, 0],
      castShadow: false,
    };
    expect(shadowed.castShadow).toBe(false);
    // pt-webgpu point vec4[2].z carries castShadowDisabled.
    const webgpu = packEmitterArrays(fixtureScene([shadowed]));
    expect(webgpu.pointLightsData[2 * 4 + 2]).toBe(1);
  });

  it('keeps mesh-area emitters out of the analytic list for the dedicated triangle-light stream', () => {
    const meshEmitter: SceneEmitter = {
      kind: 'mesh-area',
      id: 'panel',
      color: [1, 1, 1],
      intensity: 1,
      meshId: 'nonexistent',
    };
    const withMesh = [...emitters, meshEmitter];
    // pt-webgl2 filters mesh-area out of the analytic light list.
    const webgl2 = packLightsTexture(withMesh);
    expect(webgl2.lightCount).toBe(emitters.length);
    const webgpu = packEmitterArrays(fixtureScene(withMesh));
    const webgpuAnalyticCount =
      webgpu.directionalLightCount +
      webgpu.pointLightCount +
      webgpu.spotLightCount +
      webgpu.rectAreaLightCount;
    expect(webgpuAnalyticCount).toBe(emitters.length);
    expect(webgpu.meshAreaLightCount).toBe(0);
  });
});
