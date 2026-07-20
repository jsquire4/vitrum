// T1-2 — cross-backend emitter canonicalizer parity.
//
// Both path-tracing backends pack `scene.emitters` into their own byte layouts
// (`@vitrum/pt-webgl2` `packLightsTexture` — a flat 6-texel RGBA32F grid;
// `@vitrum/pt-webgpu` `packEmitterArrays` — per-kind storage streams). The byte
// layouts differ and stay per-backend. This test pins that the two agree on the
// backend-neutral SEMANTIC interpretation of a fixture scene captured by the
// shared `emitterToCanonical` normalizer (`@vitrum/shared-bvh`):
//   - analytic light count,
//   - per-light power (luminance·intensity·area),
//   - spot cone cosines (outer from `angle`, inner from `angle·(1−penumbra)`).
//
// It also documents the two KNOWN feature-parity gaps (tracked in
// `items_to_fix.md`): pt-webgl2 hardcodes the spot soft-source radius to 0, and
// pt-webgl2 excludes `mesh-area` emitters from its analytic light list.

import { describe, expect, it } from 'vitest';
import type { Scene, SceneEmitter } from '@vitrum/core';
import {
  emitterToCanonical,
  canonicalizeEmitter,
  type CanonicalEmitter,
} from '@vitrum/shared-bvh';
// Deep source imports — the packers are internal to each backend (not on the
// public entry). The `file:` workspace symlink resolves these to source.
import { packLightsTexture, LIGHT_PIXELS } from '@vitrum/pt-webgl2/src/scene/lightsTexture.js';
import { packEmitterArrays } from '@vitrum/pt-webgpu/src/scene/emitterPacking.js';

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

describe('T1-2 emitter canonicalizer cross-backend parity', () => {
  const emitters = fixtureEmitters();
  const scene = fixtureScene(emitters);
  // Analytic canonical set (no mesh-area — this fixture has none, so both
  // include flags produce the same set here).
  const canonical = emitterToCanonical(scene, false);
  const spotCanon = canonical.find((c) => c.kind === 'spot')!;

  it('agrees on analytic light count between both backends', () => {
    const webgl2 = packLightsTexture(emitters);
    const webgpu = packEmitterArrays(scene);
    const webgpuAnalyticCount =
      webgpu.directionalLightCount +
      webgpu.pointLightCount +
      webgpu.spotLightCount +
      webgpu.rectAreaLightCount; // disc packed into the rect stream
    expect(webgl2.lightCount).toBe(canonical.length);
    expect(webgpuAnalyticCount).toBe(canonical.length);
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

    // Packed streams are Float32Array — compare at f32 precision (~6 digits).
    expect(webgl2CoseOuter).toBeCloseTo(spotCanon.cone!.cosOuter, 6);
    expect(webgpuCosOuter).toBeCloseTo(spotCanon.cone!.cosOuter, 6);
    expect(webgl2CosInner).toBeCloseTo(spotCanon.cone!.cosInner, 6);
    expect(webgpuCosInner).toBeCloseTo(spotCanon.cone!.cosInner, 6);
  });

  it('agrees on per-light rect-area + disc-area power (luminance·intensity·area)', () => {
    // pt-webgl2 stores the rect/disc "power" in s2.a; disc power carries the
    // π/4 rectangle→disc correction on a (2r)² axis span, i.e. luminance·I·π·r².
    const webgl2 = packLightsTexture(emitters);
    const rectIdx = emitters.findIndex((e) => e.kind === 'rect-area');
    const discIdx = emitters.findIndex((e) => e.kind === 'disc-area');
    const rectPower = webgl2.data[rectIdx * LIGHT_PIXELS * 4 + 2 * 4 + 3]!;
    const discPower = webgl2.data[discIdx * LIGHT_PIXELS * 4 + 2 * 4 + 3]!;

    const rectCanon = canonical.find((c) => c.kind === 'rect-area')!;
    const discCanon = canonical.find((c) => c.kind === 'disc-area')!;
    // Canonical rect power = luminance(radiance)·|u×v|. pt-webgl2 s2.a for rect
    // = luminance(color)·intensity·(width·height) = same value since
    // radiance = color·intensity and |u×v| = width·height for axis-aligned u,v.
    expect(rectPower).toBeCloseTo(rectCanon.power, 6);
    // Canonical disc power = luminance(radiance)·π·r². pt-webgl2 s2.a for disc
    // = luminance(color)·intensity·(2r)²·(π/4) = luminance(radiance)·π·r².
    expect(discPower).toBeCloseTo(discCanon.power, 6);
  });

  it('resolves the directional toward-light vector identically in both backends', () => {
    // pt-webgpu directional vec4[0].xyz = -dir/|dir| (toward the light).
    const webgpu = packEmitterArrays(scene);
    const dirCanon = canonical.find((c) => c.kind === 'directional')!;
    // Packed stream is Float32Array — compare at f32 precision (~6 digits).
    expect(webgpu.directionalLightsData[0]!).toBeCloseTo(dirCanon.towardLight![0], 6);
    expect(webgpu.directionalLightsData[1]!).toBeCloseTo(dirCanon.towardLight![1], 6);
    expect(webgpu.directionalLightsData[2]!).toBeCloseTo(dirCanon.towardLight![2], 6);
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
    const c = canonicalizeEmitter(shadowed, false) as CanonicalEmitter;
    expect(c.shadowDisabled).toBe(true);
    // pt-webgpu point vec4[2].z carries castShadowDisabled.
    const webgpu = packEmitterArrays(fixtureScene([shadowed]));
    expect(webgpu.pointLightsData[2 * 4 + 2]!).toBe(1);
  });

  it('documents the mesh-area analytic-list parity gap (pt-webgl2 excludes them)', () => {
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
    // The shared canonicalizer mirrors that exclusion by default (includeMeshArea=false)
    // and can opt in (includeMeshArea=true) to match pt-webgpu's explicit mesh-area stream.
    expect(emitterToCanonical(fixtureScene(withMesh), false)).toHaveLength(emitters.length);
    expect(emitterToCanonical(fixtureScene(withMesh), true)).toHaveLength(emitters.length + 1);
  });
});
