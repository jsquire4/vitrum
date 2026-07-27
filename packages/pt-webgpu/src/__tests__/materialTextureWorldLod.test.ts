import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';

type Vec3 = readonly [number, number, number];

function transformPoint(point: Vec3, scale: Vec3, translation: Vec3): Vec3 {
  return [
    point[0] * scale[0] + translation[0],
    point[1] * scale[1] + translation[1],
    point[2] * scale[2] + translation[2],
  ];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
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

function footprint(
  local: readonly [Vec3, Vec3, Vec3],
  baryVW: readonly [number, number],
  scale: Vec3,
  translation: Vec3,
  camera: Vec3,
): { readonly area: number; readonly cameraDistance: number } {
  const [pa, pb, pc] = local.map((point) => transformPoint(point, scale, translation)) as [Vec3, Vec3, Vec3];
  const [v, w] = baryVW;
  const u = 1 - v - w;
  const hit: Vec3 = [
    pa[0] * u + pb[0] * v + pc[0] * w,
    pa[1] * u + pb[1] * v + pc[1] * w,
    pa[2] * u + pb[2] * v + pc[2] * w,
  ];
  return {
    area: 0.5 * length(cross(sub(pb, pa), sub(pc, pa))),
    cameraDistance: length(sub(hit, camera)),
  };
}

describe('material texture LOD world-space footprint', () => {
  it('transforms all BLAS-local vertices before area and camera-distance evaluation', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain('fn materialTexturePointToWorld(point: vec3f, instanceIndex: u32) -> vec3f');
    expect(wgsl).toContain('tlasInstanceLocalToWorld[m + 3u]');
    expect(wgsl).toContain('fn materialTextureWorldFootprint(tri: vec4u, baryVW: vec2f, instanceIndex: u32) -> vec2f');
    expect(wgsl).toContain('let footprint = materialTextureWorldFootprint(tri, baryVW, instanceIndex);');
    expect(wgsl).toContain('let worldHitPos = pa * u + pb * v + pc * w;');
    expect(wgsl).not.toContain('let hitPos = pa * u + pb * v + pc * w;');
  });

  it('threads the hit instance through main, ReSTIR-PT, and BDPT material sampling', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex)',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'sampleBaseColorTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex)',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex)',
    );
  });

  it('uses non-uniform scale and is invariant to a shared world translation', () => {
    const tri: readonly [Vec3, Vec3, Vec3] = [[0, 0, 0], [2, 0, 0], [0, 3, 0]];
    const baryVW = [0.25, 0.5] as const;
    const scale: Vec3 = [4, 0.5, 2];
    const translation: Vec3 = [100, -20, 7];
    const camera: Vec3 = [105, -18, 10];
    const world = footprint(tri, baryVW, scale, translation, camera);
    const local = footprint(tri, baryVW, [1, 1, 1], [0, 0, 0], [0, 0, 0]);

    expect(local.area).toBeCloseTo(3, 12);
    expect(world.area).toBeCloseTo(6, 12);

    const delta: Vec3 = [-31, 400, 0.75];
    const shiftedTranslation: Vec3 = [
      translation[0] + delta[0],
      translation[1] + delta[1],
      translation[2] + delta[2],
    ];
    const shiftedCamera: Vec3 = [
      camera[0] + delta[0],
      camera[1] + delta[1],
      camera[2] + delta[2],
    ];
    const shifted = footprint(tri, baryVW, scale, shiftedTranslation, shiftedCamera);
    expect(shifted.area).toBeCloseTo(world.area, 12);
    expect(shifted.cameraDistance).toBeCloseTo(world.cameraDistance, 12);

    const localHit: Vec3 = [0.5, 1.5, 0];
    const oldMixedSpaceDistance = length(sub(localHit, camera));
    expect(oldMixedSpaceDistance).toBeGreaterThan(world.cameraDistance * 10);
  });
});
