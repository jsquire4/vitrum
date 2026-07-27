/**
 * H55 proof oracle — pt-webgpu normal-map tangent-frame behavior.
 *
 * materialTextures.test.ts already pins descriptor packing and WGSL linkage.
 * This file independently checks the math that makes normal maps faithful:
 * authored tangent.xyzw handedness, UV-gradient fallback, normalScale, layer
 * normal face selection, and clearcoat-normal scaling.
 */
import { describe, expect, it } from 'vitest';

import {
  MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
} from '../scene/materialTextures.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';

type Vec2 = readonly [number, number];
type Vec3 = readonly [number, number, number];
type Vec4 = readonly [number, number, number, number];

interface TriangleFixture {
  readonly positions: readonly [Vec3, Vec3, Vec3];
  readonly uvs: readonly [Vec4, Vec4, Vec4];
  /** Compact GPU UV planes for slots 2+, in slot order. */
  readonly compactUvPlanes?: readonly (readonly [Vec2, Vec2, Vec2])[];
  readonly tangents?: readonly [Vec4, Vec4, Vec4];
}

interface TangentFrame {
  readonly tangent: Vec3;
  readonly bitangent: Vec3;
  readonly valid: boolean;
}

const TRI: TriangleFixture = {
  positions: [
    [0, 0, 0],
    [2, 0, 0],
    [0, 2, 0],
  ],
  uvs: [
    [0, 0, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
  ],
};

const NORMAL: Vec3 = [0, 0, 1];

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

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len <= 1e-12) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function vecClose(actual: Vec3, expected: Vec3, precision = 10): void {
  expect(actual[0]).toBeCloseTo(expected[0], precision);
  expect(actual[1]).toBeCloseTo(expected[1], precision);
  expect(actual[2]).toBeCloseTo(expected[2], precision);
}

function baryBlend3(values: readonly [Vec3, Vec3, Vec3], baryVW: Vec2): Vec3 {
  const v = baryVW[0];
  const w = baryVW[1];
  const u = 1 - v - w;
  return [
    values[0][0] * u + values[1][0] * v + values[2][0] * w,
    values[0][1] * u + values[1][1] * v + values[2][1] * w,
    values[0][2] * u + values[1][2] * v + values[2][2] * w,
  ];
}

function baryBlendScalar(values: readonly [number, number, number], baryVW: Vec2): number {
  const v = baryVW[0];
  const w = baryVW[1];
  const u = 1 - v - w;
  return values[0] * u + values[1] * v + values[2] * w;
}

function buildShadingTangentFrame(
  fixture: TriangleFixture,
  baryVW: Vec2,
  normal: Vec3,
  gpuUvSlot = 0,
  instanceDeterminantSign = 1,
): TangentFrame {
  if (gpuUvSlot === 0 && fixture.tangents) {
    let tangent = baryBlend3([
      [fixture.tangents[0][0], fixture.tangents[0][1], fixture.tangents[0][2]],
      [fixture.tangents[1][0], fixture.tangents[1][1], fixture.tangents[1][2]],
      [fixture.tangents[2][0], fixture.tangents[2][1], fixture.tangents[2][2]],
    ], baryVW);
    const handednessRaw = baryBlendScalar([
      fixture.tangents[0][3],
      fixture.tangents[1][3],
      fixture.tangents[2][3],
    ], baryVW);
    if (Math.hypot(...tangent) > 1e-8 && Math.abs(handednessRaw) > 0.5) {
      tangent = sub(tangent, scale(normal, dot(normal, tangent)));
      tangent = normalize(tangent);
      if (Math.hypot(...tangent) > 1e-8) {
        const handedness = (handednessRaw >= 0 ? 1 : -1) * instanceDeterminantSign;
        return {
          tangent,
          bitangent: scale(cross(normal, tangent), handedness),
          valid: true,
        };
      }
    }
  }

  const p0 = fixture.positions[0];
  const e1 = sub(fixture.positions[1], p0);
  const e2 = sub(fixture.positions[2], p0);
  const uvForVertex = (vertex: 0 | 1 | 2): Vec2 => {
    if (gpuUvSlot === 0) return [fixture.uvs[vertex][0], fixture.uvs[vertex][1]];
    if (gpuUvSlot === 1) return [fixture.uvs[vertex][2], fixture.uvs[vertex][3]];
    return fixture.compactUvPlanes?.[gpuUvSlot - 2]?.[vertex] ?? [0, 0];
  };
  const uv0 = uvForVertex(0);
  const uv1 = uvForVertex(1);
  const uv2 = uvForVertex(2);
  const duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]] as const;
  const duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]] as const;
  const det = duv1[0] * duv2[1] - duv2[0] * duv1[1];
  if (Math.abs(det) < 1e-10) return { tangent: [0, 0, 0], bitangent: [0, 0, 0], valid: false };
  const f = 1 / det;
  let tangent = scale(sub(scale(e1, duv2[1]), scale(e2, duv1[1])), f);
  let bitangent = scale(add(scale(e1, -duv2[0]), scale(e2, duv1[0])), f);
  tangent = sub(tangent, scale(normal, dot(normal, tangent)));
  tangent = normalize(tangent);
  if (Math.hypot(...tangent) < 1e-8) return { tangent: [0, 0, 0], bitangent: [0, 0, 0], valid: false };
  bitangent = sub(bitangent, scale(normal, dot(normal, bitangent)));
  const handedness = dot(cross(normal, tangent), bitangent) >= 0 ? 1 : -1;
  return {
    tangent,
    bitangent: scale(cross(normal, tangent), handedness),
    valid: true,
  };
}

function applyTangentSpaceNormal(frame: TangentFrame, geomNormal: Vec3, texelRgb: Vec3, normalScale: number): Vec3 {
  expect(frame.valid).toBe(true);
  const tn: Vec3 = [
    (texelRgb[0] * 2 - 1) * normalScale,
    (texelRgb[1] * 2 - 1) * normalScale,
    texelRgb[2] * 2 - 1,
  ];
  return normalize(add(add(scale(frame.tangent, tn[0]), scale(frame.bitangent, tn[1])), scale(geomNormal, tn[2])));
}

interface NormalDescriptorChoice {
  readonly idx: number;
  readonly scale: number;
  readonly uvMetaOffset: number;
  readonly uvFitScale: Vec2;
  readonly wrapMode: Vec2;
}

function chooseNormalDescriptor(isFrontFace: boolean): NormalDescriptorChoice {
  const topLevel: NormalDescriptorChoice = {
    idx: 5,
    scale: 0.4,
    uvMetaOffset: MATERIAL_TEX_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP * 2,
    uvFitScale: [0.9, 0.8],
    wrapMode: [0, 2],
  };
  const front = {
    idx: 11,
    scale: 0.75,
    uvFitScale: [0.6, 0.5] as Vec2,
    wrapMode: [1, 0] as Vec2,
  };
  const back = {
    idx: 17,
    scale: 0.25,
    uvFitScale: [0.4, 0.3] as Vec2,
    wrapMode: [2, 1] as Vec2,
  };

  const selected = isFrontFace ? front : back;
  if (selected.idx < 0) return topLevel;
  return {
    idx: selected.idx,
    scale: selected.scale,
    uvMetaOffset: MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET +
      (isFrontFace ? 0 : MATERIAL_TEX_UV_META_VEC4S_PER_MAP),
    uvFitScale: selected.uvFitScale,
    wrapMode: selected.wrapMode,
  };
}

describe('pt-webgpu normal-map tangent-frame oracle', () => {
  it('keeps the production WGSL linked to the oracle-covered tangent-frame branches', () => {
    for (const snippet of [
      'fn buildShadingTangentFrame(triIndex: u32, baryVW: vec2f, normal: vec3f, gpuUvSlot: u32, instanceIndex: u32)',
      'if (gpuUvSlot == 0u && tri.x < arrayLength(&meshTangents)',
      'let uv0 = materialUvForVertex(tri.x, gpuUvSlot);',
      'let handednessRaw = ta.w * u + tb.w * v + tc.w * w;',
      'frame.bitangent = cross(normal, tangent) * handedness;',
      'let f = 1.0 / det;',
      'var normalScale = materialTexDescriptors[base + 5u].w;',
      'tn.x = tn.x * normalScale;',
      'normalUvMetaOffset = select(MATERIAL_TEX_UV_BACK_LAYER_NORMAL, MATERIAL_TEX_UV_FRONT_LAYER_NORMAL, isFrontFace);',
      'let normalGpuUvSlot = u32(materialTexDescriptors[base + normalUvMetaOffset].x);',
      'let clearcoatNormalGpuUvSlot = u32(',
      'let bumpGpuUvSlot = u32(uvMeta.x);',
      'let clearcoatNormalScale = materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].y;',
    ]) {
      expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(snippet);
    }
    expect(MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET).toBeLessThan(MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET);
    expect(MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET).toBeLessThan(MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET);
  });

  it('authored tangent.xyzw carries handedness into the bitangent before normal-map perturbation', () => {
    const frame = buildShadingTangentFrame({
      ...TRI,
      tangents: [
        [1, 0, 0, -1],
        [1, 0, 0, -1],
        [1, 0, 0, -1],
      ],
    }, [0.25, 0.25], NORMAL);

    vecClose(frame.tangent, [1, 0, 0]);
    vecClose(frame.bitangent, [0, -1, 0]);
    vecClose(applyTangentSpaceNormal(frame, NORMAL, [0.75, 0.25, 1], 1), normalize([0.5, 0.5, 1]));
  });

  it('falls back to the UV-gradient tangent and rejects degenerate UV gradients', () => {
    const frame = buildShadingTangentFrame(TRI, [0.2, 0.3], NORMAL);
    vecClose(frame.tangent, [1, 0, 0]);
    vecClose(frame.bitangent, [0, 1, 0]);

    const degenerate = buildShadingTangentFrame({
      ...TRI,
      uvs: [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
    }, [0.2, 0.3], NORMAL);
    expect(degenerate.valid).toBe(false);
  });

  it('normal, bump, and clearcoat-normal frames follow the descriptor-selected compact UV slot', () => {
    const fixture: TriangleFixture = {
      ...TRI,
      uvs: [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
      compactUvPlanes: [[
        [0, 0],
        [-1, 0],
        [0, 1],
      ]],
      tangents: [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
      ],
    };

    // Only UV0 may consume authored glTF tangents. UV1 and a compact sparse
    // authored set (GPU slot 2) derive independent orientations.
    const uv0Frame = buildShadingTangentFrame(fixture, [0.2, 0.3], NORMAL, 0);
    const uv1Frame = buildShadingTangentFrame(fixture, [0.2, 0.3], NORMAL, 1);
    const sparseMirroredFrame = buildShadingTangentFrame(fixture, [0.2, 0.3], NORMAL, 2);
    vecClose(uv0Frame.tangent, [1, 0, 0]);
    vecClose(uv1Frame.tangent, [0, 1, 0]);
    vecClose(uv1Frame.bitangent, [1, 0, 0]);
    vecClose(sparseMirroredFrame.tangent, [-1, 0, 0]);
    vecClose(sparseMirroredFrame.bitangent, [0, 1, 0]);
    expect(dot(
      cross(sparseMirroredFrame.tangent, sparseMirroredFrame.bitangent),
      NORMAL,
    )).toBeCloseTo(-1);
  });

  it('flips authored tangent handedness for a reflected TLAS instance', () => {
    const fixture: TriangleFixture = {
      ...TRI,
      tangents: [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
      ],
    };
    const frame = buildShadingTangentFrame(
      fixture, [0.2, 0.3], NORMAL, 0, -1,
    );
    vecClose(frame.tangent, [1, 0, 0]);
    vecClose(frame.bitangent, [0, -1, 0]);
  });

  it('normalScale damps tangent-space xy tilt before normalization', () => {
    const frame = buildShadingTangentFrame(TRI, [0.2, 0.2], NORMAL);
    const full = applyTangentSpaceNormal(frame, NORMAL, [0.8, 0.5, 0.9], 1);
    const damped = applyTangentSpaceNormal(frame, NORMAL, [0.8, 0.5, 0.9], 0.25);
    vecClose(full, normalize([0.6, 0, 0.8]));
    vecClose(damped, normalize([0.15, 0, 0.8]));
    expect(damped[2]).toBeGreaterThan(full[2]);
  });

  it('face-selects layer normal descriptors ahead of the top-level normal map', () => {
    const front = chooseNormalDescriptor(true);
    expect(front).toEqual({
      idx: 11,
      scale: 0.75,
      uvMetaOffset: MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET,
      uvFitScale: [0.6, 0.5],
      wrapMode: [1, 0],
    });

    const back = chooseNormalDescriptor(false);
    expect(back).toEqual({
      idx: 17,
      scale: 0.25,
      uvMetaOffset: MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
      uvFitScale: [0.4, 0.3],
      wrapMode: [2, 1],
    });
  });

  it('clearcoat normal maps use the same tangent frame with an independent scale', () => {
    const frame = buildShadingTangentFrame(TRI, [0.25, 0.25], NORMAL);
    const baseNormal = applyTangentSpaceNormal(frame, NORMAL, [0.5, 0.75, 1], 1);
    const clearcoatNormal = applyTangentSpaceNormal(frame, NORMAL, [0.5, 0.75, 1], 0.4);
    vecClose(baseNormal, normalize([0, 0.5, 1]));
    vecClose(clearcoatNormal, normalize([0, 0.2, 1]));
    expect(clearcoatNormal[2]).toBeGreaterThan(baseNormal[2]);
  });
});
