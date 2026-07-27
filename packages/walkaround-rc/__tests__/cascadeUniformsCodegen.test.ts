import { describe, expect, it } from 'vitest';
import {
  buildCascadeUniformDataInto,
  bindingFieldRegistryFields,
  type CascadeUniformInputs,
} from '../src/cascadeDispatch.js';
import {
  CASCADE_UNIFORMS_BYTE_SIZE,
  CascadeUniformsOffset,
} from '../src/cascadeUniformsLayout.generated.js';

/**
 * T2-C (D16-8) — CascadeUniforms codegen + binding-signature field registry.
 *
 * (1) Byte golden: buildCascadeUniformDataInto must remain byte-identical after
 *     switching from raw magic slot indices (ui[29..31] etc.) to the generated
 *     CascadeUniformsOffset named offsets. The u32 arrays below pin the exact
 *     packed bytes, including the default interface budget in the former spare
 *     word at offset 136.
 * (2) Codegen offsets match the WGSL struct field order (probeRayCast.wgsl.ts).
 * (3) Binding-signature field registry pins the exact field set (a changed
 *     field must invalidate the cached bind groups; an unchanged set must not).
 */

const GOLDEN_INPUTS: CascadeUniformInputs = {
  probeOriginWorld: [1.5, -2.25, 3.75],
  roomSize: [10, 20, 30],
  sunDir: [0, 1, 0],
  sunColor: [0.9, 0.8, 0.7],
  sunCastShadowDisabled: true,
  sunAngularRadius: 0.05,
  envIntensity: 1.0,
  envRotationY: 0.75,
  scalarSkyRadiance: [0.25, 0.5, 1.5],
  hasDirectionalEnvironment: true,
  frameSeed: 12345,
  triIntersectEpsilon: 1e-5,
  bvhMode: 1,
  tlasNodeCount: 42,
  emitterCount: 7,
  lightCount: 3,
};

// Pre-refactor golden extended with the default interface budget at word 34.
const CASCADE_GOLDEN_U32: Record<number, number[]> = {
  0: [1069547520,3222274048,1081081856,0,1092616192,1101004800,1106247680,0,16,9,14,16,4,0,1094713344,0,0,1065353216,0,1028443341,1063675494,1061997773,1060320051,1065353216,12345,4,925353388,1,42,7,3,1,0,0,8,1061158912,1048576000,1056964608,1069547520,1],
  1: [1069547520,3222274048,1081081856,0,1092616192,1101004800,1106247680,0,8,5,7,64,8,1094713344,1108344832,1,0,1065353216,0,1028443341,1063675494,1061997773,1060320051,1065353216,12345,4,925353388,1,42,7,3,1,0,0,8,1061158912,1048576000,1056964608,1069547520,1],
  2: [1069547520,3222274048,1081081856,0,1092616192,1101004800,1106247680,0,4,3,4,256,16,1108344832,1119879168,2,0,1065353216,0,1028443341,1063675494,1061997773,1060320051,1065353216,12345,4,925353388,1,42,7,3,1,0,0,8,1061158912,1048576000,1056964608,1069547520,1],
};

describe('CascadeUniforms codegen byte golden', () => {
  for (const k of [0, 1, 2]) {
    it(`cascade level ${k} packs byte-identically to the pre-codegen layout`, () => {
      const d = new Float32Array(40);
      buildCascadeUniformDataInto(d, k, GOLDEN_INPUTS);
      expect(Array.from(new Uint32Array(d.buffer))).toEqual(CASCADE_GOLDEN_U32[k]);
    });
  }

  it('generated field offsets match the WGSL CascadeUniforms struct order', () => {
    // Offsets used by the packer — these mirror struct CascadeUniforms in
    // probeRayCast.wgsl.ts; a field reorder in the shader (or generator) trips this.
    expect(CascadeUniformsOffset.probeOriginWorld).toBe(0);
    expect(CascadeUniformsOffset.roomSize).toBe(16);
    expect(CascadeUniformsOffset.probeCount).toBe(32);
    expect(CascadeUniformsOffset.raysPerProbe).toBe(44);
    expect(CascadeUniformsOffset.rayGridSize).toBe(48);
    expect(CascadeUniformsOffset.intervalNear).toBe(52);
    expect(CascadeUniformsOffset.intervalFar).toBe(56);
    expect(CascadeUniformsOffset.cascadeIndex).toBe(60);
    expect(CascadeUniformsOffset.sunDirection).toBe(64);
    expect(CascadeUniformsOffset.sunAngularRadius).toBe(76);
    expect(CascadeUniformsOffset.sunColor).toBe(80);
    expect(CascadeUniformsOffset.envIntensity).toBe(92);
    expect(CascadeUniformsOffset.frameSeed).toBe(96);
    expect(CascadeUniformsOffset.lastCascade).toBe(100);
    expect(CascadeUniformsOffset.triIntersectEpsilon).toBe(104);
    expect(CascadeUniformsOffset.bvhMode).toBe(108);
    expect(CascadeUniformsOffset.tlasNodeCount).toBe(112);
    // The three former magic slots ui[29..31] are now named byte offsets.
    expect(CascadeUniformsOffset.emitterCount).toBe(116);        // was ui[29]
    expect(CascadeUniformsOffset.lightCount).toBe(120);          // was ui[30]
    expect(CascadeUniformsOffset.sunCastShadowDisabled).toBe(124); // was ui[31]
    expect(CascadeUniformsOffset.emitterDataWordOffset).toBe(128);
    expect(CascadeUniformsOffset.emitterAliasWordOffset).toBe(132);
    expect(CascadeUniformsOffset.transmittedInterfaceBudget).toBe(136);
    expect(CascadeUniformsOffset.envRotationY).toBe(140);
    expect(CascadeUniformsOffset.scalarSkyRadiance).toBe(144);
    expect(CascadeUniformsOffset.hasDirectionalEnv).toBe(156);
    expect(CASCADE_UNIFORMS_BYTE_SIZE).toBe(160);
  });

  it('every offset is 4-byte aligned (safe for f32/u32 word indexing)', () => {
    for (const off of Object.values(CascadeUniformsOffset)) {
      expect(off % 4).toBe(0);
    }
  });
});

describe('binding-signature field registry', () => {
  const EXPECTED_FIELDS = [
    'device',
    'bvhMode',
    'bvhNodesBuf',
    'bvhNodesOffset',
    'bvhNodesSize',
    'bvhIndicesBuf',
    'bvhIndicesOffset',
    'bvhIndicesSize',
    'bvhPositionsBuf',
    'bvhPositionsOffset',
    'bvhPositionsSize',
    'bvhNormalsBuf',
    'bvhNormalsOffset',
    'bvhNormalsSize',
    'cascadeBufs',
    'probeOriginWorld',
    'roomSize',
    'envTextureView',
    'envSampler',
    'materialTextureAtlasView',
    'materialMapMetaTextureView',
    'bvhTangentTextureView',
    'bvhVertexColorTextureView',
    'emittersBuf',
    'emittersOffset',
    'emittersSize',
    'emitterDataOffset',
    'emitterAliasOffset',
    'lightsBuf',
    'lightsOffset',
    'lightsSize',
  ];

  it('covers every directly bound field, in order', () => {
    expect([...bindingFieldRegistryFields]).toEqual(EXPECTED_FIELDS);
  });
});
