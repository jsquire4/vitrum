// untestedMaterialMaps.test.ts — closes the UNTESTED-promise class for the ten D3
// material-map fields on pt-webgl2 (audit item 25, trust-remediation-plan-2026-06-10.md).
//
// Findings audit context:
//   The promise inventory found ~22 UNTESTED promises. Ten of them are material-map
//   fields wired in the D3 wave: clearcoatMap, clearcoatRoughnessMap, clearcoatNormalMap,
//   sheenColorMap, sheenRoughnessMap, iridescenceMap, iridescenceThicknessMap,
//   specularColorMap, specularIntensityMap (pt-webgl2 packer), plus filteredGlossyFactor
//   upload and the texCoord finding (documented below).
//
// Per-test strategy:
//   (a) PACKER test — pack a MaterialSpec with that map + a layerOf assignment; assert the
//       EXACT texel/float offset carries the layer id. Offsets are READ from
//       materialsTexture.ts (the packer) and cross-verified against material_struct.glsl.js
//       (the GLSL decoder) — we assert the decoder's documented read site, not what the
//       packer currently emits blind.
//   (b) DECODER structural test — the composed GLSL contains the sampling site for that
//       field (e.g. 'material.clearcoatMap != - 1') confirming the consumption is wired.
//
// texCoord finding (closed):
//   pt-webgl2 packs one dense attribute-layer selector for every mapped-rich
//   slot. MAP_UV(mapIndex) decodes that selector and barycentrically interpolates
//   the exact scene-local layer, supporting arbitrary non-negative texCoord ids.

import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialsTexture, MATERIAL_PIXELS } from './materialsTexture.js';

// .glsl.js modules are typed by the wildcard `declare module '*.glsl.js' { default string }`
// in glsl-modules.d.ts (which cannot declare named exports). We pull named string
// exports via namespace imports cast to Record<string,string> — the same pattern
// used in src/glsl/bvh/index.ts.
import { MATERIAL_MAPPED_RICH_GLSL } from '../glsl/shader/structs/material_mapped_rich.glsl.js';
import * as GetSurfaceNS from '../glsl/render/get_surface_record_function.glsl.js';
import * as AttenuateHitNS from '../glsl/render/attenuate_hit_function.glsl.js';
import * as UtilNS from '../glsl/shader/common/util_functions.glsl.js';
import { GET_SURFACE_RECORD_MAPPED_PBR_GLSL } from '../glsl/render/get_surface_record_mapped_pbr.glsl.js';

const material_struct = MATERIAL_MAPPED_RICH_GLSL;
const get_surface_record_function: string = (GetSurfaceNS as Record<string, string>)['get_surface_record_function'] ?? '';
const attenuate_hit_function: string = (AttenuateHitNS as Record<string, string>)['attenuate_hit_function'] ?? '';
const util_functions: string = (UtilNS as Record<string, string>)['util_functions'] ?? '';
const compactSurface = get_surface_record_function.replace(/\s+/g, ' ');
const compactAttenuation = attenuate_hit_function.replace(/\s+/g, ' ');
const compactUtil = util_functions.replace(/\s+/g, ' ');
const compactMappedPbr = GET_SURFACE_RECORD_MAPPED_PBR_GLSL.replace(/\s+/g, ' ');

function decoderTexel(texel: number): string {
  const marker = `s = texelFetch1D( tex, i + ${texel}u );`;
  const start = material_struct.indexOf(marker);
  expect(start, `material decoder texel ${texel}`).toBeGreaterThanOrEqual(0);
  const next = material_struct.indexOf('\n  s = texelFetch1D(', start + marker.length);
  return material_struct.slice(start, next < 0 ? undefined : next);
}

// Helper: float offset of pixel `s`, channel `c` (0=r,1=g,2=b,3=a) for material 0.
function f(s: number, c: number): number {
  return s * 4 + c;
}

// Create a minimal MaterialSpec base + one specific map set to a handle we track.
function matWithMap(field: keyof MaterialSpec, handle: object): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0,
    [field]: { handle },
  };
}

// Pack the material with the given layerOf map and return the data array.
// MaterialsTextureData.data is typed Float32Array|Uint32Array (TexelGrid base), but
// 'rgba32f' materials always produce Float32Array — cast is safe here.
function pack(m: MaterialSpec, handle: object, layerId: number): Float32Array {
  const layerOf = new Map<unknown, number>([[handle, layerId]]);
  return packMaterialsTexture([m], layerOf).data as Float32Array;
}

describe('pt-webgl2 D3 material-map packer offsets — UNTESTED-promise closure (item 25)', () => {

  // clearcoatMap — packer sample 5 channel 0 (s5.r); GLSL: m.clearcoatMap = int(round(s5.r))
  it('clearcoatMap: layer id packed at sample 5 channel r (s5.r)', () => {
    const handle = {};
    const d = pack(matWithMap('clearcoatMap', handle), handle, 7);
    // s5.r = clearcoatMap layer id
    expect(d[f(5, 0)]).toBe(7);
    // decoder side: GLSL material_struct reads s5.r as clearcoatMap
    expect(decoderTexel(5)).toContain('m.clearcoatMap = int( round( s.r ) )');
  });

  // clearcoatRoughnessMap — packer sample 5 channel 2 (s5.b); GLSL: m.clearcoatRoughnessMap = int(round(s5.b))
  it('clearcoatRoughnessMap: layer id packed at sample 5 channel b (s5.b)', () => {
    const handle = {};
    const d = pack(matWithMap('clearcoatRoughnessMap', handle), handle, 3);
    expect(d[f(5, 2)]).toBe(3);
    expect(decoderTexel(5)).toContain('m.clearcoatRoughnessMap = int( round( s.b ) )');
  });

  // clearcoatNormalMap — packer sample 5 channel 3 (s5.a); GLSL: m.clearcoatNormalMap = int(round(s5.a))
  it('clearcoatNormalMap: layer id packed at sample 5 channel a (s5.a)', () => {
    const handle = {};
    const d = pack(matWithMap('clearcoatNormalMap', handle), handle, 5);
    expect(d[f(5, 3)]).toBe(5);
    expect(decoderTexel(5)).toContain('m.clearcoatNormalMap = int( round( s.a ) )');
  });

  // sheenColorMap — packer sample 7 channel 3 (s7.a); GLSL: m.sheenColorMap = int(round(s7.a))
  it('sheenColorMap: layer id packed at sample 7 channel a (s7.a)', () => {
    const handle = {};
    const d = pack(matWithMap('sheenColorMap', handle), handle, 2);
    expect(d[f(7, 3)]).toBe(2);
    expect(decoderTexel(7)).toContain('m.sheenColorMap = int( round( s.a ) )');
  });

  // sheenRoughnessMap — packer sample 8 channel 1 (s8.g); GLSL: m.sheenRoughnessMap = int(round(s8.g))
  it('sheenRoughnessMap: layer id packed at sample 8 channel g (s8.g)', () => {
    const handle = {};
    const d = pack(matWithMap('sheenRoughnessMap', handle), handle, 9);
    expect(d[f(8, 1)]).toBe(9);
    expect(decoderTexel(8)).toContain('m.sheenRoughnessMap = int( round( s.g ) )');
  });

  // iridescenceMap — packer sample 8 channel 2 (s8.b); GLSL: m.iridescenceMap = int(round(s8.b))
  it('iridescenceMap: layer id packed at sample 8 channel b (s8.b)', () => {
    const handle = {};
    const d = pack(matWithMap('iridescenceMap', handle), handle, 4);
    expect(d[f(8, 2)]).toBe(4);
    expect(decoderTexel(8)).toContain('m.iridescenceMap = int( round( s.b ) )');
  });

  // iridescenceThicknessMap — packer sample 8 channel 3 (s8.a); GLSL: m.iridescenceThicknessMap = int(round(s8.a))
  it('iridescenceThicknessMap: layer id packed at sample 8 channel a (s8.a)', () => {
    const handle = {};
    const d = pack(matWithMap('iridescenceThicknessMap', handle), handle, 6);
    expect(d[f(8, 3)]).toBe(6);
    expect(decoderTexel(8)).toContain('m.iridescenceThicknessMap = int( round( s.a ) )');
  });

  // specularColorMap — packer sample 10 channel 3 (s10.a); GLSL: m.specularColorMap = int(round(s10.a))
  it('specularColorMap: layer id packed at sample 10 channel a (s10.a)', () => {
    const handle = {};
    const d = pack(matWithMap('specularColorMap', handle), handle, 1);
    expect(d[f(10, 3)]).toBe(1);
    expect(decoderTexel(10)).toContain('m.specularColorMap = int( round( s.a ) )');
  });

  // specularIntensityMap — packer sample 11 channel 1 (s11.g); GLSL: m.specularIntensityMap = int(round(s11.g))
  it('specularIntensityMap: layer id packed at sample 11 channel g (s11.g)', () => {
    const handle = {};
    const d = pack(matWithMap('specularIntensityMap', handle), handle, 8);
    expect(d[f(11, 1)]).toBe(8);
    expect(decoderTexel(11)).toContain('m.specularIntensityMap = int( round( s.g ) )');
  });

  // Absent maps should be packed as -1 at those offsets.
  it('absent D3 maps default to -1 at their documented offsets', () => {
    const m: MaterialSpec = { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 };
    const d = packMaterialsTexture([m]).data as Float32Array;
    expect(d[f(5, 0)]).toBe(-1);  // clearcoatMap
    expect(d[f(5, 2)]).toBe(-1);  // clearcoatRoughnessMap
    expect(d[f(5, 3)]).toBe(-1);  // clearcoatNormalMap
    expect(d[f(7, 3)]).toBe(-1);  // sheenColorMap
    expect(d[f(8, 1)]).toBe(-1);  // sheenRoughnessMap
    expect(d[f(8, 2)]).toBe(-1);  // iridescenceMap
    expect(d[f(8, 3)]).toBe(-1);  // iridescenceThicknessMap
    expect(d[f(10, 3)]).toBe(-1); // specularColorMap
    expect(d[f(11, 1)]).toBe(-1); // specularIntensityMap
  });

  // Transform slots — D3 clearcoat/sheen/iridescence/specular maps have 2-texel
  // transform slots starting at texel 67 (clearcoatMap: texel 67/68, etc.).
  // We assert that a non-identity transform is written to the documented slot.
  it('clearcoatMap transform lands in texel 67/68 (firstTextureTransformIdx + 12)', () => {
    const handle = {};
    // Non-identity transform: scale (2,3), offset (0.1,0.2).
    const m: MaterialSpec = {
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      clearcoatMap: { handle, transform: { scale: [2, 3], offset: [0.1, 0.2], rotation: 0 } },
    };
    const layerOf = new Map<unknown, number>([[handle, 0]]);
    const d = packMaterialsTexture([m], layerOf).data as Float32Array;
    // Row 1 of mat3: (sx·cos, sx·sin, offsetX) at texel 67 = floats 67*4+0..+3
    const row1base = 67 * 4;
    expect(d[row1base + 0]).toBeCloseTo(2, 6); // sx·cos(0) = 2
    expect(d[row1base + 1]).toBeCloseTo(0, 6); // sx·sin(0) = 0
    expect(d[row1base + 2]).toBeCloseTo(0.1, 6); // offsetX
    // Row 2 of mat3: (-sy·sin, sy·cos, offsetY) at texel 68 = floats 68*4+0..+3
    const row2base = 68 * 4;
    expect(d[row2base + 0]).toBeCloseTo(-0, 6); // -sy·sin(0) = 0
    expect(d[row2base + 1]).toBeCloseTo(3, 6); // sy·cos(0) = 3
    expect(d[row2base + 2]).toBeCloseTo(0.2, 6); // offsetY
  });

  // DECODER structural: confirm the GLSL get_surface_record samples all nine D3 maps.
  it('GLSL decoder samples all nine D3 maps at their documented sites', () => {
    const sr = get_surface_record_function;
    // clearcoat group
    expect(sr).toContain('material.clearcoatMap != - 1');
    expect(sr).toContain('material.clearcoatRoughnessMap != - 1');
    expect(sr).toContain('material.clearcoatNormalMap != - 1');
    // sheen group
    expect(sr).toContain('material.sheenColorMap != - 1');
    expect(sr).toContain('material.sheenRoughnessMap != - 1');
    // iridescence group
    expect(sr).toContain('material.iridescenceMap != - 1');
    expect(sr).toContain('material.iridescenceThicknessMap != - 1');
    // specular group
    expect(sr).toContain('material.specularColorMap != - 1');
    expect(sr).toContain('material.specularIntensityMap != - 1');
  });
});

describe('pt-webgl2 filteredGlossyFactor upload — UNTESTED promise (item 25)', () => {
  // filteredGlossyFactor is read from FrameQualitySettings by the frame-uniforms
  // builder (index.ts line ~419: filterGlossyFactor: input.quality?.filteredGlossyFactor ?? 0)
  // and uploaded to the GPU via prog.setFloat('filterGlossyFactor', frame.filterGlossyFactor)
  // (glResources.ts line ~356). The upload-gap guard recording pattern lets us verify
  // the uniform is uploaded without a GPU.

  it('filterGlossyFactor uniform is uploaded with the value from quality.filteredGlossyFactor', async () => {
    const { createPTEngine_WebGL2 } = await import('../index.js');
    const { createMockGl } = await import('../__tests__/mockGl.js');
    const record = new Map<string, unknown>();
    const gl = createMockGl(record);
    const engine = await createPTEngine_WebGL2({ device: gl });
    const positions = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
    const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const uvs = new Float32Array(8);
    const indices = new Uint32Array([0, 2, 1, 2, 0, 3]);
    const material = { baseColor: [0.5, 0.5, 0.5] as [number, number, number], roughness: 1, metallic: 0 };
    engine.setScene({
      primitives: [{ kind: 'mesh', id: 't', positions, normals, uvs, indices, material }],
      emitters: [],
      environment: { kind: 'none' },
    } as never);
    const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
    const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
    engine.renderFrame({
      viewMatrix: view as never, projMatrix: proj as never,
      cameraPosition: [0, 0, 5] as never,
      viewport: { width: 32, height: 32, devicePixelRatio: 1 },
      frameIndex: 0, frameSeed: 0,
      quality: { samplesTarget: 1, filteredGlossyFactor: 0.75 },
    });
    // The uniform must have been set with the supplied value.
    expect(record.has('filterGlossyFactor')).toBe(true);
    expect(record.get('filterGlossyFactor')).toBeCloseTo(0.75, 6);
  });

  it('filterGlossyFactor defaults to 0 when absent from quality', async () => {
    const { createPTEngine_WebGL2 } = await import('../index.js');
    const { createMockGl } = await import('../__tests__/mockGl.js');
    const record = new Map<string, unknown>();
    const gl = createMockGl(record);
    const engine = await createPTEngine_WebGL2({ device: gl });
    const positions = new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
    const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const uvs = new Float32Array(8);
    const indices = new Uint32Array([0, 2, 1, 2, 0, 3]);
    const material = { baseColor: [0.5, 0.5, 0.5] as [number, number, number], roughness: 1, metallic: 0 };
    engine.setScene({
      primitives: [{ kind: 'mesh', id: 't', positions, normals, uvs, indices, material }],
      emitters: [],
      environment: { kind: 'none' },
    } as never);
    const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1]);
    const proj = new Float32Array([1.5, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.2, 0]);
    engine.renderFrame({
      viewMatrix: view as never, projMatrix: proj as never,
      cameraPosition: [0, 0, 5] as never,
      viewport: { width: 32, height: 32, devicePixelRatio: 1 },
      frameIndex: 0, frameSeed: 0,
      quality: { samplesTarget: 1 },
    });
    expect(record.has('filterGlossyFactor')).toBe(true);
    expect(record.get('filterGlossyFactor')).toBe(0);
  });
});

describe('pt-webgl2 texCoord — scalable UV-layer selection', () => {
  // TextureRef.texCoord is consumed through one dense attribute-layer selector
  // per map slot at texels 130..135. MAP_UV interpolates that layer directly.
  // Texel 86.a remains only as a backwards-compatible texCoord-1 mirror.

  it('pt-webgl2 MATERIAL_PIXELS stride includes alphaMapTransform, sampler policy, spectral, and layer-normal texels', () => {
    // The bitmask lives at texel 86.a; texels 93/94 carry alphaMapTransform;
    // texels 95/96 carry anisotropyMapTransform; texel 97 carries thickness;
    // texels 98/99 carry thicknessMapTransform; texels 100..120 carry per-map sampler policy;
    // texel 121 carries per-material spectral reflectance coefficients;
    // texels 122..129 carry front/back layer normal map payloads; 130..135
    // carry four dense UV attribute-layer selectors per texel.
    expect(MATERIAL_PIXELS).toBe(136);
  });

  it('packer retains the legacy UV1 mirror when a map has texCoord:1', () => {
    const handle = {};
    const layerOf = new Map<unknown, number>([[handle, 0]]);
    // baseColorMap at texCoord:1 → bit 0 set = 1.
    const m: MaterialSpec = {
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      baseColorMap: { handle, texCoord: 1 },
    };
    const d = packMaterialsTexture([m], layerOf).data as Float32Array;
    // texel 86.a = float offset 86*4+3 = 347
    expect(d[86 * 4 + 3]).toBe(1); // bit 0 only
  });

  it('packer writes 0 bitmask when all maps have texCoord:0 or absent', () => {
    const handle = {};
    const layerOf = new Map<unknown, number>([[handle, 0]]);
    const m: MaterialSpec = {
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      baseColorMap: { handle, texCoord: 0 }, // explicit uv0
    };
    const d = packMaterialsTexture([m], layerOf).data as Float32Array;
    expect(d[86 * 4 + 3]).toBe(0); // no bit set
  });

  it('legacy UV1 mirror correctly encodes multiple texCoord-1 maps', () => {
    const handle1 = {}; const handle2 = {};
    const layerOf = new Map<unknown, number>([[handle1, 0], [handle2, 1]]);
    // roughnessMap (bit 2) + emissiveMap (bit 4) both at texCoord:1
    const m: MaterialSpec = {
      baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
      roughnessMap: { handle: handle1, texCoord: 1 },
      emissiveMap: { handle: handle2, texCoord: 1 },
    };
    const d = packMaterialsTexture([m], layerOf).data as Float32Array;
    const mask = d[86 * 4 + 3]!;
    expect(mask & (1 << 2)).not.toBe(0); // roughnessMap bit
    expect(mask & (1 << 4)).not.toBe(0); // emissiveMap bit
    expect(mask & (1 << 0)).toBe(0);     // baseColorMap not set
  });

  it('GLSL get_surface_record dynamically interpolates the packed per-map layer', () => {
    const sr = get_surface_record_function;
    expect(sr).toContain('MAP_UV');
    expect(sr).toContain('readMaterialMapUvLayer( materials, materialIndex, mapIndex )');
    expect(sr).toContain('textureSampleBarycoord( attributesArray');
  });

  it('derives tangent frames from triangle positions and the exact selected UV layer', () => {
    expect(compactUtil).toContain(
      'mat3 getBasisFromSelectedUv( sampler2D positionAttr, sampler2DArray attributesArray, int uvLayer',
    );
    expect(compactUtil).toContain('vec3 p0 = texelFetch1D( positionAttr, faceIndices.x ).xyz;');
    expect(compactUtil).toContain(
      'vec2 uv0 = texelFetch1D( attributesArray, uvLayer, faceIndices.x ).xy;',
    );
    expect(compactUtil).toContain(
      'float determinant = delta1.x * delta2.y - delta1.y * delta2.x;',
    );
    expect(compactUtil).toContain(
      'tangent = ( edge1 * delta2.y - edge2 * delta1.y ) * inverseDeterminant;',
    );
    expect(compactUtil).toContain('if ( uvLayer == ATTR_UV && length( uv0TangentSample.xyz ) > 1e-6 )');
  });

  it('uses selected-UV tangent frames for mapped-rich normal, bump, layer, and clearcoat maps', () => {
    expect(compactSurface).toContain(
      'int activeNormalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 5u );',
    );
    expect(compactSurface).toContain(
      'bvh.position, attributesArray, activeNormalUvLayer, surfaceHit.faceIndices.xyz, normal, tangentSample',
    );
    expect(compactSurface).toContain(
      'int bumpUvLayer = readMaterialMapUvLayer( materials, materialIndex, 18u );',
    );
    expect(compactSurface).toContain(
      'int clearcoatNormalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 9u );',
    );
    expect(compactSurface).toContain(
      'activeNormalUvLayer = int( round( material.frontLayerNormalTexCoord ) );',
    );
    expect(compactAttenuation).toContain(
      'bvh.position, attributesArray, activeShadowNormalUvLayer, surfaceHit.faceIndices.xyz, faceN, tangentSample',
    );
  });

  it('uses selected-UV tangent frames in the mapped-PBR normal and bump tier', () => {
    expect(compactMappedPbr).toContain(
      'int normalUvLayer = readMaterialMapUvLayer( materials, materialIndex, 5u );',
    );
    expect(compactMappedPbr).toContain(
      'bvh.position, attributesArray, normalUvLayer, surfaceHit.faceIndices.xyz, normal, tangentSample',
    );
    expect(compactMappedPbr).toContain(
      'int bumpUvLayer = readMaterialMapUvLayer( materials, materialIndex, 18u );',
    );
    expect(compactMappedPbr).toContain(
      'bvh.position, attributesArray, bumpUvLayer, surfaceHit.faceIndices.xyz, normal, tangentSample',
    );
  });

  it('GLSL MAP_UV macro reads the baseColorMap selector at map index 0', () => {
    // Map index 0 addresses the baseColorMap layer selector.
    const sr = get_surface_record_function;
    expect(sr).toContain('MAP_UV( 0u )');
  });

  it('GLSL alphaMap sampling consumes its transform in surface and attenuation paths', () => {
    expect(material_struct).toContain('mat3 readMaterialMapTransform(');
    expect(material_struct).not.toContain('mat3 alphaMapTransform');
    expect(compactSurface).toContain('material.alphaMap, 93u, 106u, MAP_UV( 6u )');
    expect(compactAttenuation).toContain(
      'material.alphaMap, 93u, 106u, ATTENUATE_MAP_UV( 6u )',
    );
  });

  it('material_struct carries uvTexCoordMask decoded from texel 86.a', () => {
    const ms = material_struct;
    expect(ms).toContain('uvTexCoordMask');
    expect(decoderTexel(86)).toContain('m.uvTexCoordMask = uint( round( s.a ) )');
  });

  it('material_struct lazily decodes per-map policies and exposes the wrap-aware sample helper', () => {
    expect(material_struct).toContain('vec4 sampleMaterialTexture(');
    expect(material_struct).toContain('sampler2DArray tex, vec2 uv, int layer, vec4 policy');
    expect(material_struct).toContain('sampleMaterialTextureLinearLevel');
    expect(material_struct).toContain('vec4 readMaterialMapPolicy(');
    expect(material_struct).not.toContain('vec4 mapWrap;');
    expect(compactSurface).toContain('material.map, 55u, 100u, MAP_UV( 0u )');
    expect(compactSurface).toContain('material.metalnessMap, 57u, 101u, MAP_UV( 1u )');
    expect(get_surface_record_function).toContain('MAP_POLICY( 118u )');
  });

  it('GLSL material fetches use wrap-aware sampling instead of raw texture2D calls', () => {
    const sr = get_surface_record_function;
    expect(compactSurface).toContain('material.map, 55u, 100u, MAP_UV( 0u )');
    expect(compactSurface).toContain('material.alphaMap, 93u, 106u, MAP_UV( 6u )');
    expect(sr).toContain('vec4 bumpMapPolicy = MAP_POLICY( 118u );');
    expect(sr).toContain('material.specularIntensityMap,');
    expect(compactSurface).toContain(
      'material.specularIntensityMap, 83u, 115u, MAP_UV( 15u )',
    );
    expect(sr).not.toContain('texture2D( textures');
  });

  it('attenuation path uses dynamic UV-layer selection and wrap-aware material sampling', () => {
    const ah = attenuate_hit_function;
    expect(ah).toContain('readMaterialMapUvLayer( materials, materialIndex, mapIndex )');
    expect(ah).toContain('ATTENUATE_MAP_UV( 0u )');
    expect(ah).toContain('ATTENUATE_MAP_UV( 6u )');
    expect(compactAttenuation).toContain(
      'material.map, 55u, 100u, ATTENUATE_MAP_UV( 0u )',
    );
    expect(compactAttenuation).toContain(
      'material.alphaMap, 93u, 106u, ATTENUATE_MAP_UV( 6u )',
    );
    expect(compactAttenuation).toContain(
      'material.transmissionMap, 61u, 103u, ATTENUATE_MAP_UV( 3u )',
    );
    expect(ah).not.toContain('texture2D( textures');
  });
});
