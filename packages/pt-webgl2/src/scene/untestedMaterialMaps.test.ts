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
// texCoord finding (documented, not faked):
//   pt-webgl2 does NOT implement texCoord selection. The packer (materialsTexture.ts)
//   has no texCoord field in the material layout; the GLSL
//   get_surface_record_function.glsl.js reads a single `uv` from ATTR_UV throughout
//   and never branches on a texCoord selector. TextureRef.texCoord is accepted by the
//   contract (core/frame.ts:43) but is silently ignored in pt-webgl2. This is an
//   unkept promise; there is no per-map uv-set selector lane in the packer or shader.
//   Evidence: grep texCoord in packages/pt-webgl2/src/ returns zero hits.
//   Test below asserts the ABSENCE of texCoord handling in the decoder so a future
//   implementation must update this test when wiring it.

import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialsTexture, MATERIAL_PIXELS } from './materialsTexture.js';

// .glsl.js modules are typed by the wildcard `declare module '*.glsl.js' { default string }`
// in glsl-modules.d.ts (which cannot declare named exports). We pull named string
// exports via namespace imports cast to Record<string,string> — the same pattern
// used in src/glsl/bvh/index.ts.
import * as MaterialStructNS from '../glsl/shader/structs/material_struct.glsl.js';
import * as GetSurfaceNS from '../glsl/render/get_surface_record_function.glsl.js';

const material_struct: string = (MaterialStructNS as Record<string, string>)['material_struct'] ?? '';
const get_surface_record_function: string = (GetSurfaceNS as Record<string, string>)['get_surface_record_function'] ?? '';

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
  } as MaterialSpec;
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
    expect(material_struct).toContain('m.clearcoatMap = int( round( s5.r ) )');
  });

  // clearcoatRoughnessMap — packer sample 5 channel 2 (s5.b); GLSL: m.clearcoatRoughnessMap = int(round(s5.b))
  it('clearcoatRoughnessMap: layer id packed at sample 5 channel b (s5.b)', () => {
    const handle = {};
    const d = pack(matWithMap('clearcoatRoughnessMap', handle), handle, 3);
    expect(d[f(5, 2)]).toBe(3);
    expect(material_struct).toContain('m.clearcoatRoughnessMap = int( round( s5.b ) )');
  });

  // clearcoatNormalMap — packer sample 5 channel 3 (s5.a); GLSL: m.clearcoatNormalMap = int(round(s5.a))
  it('clearcoatNormalMap: layer id packed at sample 5 channel a (s5.a)', () => {
    const handle = {};
    const d = pack(matWithMap('clearcoatNormalMap', handle), handle, 5);
    expect(d[f(5, 3)]).toBe(5);
    expect(material_struct).toContain('m.clearcoatNormalMap = int( round( s5.a ) )');
  });

  // sheenColorMap — packer sample 7 channel 3 (s7.a); GLSL: m.sheenColorMap = int(round(s7.a))
  it('sheenColorMap: layer id packed at sample 7 channel a (s7.a)', () => {
    const handle = {};
    const d = pack(matWithMap('sheenColorMap', handle), handle, 2);
    expect(d[f(7, 3)]).toBe(2);
    expect(material_struct).toContain('m.sheenColorMap = int( round( s7.a ) )');
  });

  // sheenRoughnessMap — packer sample 8 channel 1 (s8.g); GLSL: m.sheenRoughnessMap = int(round(s8.g))
  it('sheenRoughnessMap: layer id packed at sample 8 channel g (s8.g)', () => {
    const handle = {};
    const d = pack(matWithMap('sheenRoughnessMap', handle), handle, 9);
    expect(d[f(8, 1)]).toBe(9);
    expect(material_struct).toContain('m.sheenRoughnessMap = int( round( s8.g ) )');
  });

  // iridescenceMap — packer sample 8 channel 2 (s8.b); GLSL: m.iridescenceMap = int(round(s8.b))
  it('iridescenceMap: layer id packed at sample 8 channel b (s8.b)', () => {
    const handle = {};
    const d = pack(matWithMap('iridescenceMap', handle), handle, 4);
    expect(d[f(8, 2)]).toBe(4);
    expect(material_struct).toContain('m.iridescenceMap = int( round( s8.b ) )');
  });

  // iridescenceThicknessMap — packer sample 8 channel 3 (s8.a); GLSL: m.iridescenceThicknessMap = int(round(s8.a))
  it('iridescenceThicknessMap: layer id packed at sample 8 channel a (s8.a)', () => {
    const handle = {};
    const d = pack(matWithMap('iridescenceThicknessMap', handle), handle, 6);
    expect(d[f(8, 3)]).toBe(6);
    expect(material_struct).toContain('m.iridescenceThicknessMap = int( round( s8.a ) )');
  });

  // specularColorMap — packer sample 10 channel 3 (s10.a); GLSL: m.specularColorMap = int(round(s10.a))
  it('specularColorMap: layer id packed at sample 10 channel a (s10.a)', () => {
    const handle = {};
    const d = pack(matWithMap('specularColorMap', handle), handle, 1);
    expect(d[f(10, 3)]).toBe(1);
    expect(material_struct).toContain('m.specularColorMap = int( round( s10.a ) )');
  });

  // specularIntensityMap — packer sample 11 channel 1 (s11.g); GLSL: m.specularIntensityMap = int(round(s11.g))
  it('specularIntensityMap: layer id packed at sample 11 channel g (s11.g)', () => {
    const handle = {};
    const d = pack(matWithMap('specularIntensityMap', handle), handle, 8);
    expect(d[f(11, 1)]).toBe(8);
    expect(material_struct).toContain('m.specularIntensityMap = int( round( s11.g ) )');
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
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
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
    const engine = await createPTEngine_WebGL2({ device: gl } as never);
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

describe('pt-webgl2 texCoord — unkept-promise documentation (item 25)', () => {
  // TextureRef.texCoord (core/frame.ts) is accepted by the API but pt-webgl2 has
  // no UV-set selector in its packer or GLSL. This test documents the absence of
  // texCoord handling so a future implementation is forced to update this test.

  it('pt-webgl2 MATERIAL_PIXELS packer layout has no texCoord lane (single UV, uv0 only)', () => {
    // The packer stores 93 pixels per material (MATERIAL_PIXELS = 93). The layout
    // has no channel reserved for a per-map texCoord selector — confirmed by reading
    // materialsTexture.ts (no texCoord field packed anywhere in the 93-px layout).
    // This assertion pins the total pixel count; if texCoord were added, the stride
    // would grow and this test would need updating.
    expect(MATERIAL_PIXELS).toBe(93);
  });

  it('GLSL get_surface_record has no texCoord branch (reads only ATTR_UV)', () => {
    // The GLSL explicitly reads a single UV via textureSampleBarycoord(..., ATTR_UV, ...).
    // If texCoord were wired, this would read 'ATTR_UV1' or branch on a texCoord selector.
    // This structural check documents the current uv0-only behaviour.
    const sr = get_surface_record_function;
    expect(sr).toContain('ATTR_UV');
    // No UV1 selector or texCoord branch:
    expect(sr).not.toContain('ATTR_UV1');
    expect(sr).not.toContain('texCoord');
  });

  // Promise status: TextureRef.texCoord is an UNKEPT promise on pt-webgl2.
  // The field is accepted by the @vitrum/core contract (frame.ts:43) but is
  // silently ignored. A multi-UV implementation would need to add an ATTR_UV1
  // attribute stream, a per-map texCoord selector lane in the material layout,
  // and a branch in get_surface_record_function.glsl.js.
});
