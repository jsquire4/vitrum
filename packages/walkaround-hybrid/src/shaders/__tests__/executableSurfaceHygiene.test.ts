import { describe, expect, it } from 'vitest';
import { DDGI_SH_WGSL } from '../../ddgi/wgsl/ddgiSH.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../materialAtlas.wgsl.js';
import { RESERVOIR_GI_WGSL } from '../reservoirGi.wgsl.js';
import { SCENE_STORAGE_ARENA_WGSL } from '../sceneStorageArena.wgsl.js';
import { SCENE_TRAVERSAL_WGSL } from '../sceneTraversal.wgsl.js';

describe('walkaround executable WGSL surface', () => {
  it('keeps only the live alpha and shadow traversal paths', () => {
    expect(SCENE_TRAVERSAL_WGSL).toContain('fn traceSceneFirstHit(');
    expect(SCENE_TRAVERSAL_WGSL).toContain('fn traceSceneAnyCastMask(');
    expect(SCENE_TRAVERSAL_WGSL).not.toContain('fn traceSceneFirstHitAlphaMask(');
    expect(SCENE_TRAVERSAL_WGSL).not.toContain('fn traceSceneAny(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn traceSceneFirstHitAlphaMaskTextured(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialShadowTransmittanceForHit(');
    expect(MATERIAL_ATLAS_WGSL).not.toContain('fn materialShadowOccluderForHit(');
  });

  it('keeps the metadata-aware GI update as the sole reservoir update entry', () => {
    expect(RESERVOIR_GI_WGSL).toContain('fn updateReservoirGIWithMetadata(');
    expect(RESERVOIR_GI_WGSL).not.toContain('fn updateReservoirGI(');
  });

  it('keeps producer SH math without the duplicate receiver sampler', () => {
    expect(DDGI_SH_WGSL).toContain('fn ddgiShBasis(');
    expect(DDGI_SH_WGSL).toContain('fn ddgiShCosineA(');
    expect(DDGI_SH_WGSL).not.toContain('fn ddgiSampleSHProbe(');
    expect(DDGI_SH_WGSL).not.toContain('fn ddgiShCoeffTexel(');
  });

  it('keeps arena loaders without the uncalled shader-side validation chain', () => {
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('fn sceneGeometryU32(');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('fn bvhLoadNode(');
    expect(SCENE_STORAGE_ARENA_WGSL).not.toContain('fn sceneStorageArenasValid(');
    expect(SCENE_STORAGE_ARENA_WGSL).not.toMatch(/fn scene(?:Geometry|Tlas|Lighting)ArenaValid\(/);
    expect(SCENE_STORAGE_ARENA_WGSL).not.toContain('SCENE_ARENA_MAGIC');
  });
});
