/**
 * materialTextureRef.test.ts — P1 contract additions:
 *  - structured `TextureRef` with KHR_texture_transform / texCoord extraction
 *    (`toTextureRef`),
 *  - glTF alpha-mode (alphaTest → mask, transparent → blend, opacity),
 *  - AO + clearcoat/sheen/iridescence/anisotropy map slots,
 *  - the `toTextureRef` ↔ `fromTextureRef` round-trip via vitrumSceneToThree.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { convertMaterial, toTextureRef } from '../material.js';
import { vitrumSceneToThree } from '../vitrumSceneToThree.js';

describe('toTextureRef — KHR_texture_transform extraction', () => {
  it('extracts non-identity offset / repeat / rotation + channel', () => {
    const tex = new THREE.Texture();
    tex.offset.set(0.25, 0.5);
    tex.repeat.set(2, 3);
    tex.rotation = Math.PI / 4;
    (tex as { channel?: number }).channel = 1;
    const ref = toTextureRef(tex);
    expect(ref.handle).toBe(tex);
    expect(ref.texCoord).toBe(1);
    expect(ref.transform?.offset).toEqual([0.25, 0.5]);
    expect(ref.transform?.scale).toEqual([2, 3]);
    expect(ref.transform?.rotation).toBeCloseTo(Math.PI / 4);
  });

  it('omits identity transform + channel 0', () => {
    const ref = toTextureRef(new THREE.Texture());
    expect(ref.texCoord).toBeUndefined();
    expect(ref.transform).toBeUndefined();
  });
});

describe('convertMaterial — alpha mode + AO + lobe maps', () => {
  it('alphaTest → mask + alphaCutoff', () => {
    const m = new THREE.MeshStandardMaterial();
    m.alphaTest = 0.3;
    const spec = convertMaterial(m);
    expect(spec.alphaMode).toBe('mask');
    expect(spec.alphaCutoff).toBeCloseTo(0.3);
  });

  it('transparent → blend + opacity', () => {
    const m = new THREE.MeshStandardMaterial();
    m.transparent = true;
    m.opacity = 0.5;
    const spec = convertMaterial(m);
    expect(spec.alphaMode).toBe('blend');
    expect(spec.opacity).toBeCloseTo(0.5);
  });

  it('opaque default omits alphaMode', () => {
    expect(convertMaterial(new THREE.MeshStandardMaterial()).alphaMode).toBeUndefined();
  });

  it('captures AO map + intensity (structured TextureRef)', () => {
    const m = new THREE.MeshStandardMaterial();
    m.aoMap = new THREE.Texture();
    m.aoMapIntensity = 0.7;
    const spec = convertMaterial(m);
    expect(spec.aoMap?.handle).toBe(m.aoMap);
    expect(spec.aoMapIntensity).toBeCloseTo(0.7);
  });

  it('captures clearcoat / sheen / iridescence / anisotropy maps on physical material', () => {
    const m = new THREE.MeshPhysicalMaterial();
    m.clearcoat = 1; m.clearcoatMap = new THREE.Texture();
    m.sheen = 1; m.sheenColorMap = new THREE.Texture();
    m.iridescence = 1; m.iridescenceMap = new THREE.Texture();
    m.anisotropy = 1; m.anisotropyMap = new THREE.Texture();
    const spec = convertMaterial(m);
    expect(spec.clearcoatMap?.handle).toBe(m.clearcoatMap);
    expect(spec.sheenColorMap?.handle).toBe(m.sheenColorMap);
    expect(spec.iridescenceMap?.handle).toBe(m.iridescenceMap);
    expect(spec.anisotropyMap?.handle).toBe(m.anisotropyMap);
  });
});

describe('toTextureRef ↔ fromTextureRef round-trip', () => {
  it('preserves a UV-transformed baseColor map through THREE → vitrum → THREE', () => {
    const tex = new THREE.Texture();
    tex.offset.set(0.1, 0.2);
    tex.repeat.set(4, 4);
    const src = new THREE.MeshStandardMaterial();
    src.map = tex;
    const spec = convertMaterial(src);
    const scene = vitrumSceneToThree({
      primitives: [{
        kind: 'mesh',
        id: 't',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: spec,
      }],
      emitters: [],
      environment: { kind: 'none' },
    });
    const outMat = (scene.children[0] as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
    expect(outMat.map).toBe(tex);
    expect(outMat.map!.offset.x).toBeCloseTo(0.1);
    expect(outMat.map!.offset.y).toBeCloseTo(0.2);
    expect(outMat.map!.repeat.x).toBeCloseTo(4);
  });
});
