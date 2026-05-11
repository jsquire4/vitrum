import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { extractThreePbrScalars, PBR_DEFAULTS_DEFAULT } from '../material.js';

describe('extractThreePbrScalars — P2-6.1 cross-engine helper', () => {
  it('extracts all fields from a MeshPhysicalMaterial', () => {
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0.25, 0.5, 0.75),
      emissive: new THREE.Color(0.1, 0.2, 0.3),
      emissiveIntensity: 2,
      roughness: 0.4,
      metalness: 0.2,
      transmission: 0.6,
      ior: 1.7,
      attenuationColor: new THREE.Color(0.9, 0.85, 0.8),
      attenuationDistance: 5,
      thickness: 0.1,
    });
    const p = extractThreePbrScalars(mat);
    expect(p.baseColor).toEqual([0.25, 0.5, 0.75]);
    expect(p.emissive).toEqual([0.1, 0.2, 0.3]);
    expect(p.emissiveIntensity).toBe(2);
    expect(p.roughness).toBeCloseTo(0.4);
    expect(p.metallic).toBeCloseTo(0.2);
    expect(p.transmission).toBeCloseTo(0.6);
    expect(p.ior).toBeCloseTo(1.7);
    expect(p.attenuationColor).toEqual([0.9, 0.85, 0.8]);
    expect(p.attenuationDistance).toBe(5);
    expect(p.thickness).toBeCloseTo(0.1);
  });

  it('uses library defaults when fields are absent', () => {
    const mat = new THREE.MeshStandardMaterial();
    const p = extractThreePbrScalars(mat);
    // MeshStandardMaterial defaults: color = white, emissive = black,
    // emissiveIntensity = 1, roughness = 1, metalness = 0.
    // THREE's actual defaults trickle through; our defaults only kick in
    // when the field is undefined.
    expect(p.baseColor).toEqual([1, 1, 1]);
    expect(p.emissive).toEqual([0, 0, 0]);
    expect(p.emissiveIntensity).toBe(1);
    expect(p.roughness).toBe(1); // THREE.MeshStandardMaterial.roughness default is 1
    expect(p.metallic).toBe(0);
    // physical-only fields fall back to our defaults on a standard mat
    expect(p.transmission).toBe(0);
    expect(p.ior).toBeCloseTo(1.5);
    expect(p.attenuationColor).toEqual([1, 1, 1]);
    expect(p.attenuationDistance).toBe(Infinity);
    expect(p.thickness).toBe(0);
  });

  it('caller-supplied defaults override the library defaults', () => {
    const mat = new THREE.MeshStandardMaterial();
    const p = extractThreePbrScalars(mat, {
      baseColor: [0.5, 0.5, 0.5],
      roughness: 0.7,
      ior: 1.4,
    });
    // Caller's defaults only kick in when THREE has no value. THREE's
    // own constructor defaults (color=white, roughness=1) still win.
    expect(p.baseColor).toEqual([1, 1, 1]);
    expect(p.roughness).toBe(1);
    // ior is physical-only — standard mat has no value, so caller's
    // default applies.
    expect(p.ior).toBeCloseTo(1.4);
  });

  it('PBR_DEFAULTS_DEFAULT is exported and matches the internal defaults', () => {
    expect(PBR_DEFAULTS_DEFAULT.baseColor).toEqual([1, 1, 1]);
    expect(PBR_DEFAULTS_DEFAULT.emissive).toEqual([0, 0, 0]);
    expect(PBR_DEFAULTS_DEFAULT.roughness).toBe(0.5);
    expect(PBR_DEFAULTS_DEFAULT.metallic).toBe(0);
    expect(PBR_DEFAULTS_DEFAULT.transmission).toBe(0);
    expect(PBR_DEFAULTS_DEFAULT.ior).toBeCloseTo(1.5);
    expect(PBR_DEFAULTS_DEFAULT.attenuationColor).toEqual([1, 1, 1]);
    expect(PBR_DEFAULTS_DEFAULT.attenuationDistance).toBe(Infinity);
    expect(PBR_DEFAULTS_DEFAULT.thickness).toBe(0);
  });

  it('byte-equivalence with the pre-helper DDGI _uploadMaterials packer', () => {
    // Sanity: a glassy material packs to the same 16 floats the DDGI buffer
    // expects, in the same order, with `transmission > 0` setting the flag.
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0.9, 0.95, 1),
      roughness: 0.05,
      ior: 1.52,
      transmission: 1,
      attenuationColor: new THREE.Color(0.8, 0.85, 0.9),
    });
    const p = extractThreePbrScalars(mat);

    const expected = new Float32Array([
      0.9, 0.95, 1, 0,          // baseColor + pad
      0, 0, 0, 0.05,            // emissive + roughness (METAL is 0; ROUGH=0.05)
      0, 1.52, 1, 0,            // metallic, ior, transmission, pad
      0.8, 0.85, 0.9, 0,        // attenuationColor + flags placeholder
    ]);
    // flags slot is written via u32view — encode that here too.
    const buf = new Float32Array(16);
    buf[0] = p.baseColor[0];
    buf[1] = p.baseColor[1];
    buf[2] = p.baseColor[2];
    buf[4] = p.emissive[0];
    buf[5] = p.emissive[1];
    buf[6] = p.emissive[2];
    buf[7] = p.roughness;
    buf[8] = p.metallic;
    buf[9] = p.ior;
    buf[10] = p.transmission;
    buf[12] = p.attenuationColor[0];
    buf[13] = p.attenuationColor[1];
    buf[14] = p.attenuationColor[2];
    const u32 = new Uint32Array(buf.buffer);
    u32[15] = p.transmission > 0 ? 1 : 0;

    expect(Array.from(buf.subarray(0, 15))).toEqual(Array.from(expected.subarray(0, 15)));
    expect(u32[15]).toBe(1);
  });
});
