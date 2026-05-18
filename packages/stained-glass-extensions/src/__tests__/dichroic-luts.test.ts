import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  convertMaterial,
  vitrumMaterialToThree,
  vitrumSceneToThree,
  sceneFromThreeJS,
} from '@vitrum/three-bindings';
import {
  DICHROIC_LUTS_EXTENSION_ID,
  STAINED_GLASS_USER_DATA_KEYS,
  dichroicLUTsExtensionConverter,
} from '../dichroic-luts.js';

const CONVERTERS = [dichroicLUTsExtensionConverter];

describe('dichroicLUTsExtensionConverter — forward (THREE → vitrum)', () => {
  it('reads userData.vitrumDichroic*LUT into Material.extensions.dichroicLUTs', () => {
    const reflectanceLut = { __marker: 'reflectance' };
    const transmittanceLut = { __marker: 'transmittance' };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT] = reflectanceLut;
    m.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT] = transmittanceLut;
    const v = convertMaterial(m, { extensionConverters: CONVERTERS });
    const dichroic = v.extensions?.[DICHROIC_LUTS_EXTENSION_ID] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic).toBeDefined();
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBe(transmittanceLut);
  });

  it('produces no extensions field when neither LUT key is present', () => {
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    const v = convertMaterial(m, { extensionConverters: CONVERTERS });
    expect(v.extensions).toBeUndefined();
  });

  it('omits the missing slot when only one LUT key is present', () => {
    const reflectanceLut = { __marker: 'reflectance-only' };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT] = reflectanceLut;
    const v = convertMaterial(m, { extensionConverters: CONVERTERS });
    const dichroic = v.extensions?.[DICHROIC_LUTS_EXTENSION_ID] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBeUndefined();
  });

  it('does NOT populate dichroic when the converter is not registered (library default)', () => {
    // The whole point of D3: library defaults to host-agnostic behavior.
    const reflectanceLut = { __marker: 'should-not-show-up' };
    const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    m.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT] = reflectanceLut;
    const v = convertMaterial(m); // no options
    expect(v.extensions).toBeUndefined();
  });
});

describe('dichroicLUTsExtensionConverter — reverse (vitrum → THREE)', () => {
  it('stamps userData.vitrumDichroic*LUT from Material.extensions.dichroicLUTs', () => {
    const reflectanceLut = { __marker: 'reflectance' };
    const transmittanceLut = { __marker: 'transmittance' };
    const threeMat = vitrumMaterialToThree(
      {
        baseColor: [1, 1, 1], roughness: 0, metallic: 0,
        extensions: {
          [DICHROIC_LUTS_EXTENSION_ID]: {
            reflectance: reflectanceLut,
            transmittance: transmittanceLut,
          },
        },
      },
      undefined,
      { extensionConverters: CONVERTERS },
    );
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT]).toBe(reflectanceLut);
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT]).toBe(transmittanceLut);
  });

  it('does NOT stamp dichroic keys when extensions.dichroicLUTs is absent', () => {
    const threeMat = vitrumMaterialToThree(
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      undefined,
      { extensionConverters: CONVERTERS },
    );
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT]).toBeUndefined();
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT]).toBeUndefined();
  });

  it('does NOT stamp dichroic keys when the converter is not registered (library default)', () => {
    const threeMat = vitrumMaterialToThree(
      {
        baseColor: [1, 1, 1], roughness: 0, metallic: 0,
        extensions: {
          [DICHROIC_LUTS_EXTENSION_ID]: { reflectance: { __marker: 'r' } },
        },
      },
      undefined,
      // no converters
    );
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT]).toBeUndefined();
  });

  it('survives partial extension (only reflectance present)', () => {
    const reflectanceLut = { __marker: 'reflectance-only' };
    const threeMat = vitrumMaterialToThree(
      {
        baseColor: [1, 1, 1], roughness: 0, metallic: 0,
        extensions: {
          [DICHROIC_LUTS_EXTENSION_ID]: { reflectance: reflectanceLut },
        },
      },
      undefined,
      { extensionConverters: CONVERTERS },
    );
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT]).toBe(reflectanceLut);
    expect(threeMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT]).toBeUndefined();
  });
});

describe('dichroicLUTsExtensionConverter — full round trip', () => {
  it('preserves both LUTs through THREE → vitrum → THREE', () => {
    const reflectanceLut = { __marker: 'reflectance-rt' };
    const transmittanceLut = { __marker: 'transmittance-rt' };
    const original = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    original.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT] = reflectanceLut;
    original.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT] = transmittanceLut;

    const vitrumMat = convertMaterial(original, { extensionConverters: CONVERTERS });
    const dichroic = vitrumMat.extensions?.[DICHROIC_LUTS_EXTENSION_ID] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBe(transmittanceLut);

    const backToThree = vitrumMaterialToThree(vitrumMat, undefined, {
      extensionConverters: CONVERTERS,
    });
    expect(backToThree.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT]).toBe(reflectanceLut);
    expect(backToThree.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT]).toBe(transmittanceLut);
  });

  it('round-trips at scene level (sceneFromThreeJS / vitrumSceneToThree)', () => {
    const reflectanceLut = { __marker: 'reflectance-scene' };
    const transmittanceLut = { __marker: 'transmittance-scene' };
    const mat = new THREE.MeshPhysicalMaterial({ color: 0xffffff });
    mat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT] = reflectanceLut;
    mat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT] = transmittanceLut;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      3,
    ));
    geo.setAttribute('normal', new THREE.BufferAttribute(
      new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      3,
    ));
    const mesh = new THREE.Mesh(geo, mat);
    const scene = new THREE.Scene();
    scene.add(mesh);

    const vitrumScene = sceneFromThreeJS(scene, { extensionConverters: CONVERTERS });
    const firstPrim = vitrumScene.primitives[0];
    expect(firstPrim?.kind).toBe('mesh');
    if (firstPrim?.kind !== 'mesh') throw new Error('expected mesh primitive');
    const dichroic = firstPrim.material.extensions?.[DICHROIC_LUTS_EXTENSION_ID] as
      | { reflectance?: unknown; transmittance?: unknown }
      | undefined;
    expect(dichroic?.reflectance).toBe(reflectanceLut);
    expect(dichroic?.transmittance).toBe(transmittanceLut);

    const threeRoundTrip = vitrumSceneToThree(vitrumScene, {
      extensionConverters: CONVERTERS,
    });
    const meshChild = threeRoundTrip.children.find((c) => c instanceof THREE.Mesh) as
      | THREE.Mesh
      | undefined;
    expect(meshChild).toBeDefined();
    const rtMat = meshChild?.material as THREE.MeshPhysicalMaterial;
    expect(rtMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT]).toBe(reflectanceLut);
    expect(rtMat.userData[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT]).toBe(transmittanceLut);
  });
});
