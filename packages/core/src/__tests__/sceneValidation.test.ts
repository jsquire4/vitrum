import { describe, expect, it } from 'vitest';
import {
  asMat4,
  getPrimitiveColorSet,
  getPrimitiveUvSet,
  validateMaterialSpec,
  validateScene,
  type MaterialSpec,
  type MeshPrimitive,
  type Scene,
  type SceneEmitter,
  type SceneEnvironment,
  type ScenePrimitive,
} from '../index.js';

const MATERIAL: MaterialSpec = {
  baseColor: [0.8, 0.7, 0.6],
  roughness: 0.4,
  metallic: 0.1,
};

function triangle(id = 'triangle'): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    indices: new Uint16Array([0, 1, 2]),
    material: MATERIAL,
  };
}

function sceneWith(primitives: ScenePrimitive[] = [triangle()]): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

describe('validateScene', () => {
  it('accepts a complete triangle scene and finite public material extensions', () => {
    expect(() => validateScene(sceneWith())).not.toThrow();
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      attenuationDistance: Infinity,
      dispersionAbbeNumber: 250,
      doubleSided: true,
      extensions: { host: { value: 1 } },
    })).not.toThrow();
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      doubleSided: 'true' as never,
    })).toThrow(/doubleSided.*boolean/);
  });

  it('retains valid empty geometry and zero-instance dynamic states', () => {
    const empty: ScenePrimitive = {
      kind: 'instanced-mesh',
      id: 'empty',
      positions: new Float32Array(),
      normals: new Float32Array(),
      indices: new Uint32Array(),
      material: MATERIAL,
      instances: [],
    };
    expect(() => validateScene(sceneWith([empty]))).not.toThrow();
  });

  it('rejects non-finite and inconsistent mesh streams before upload', () => {
    const nonFinite = triangle();
    nonFinite.positions[2] = Number.NaN;
    expect(() => validateScene(sceneWith([nonFinite]))).toThrow(/positions\[2\].*finite/);

    const badNormals = { ...triangle(), normals: new Float32Array(6) } as ScenePrimitive;
    expect(() => validateScene(sceneWith([badNormals]))).toThrow(/normals.*positions\.length/);

    const badIndex = { ...triangle(), indices: new Uint32Array([0, 1, 3]) } as ScenePrimitive;
    expect(() => validateScene(sceneWith([badIndex]))).toThrow(/indices\[2\].*vertex/);
  });

  it('rejects malformed texture metadata and unavailable authored UV lanes', () => {
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      baseColorMap: { handle: {}, texCoord: -1 },
    })).toThrow(/texCoord.*non-negative/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      normalMap: { handle: {}, wrapS: 'discard' as never },
    })).toThrow(/wrapS.*repeat/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      lightMap: { handle: {}, transform: { rotation: Number.NaN } },
    })).toThrow(/rotation.*finite/);

    const noUv = {
      ...triangle(),
      uvs: undefined,
      material: { ...MATERIAL, baseColorMap: { handle: {}, texCoord: 7 } },
    } as unknown as ScenePrimitive;
    expect(() => validateScene(sceneWith([noUv]))).toThrow(/TEXCOORD_7.*does not provide/);
  });

  it('validates and resolves arbitrary sparse UV sets while retaining legacy aliases', () => {
    const uv0 = new Float32Array([0, 0, 1, 0, 0, 1]);
    const uv2 = new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
    const primitive: MeshPrimitive = {
      ...triangle(),
      uvs: uv0,
      uvSets: [uv0, undefined, uv2],
      material: { ...MATERIAL, baseColorMap: { handle: {}, texCoord: 2 } },
    };
    expect(() => validateScene(sceneWith([primitive]))).not.toThrow();
    expect(getPrimitiveUvSet(primitive, 0)).toBe(uv0);
    expect(getPrimitiveUvSet(primitive, 1)).toBeUndefined();
    expect(getPrimitiveUvSet(primitive, 2)).toBe(uv2);
    expect(getPrimitiveUvSet(primitive, -1)).toBeUndefined();

    const malformed = {
      ...primitive,
      uvSets: [new Float32Array([0, 0, 0, 0, 0, 0]), undefined, uv2],
    } as MeshPrimitive;
    expect(() => validateScene(sceneWith([malformed]))).toThrow(/uvSets\[0\].*legacy uvs alias/);

    const short = { ...primitive, uvSets: [uv0, undefined, new Float32Array(2)] } as MeshPrimitive;
    expect(() => validateScene(sceneWith([short]))).toThrow(/uvSets\[2\].*length/);
  });

  it('validates and resolves arbitrary sparse vertex-color sets', () => {
    const color0 = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const color3 = new Float32Array([
      0.1, 0.2, 0.3, 1,
      0.4, 0.5, 0.6, 0.8,
      0.7, 0.8, 0.9, 0.6,
    ]);
    const primitive: MeshPrimitive = {
      ...triangle(),
      colors: color0,
      colorSets: [color0, undefined, undefined, color3],
    };
    expect(() => validateScene(sceneWith([primitive]))).not.toThrow();
    expect(getPrimitiveColorSet(primitive, 0)).toBe(color0);
    expect(getPrimitiveColorSet(primitive, 1)).toBeUndefined();
    expect(getPrimitiveColorSet(primitive, 3)).toBe(color3);
    expect(getPrimitiveColorSet(primitive, -1)).toBeUndefined();

    const mismatchedAlias = {
      ...primitive,
      colorSets: [new Float32Array(color0.length), undefined, undefined, color3],
    } as MeshPrimitive;
    expect(() => validateScene(sceneWith([mismatchedAlias]))).toThrow(
      /colorSets\[0\].*legacy colors alias/,
    );

    const malformed = {
      ...primitive,
      colorSets: [color0, undefined, undefined, new Float32Array(5)],
    } as MeshPrimitive;
    expect(() => validateScene(sceneWith([malformed]))).toThrow(/colorSets\[3\].*length/);
  });

  it('rejects non-affine and singular transforms', () => {
    const projective = asMat4([1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(() => validateScene(sceneWith([{ ...triangle(), transform: projective }]))).toThrow(/affine/);

    const singular = asMat4([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(() => validateScene(sceneWith([{ ...triangle(), transform: singular }]))).toThrow(/invertible/);
  });

  it('rejects duplicate scene ids and dangling or analytic mesh emitters', () => {
    expect(() => validateScene(sceneWith([triangle('same'), triangle('same')]))).toThrow(/duplicates/);

    const missing: Scene = {
      ...sceneWith(),
      emitters: [{ kind: 'mesh-area', id: 'light', meshId: 'missing', color: [1, 1, 1], intensity: 1 }],
    };
    expect(() => validateScene(missing)).toThrow(/references missing primitive/);

    const analytic: ScenePrimitive = {
      kind: 'analytic',
      id: 'sphere',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: MATERIAL,
    };
    expect(() => validateScene({
      primitives: [analytic],
      emitters: [{ kind: 'mesh-area', id: 'light', meshId: 'sphere', color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    })).toThrow(/mesh-like primitive/);
  });

  it('validates emitter and environment physical domains', () => {
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{ kind: 'directional', id: 'sun', direction: [0, 2, 0], color: [1, 1, 1], intensity: 1 }],
    })).toThrow(/unit length/);

    expect(() => validateScene({
      ...sceneWith(),
      environment: {
        kind: 'procedural-sky',
        sunDirection: [0, 1, 0],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.01,
        mieDirectionalG: 1,
      },
    })).toThrow(/strictly between -1 and 1/);

    expect(() => validateScene({
      ...sceneWith(),
      emitters: [
        {
          kind: 'disc-area', id: 'tiny-disc', position: [0, 0, 0], normal: [0, 1, 0],
          radius: 1e-30, color: [1, 1, 1], intensity: 1,
        },
        {
          kind: 'rect-area', id: 'tiny-rect', position: [0, 0, 0],
          uAxis: [1e-20, 0, 0], vAxis: [0, 1e-20, 0],
          color: [1, 1, 1], intensity: 1,
        },
      ],
    })).not.toThrow();

    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'disc-area', id: 'underflow-radius', position: [0, 0, 0],
        normal: [0, 1, 0], radius: 1e-50, color: [1, 1, 1], intensity: 1,
      }],
    })).toThrow(/radius.*(?:representable|underflow).*float32/);

    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'rect-area', id: 'underflow-area', position: [0, 0, 0],
        uAxis: [1e-30, 0, 0], vAxis: [0, 1e-30, 0],
        color: [1, 1, 1], intensity: 1,
      }],
    })).toThrow(/uAxis×vAxis.*representable as float32/);
  });

  it('validates every skinned and inactive morph stream', () => {
    const skinWeights = new Float32Array(12);
    const skinIndices = new Uint32Array(12);
    for (let vertex = 0; vertex < 3; vertex += 1) skinWeights[vertex * 4] = 1;
    const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const skinned: ScenePrimitive = {
      ...triangle('skin'),
      kind: 'skinned-mesh',
      skinIndices,
      skinWeights,
      bones: identity,
      boneInverses: identity.slice(),
      morphTargets: [new Float32Array(9)],
      morphWeights: new Float32Array([0]),
    };
    expect(() => validateScene(sceneWith([skinned]))).not.toThrow();

    const { morphWeights: omittedMorphWeights, ...zeroWeightByOmission } = skinned;
    void omittedMorphWeights;
    expect(() => validateScene(sceneWith([zeroWeightByOmission]))).not.toThrow();

    const wrongWeightCount = {
      ...skinned,
      morphWeights: new Float32Array(0),
    } as ScenePrimitive;
    expect(() => validateScene(sceneWith([wrongWeightCount]))).toThrow(/morphWeights.*length/);

    const nonFiniteWeight = {
      ...skinned,
      morphWeights: new Float32Array([Number.NaN]),
    } as ScenePrimitive;
    expect(() => validateScene(sceneWith([nonFiniteWeight]))).toThrow(/morphWeights.*finite/);

    const malformedInactive = {
      ...skinned,
      morphTargets: [new Float32Array(6)],
      morphWeights: new Float32Array([0]),
    } as ScenePrimitive;
    expect(() => validateScene(sceneWith([malformedInactive]))).toThrow(/morphTargets\[0\].*length/);

    const badSum = { ...skinned, skinWeights: new Float32Array(12) } as ScenePrimitive;
    expect(() => validateScene(sceneWith([badSum]))).toThrow(/must sum to 1/);

    const partialBind = { ...skinned, bindMatrix: identity } as ScenePrimitive;
    expect(() => validateScene(sceneWith([partialBind]))).toThrow(/supplied together/);

    const singularBone = identity.slice();
    singularBone[0] = 0;
    expect(() => validateScene(sceneWith([{
      ...skinned,
      bones: singularBone,
    }]))).toThrow(/bones\[0\].*invertible/);

    const nonAffineInverse = identity.slice();
    nonAffineInverse[3] = 1;
    expect(() => validateScene(sceneWith([{
      ...skinned,
      boneInverses: nonAffineInverse,
    }]))).toThrow(/boneInverses\[0\].*affine/);

    const bindMatrix = asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]);
    const bindMatrixInverse = asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -3, 0, 0, 1]);
    expect(() => validateScene(sceneWith([{
      ...skinned,
      bindMatrix,
      bindMatrixInverse,
    }]))).not.toThrow();
    expect(() => validateScene(sceneWith([{
      ...skinned,
      bindMatrix,
      bindMatrixInverse: asMat4(identity.slice()),
    }]))).toThrow(/bindMatrix\/bindMatrixInverse.*reciprocal/);

    for (const [field, value, baseField] of [
      ['morphTargetTangents', [new Float32Array(9)], 'tangents'],
      ['morphTargetUvs', [new Float32Array(6)], 'uvs'],
      ['morphTargetUv1s', [new Float32Array(6)], 'uv1'],
    ] as const) {
      const missingBase = {
        ...skinned,
        ...(field === 'morphTargetUvs' ? { uvs: undefined } : {}),
        [field]: value,
      } as unknown as ScenePrimitive;
      expect(() => validateScene(sceneWith([missingBase]))).toThrow(
        new RegExp(`${field}.*base ${baseField} stream`),
      );
    }
  });

  it('validates scalable morph UV lanes and their legacy aliases', () => {
    const skinWeights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const uv2 = new Float32Array(6);
    const delta2 = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const skinned = {
      ...triangle('morph-uv-sets'),
      kind: 'skinned-mesh' as const,
      uvSets: [triangle().uvs, undefined, uv2],
      skinIndices: new Uint32Array(12),
      skinWeights,
      bones: identity,
      boneInverses: identity.slice(),
      morphTargets: [new Float32Array(9)],
      morphTargetUvSets: [undefined, undefined, [delta2]],
      morphWeights: new Float32Array([1]),
    };
    expect(() => validateScene(sceneWith([skinned]))).not.toThrow();

    const missingBase = {
      ...skinned,
      uvSets: [skinned.uvs],
    } as ScenePrimitive;
    expect(() => validateScene(sceneWith([missingBase]))).toThrow(/morphTargetUvSets\[2\].*matching/);
  });

  it('routes analytic shape-domain checks through the canonical validator', () => {
    const invalid: ScenePrimitive = {
      kind: 'analytic',
      id: 'bad-sphere',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 0]),
      material: MATERIAL,
    };
    expect(() => validateScene(sceneWith([invalid]))).toThrow(/radius.*> 0/);
  });

  it('rejects unknown fields on every primitive, emitter, and environment variant', () => {
    const identity = asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const skinWeights = new Float32Array(12);
    for (let vertex = 0; vertex < 3; vertex += 1) skinWeights[vertex * 4] = 1;
    const primitives: ScenePrimitive[] = [
      triangle('mesh'),
      { ...triangle('instances'), kind: 'instanced-mesh', instances: [] },
      {
        ...triangle('skin'),
        kind: 'skinned-mesh',
        skinIndices: new Uint32Array(12),
        skinWeights,
        bones: identity,
        boneInverses: identity.slice(),
      },
      {
        kind: 'analytic',
        id: 'sphere',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: MATERIAL,
      },
    ];
    for (const primitive of primitives) {
      expect(() => validateScene(sceneWith([
        { ...primitive, unknownPrimitiveField: true } as never,
      ]))).toThrow(/unknownPrimitiveField.*known contract field/);
    }

    const emitters: SceneEmitter[] = [
      { kind: 'directional', id: 'e', direction: [0, 1, 0], color: [1, 1, 1], intensity: 1 },
      { kind: 'disc-area', id: 'e', position: [0, 0, 0], normal: [0, 1, 0], radius: 1, color: [1, 1, 1], intensity: 1 },
      { kind: 'rect-area', id: 'e', position: [0, 0, 0], uAxis: [1, 0, 0], vAxis: [0, 0, 1], color: [1, 1, 1], intensity: 1 },
      { kind: 'point', id: 'e', position: [0, 0, 0], color: [1, 1, 1], intensity: 1 },
      { kind: 'spot', id: 'e', position: [0, 0, 0], direction: [0, 1, 0], angle: 0.5, color: [1, 1, 1], intensity: 1 },
      { kind: 'mesh-area', id: 'e', meshId: 'triangle', color: [1, 1, 1], intensity: 1 },
    ];
    for (const emitter of emitters) {
      expect(() => validateScene({
        ...sceneWith(),
        emitters: [{ ...emitter, unknownEmitterField: true } as never],
      })).toThrow(/unknownEmitterField.*known contract field/);
    }
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'cross-kind', position: [0, 0, 0], direction: [0, 1, 0],
        color: [1, 1, 1], intensity: 1,
      } as never],
    })).toThrow(/direction.*known contract field/);

    const environments: SceneEnvironment[] = [
      { kind: 'none' },
      { kind: 'hdri', hdri: { opaque: { fields: 'are not inspected' } } },
      {
        kind: 'procedural-sky',
        sunDirection: [0, 1, 0],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.01,
        mieDirectionalG: 0.8,
      },
    ];
    for (const environment of environments) {
      expect(() => validateScene({
        ...sceneWith(),
        environment: { ...environment, unknownEnvironmentField: true } as never,
      })).toThrow(/unknownEnvironmentField.*known contract field/);
    }
    expect(() => validateScene({ ...sceneWith(), unknownSceneField: true } as never)).toThrow(
      /unknownSceneField.*known contract field/,
    );
  });

  it('rejects unknown nested contract fields while keeping extensions and handles opaque', () => {
    expect(() => validateMaterialSpec({ ...MATERIAL, unknownMaterialField: true } as never)).toThrow(
      /unknownMaterialField.*known contract field/,
    );
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      baseColorMap: { handle: { arbitrary: { nested: true } }, unknownTextureField: true } as never,
    })).toThrow(/unknownTextureField.*known contract field/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      baseColorMap: { handle: {}, transform: { unknownTransformField: true } as never },
    })).toThrow(/unknownTransformField.*known contract field/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      frontLayer: { transmission: [1, 1, 1], unknownLayerField: true } as never,
    })).toThrow(/unknownLayerField.*known contract field/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      thinFilmStack: {
        layers: [{ ior: 1.5, thicknessNm: 200 }],
        unknownStackField: true,
      } as never,
    })).toThrow(/unknownStackField.*known contract field/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      thinFilmStack: {
        layers: [{ ior: 1.5, thicknessNm: 200, unknownThinFilmField: true } as never],
      },
    })).toThrow(/unknownThinFilmField.*known contract field/);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0, 0, 0]),
        unknownSpectralField: true,
      } as never,
    })).toThrow(/unknownSpectralField.*known contract field/);
    expect(() => validateScene(sceneWith([{
      kind: 'analytic',
      id: 'fallback',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: MATERIAL,
      fallbackMesh: {
        positions: triangle().positions,
        normals: triangle().normals,
        indices: triangle().indices,
        unknownFallbackField: true,
      } as never,
    }]))).toThrow(/unknownFallbackField.*known contract field/);

    expect(() => validateMaterialSpec({
      ...MATERIAL,
      extensions: {
        anyBackendKey: { anyNestedShape: [1, 2, 3] },
      },
      baseColorMap: { handle: { anyOpaquePayload: { accepted: true } } },
    })).not.toThrow();
  });

  it('rejects tuple lookalikes, tuple properties, and typed-array tag spoofing', () => {
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point',
        id: 'array-like',
        position: { 0: 0, 1: 0, 2: 0, length: 3 } as never,
        color: [1, 1, 1],
        intensity: 1,
      }],
    })).toThrow(/position.*must be an array/);

    const colorWithExtra = Object.assign([1, 1, 1], { silentlyIgnored: true });
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'extra-array-field', position: [0, 0, 0],
        color: colorWithExtra as never, intensity: 1,
      }],
    })).toThrow(/silentlyIgnored.*numeric indices/);

    const colorWithSymbol = [1, 1, 1];
    Object.defineProperty(colorWithSymbol, Symbol('tuple-field'), {
      value: true,
      enumerable: true,
    });
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'symbol-array-field', position: [0, 0, 0],
        color: colorWithSymbol as never, intensity: 1,
      }],
    })).toThrow(/Symbol\(tuple-field\).*numeric indices/);

    const colorWithHiddenField = [1, 1, 1];
    Object.defineProperty(colorWithHiddenField, 'hiddenTupleField', {
      value: true,
      enumerable: false,
    });
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'hidden-array-field', position: [0, 0, 0],
        color: colorWithHiddenField as never, intensity: 1,
      }],
    })).toThrow(/hiddenTupleField.*numeric indices/);

    const colorWithHiddenSymbol = [1, 1, 1];
    Object.defineProperty(colorWithHiddenSymbol, Symbol('hidden-tuple-symbol'), {
      value: true,
      enumerable: false,
    });
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'hidden-symbol-array-field', position: [0, 0, 0],
        color: colorWithHiddenSymbol as never, intensity: 1,
      }],
    })).toThrow(/Symbol\(hidden-tuple-symbol\).*numeric indices/);

    const float32Spoof = {
      0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0,
      length: 9,
      [Symbol.toStringTag]: 'Float32Array',
    } as unknown as Float32Array;
    expect(() => validateScene(sceneWith([{
      ...triangle('spoofed-positions'),
      positions: float32Spoof,
    }]))).toThrow(/positions.*must be a Float32Array/);

    const mat4Spoof = {
      length: 16,
      [Symbol.toStringTag]: 'Float32Array',
    } as unknown as ReturnType<typeof asMat4>;
    expect(() => validateScene(sceneWith([{
      ...triangle('spoofed-transform'),
      transform: mat4Spoof,
    }]))).toThrow(/transform.*16-element Float32Array/);

    const dataViewSpoof = new DataView(new ArrayBuffer(36));
    Object.defineProperty(dataViewSpoof, Symbol.toStringTag, { value: 'Float32Array' });
    expect(() => validateScene(sceneWith([{
      ...triangle('dataview-spoofed-positions'),
      positions: dataViewSpoof as never,
    }]))).toThrow(/positions.*must be a Float32Array/);

    for (const advertisedBrand of ['Uint16Array', 'Uint32Array']) {
      const wrongIndexView = new Float32Array([0, 1, 2]);
      Object.defineProperty(wrongIndexView, Symbol.toStringTag, { value: advertisedBrand });
      expect(() => validateScene(sceneWith([{
        ...triangle(`spoofed-${advertisedBrand}`),
        indices: wrongIndexView as never,
      }]))).toThrow(/indices.*Uint16Array or Uint32Array/);
    }

    const inheritedTuple = [1, 1, 1];
    Reflect.deleteProperty(inheritedTuple, '1');
    const poisonedPrototype = Object.create(Array.prototype) as number[];
    Object.defineProperty(poisonedPrototype, '1', {
      get: () => 1,
      enumerable: true,
    });
    Object.setPrototypeOf(inheritedTuple, poisonedPrototype);
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'prototype-tuple', position: [0, 0, 0],
        color: inheritedTuple as never, intensity: 1,
      }],
    })).toThrow(/color.*dense.*own data property/);

    for (const enumerable of [true, false]) {
      const descriptorTuple = [1, 1, 1];
      Object.defineProperty(descriptorTuple, '1', {
        get: () => 1,
        enumerable,
      });
      expect(() => validateScene({
        ...sceneWith(),
        emitters: [{
          kind: 'point', id: `descriptor-tuple-${enumerable}`, position: [0, 0, 0],
          color: descriptorTuple as never, intensity: 1,
        }],
      })).toThrow(/color\[1\].*own enumerable data properties/);
    }

    const ownLengthPositions = triangle('intrinsic-float32-length').positions;
    Object.defineProperty(ownLengthPositions, 'length', { value: 0 });
    const ownLengthIndices = new Uint32Array([0, 1, 2]);
    Object.defineProperty(ownLengthIndices, 'length', { value: 0 });
    expect(() => validateScene(sceneWith([{
      ...triangle('intrinsic-typed-lengths'),
      positions: ownLengthPositions,
      indices: ownLengthIndices,
    }]))).not.toThrow();
  });

  it('accepts tiny invertible affine matrices and rejects projective or non-representable inverses', () => {
    const tinyAffine = asMat4([
      1e-6, 0, 0, 0,
      0, 1e-6, 0, 0,
      0, 0, 1e-6, 0,
      0, 0, 0, 1,
    ]);
    expect(() => validateScene(sceneWith([{
      ...triangle('tiny-affine'),
      transform: tinyAffine,
    }]))).not.toThrow();

    const anisotropic = asMat4([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1e-13, 0,
      0, 0, 0, 1,
    ]);
    expect(() => validateScene(sceneWith([{
      ...triangle('anisotropic-affine'),
      transform: anisotropic,
    }]))).not.toThrow();

    const projective = tinyAffine.slice();
    projective[3] = 1e-7;
    expect(() => validateScene(sceneWith([{
      ...triangle('projective'),
      transform: projective as never,
    }]))).toThrow(/affine.*bottom row/);

    const inverseOverflow = tinyAffine.slice();
    inverseOverflow[0] = 1e-40;
    inverseOverflow[5] = 1e-40;
    inverseOverflow[10] = 1e-40;
    expect(() => validateScene(sceneWith([{
      ...triangle('inverse-overflow'),
      transform: inverseOverflow as never,
    }]))).toThrow(/inverse representable as float32/);
  });

  it('requires every material texture texCoord on its primitive or fallback mesh', () => {
    expect(() => validateScene(sceneWith([{
      ...triangle('missing-uv2'),
      material: {
        ...MATERIAL,
        baseColorMap: { handle: {}, texCoord: 2 },
      },
    }]))).toThrow(/baseColorMap\.texCoord.*TEXCOORD_2.*does not provide/);

    expect(() => validateScene(sceneWith([{
      ...triangle('missing-layer-uv3'),
      material: {
        ...MATERIAL,
        frontLayer: {
          transmission: [1, 1, 1],
          normalMap: { handle: {}, texCoord: 3 },
        },
      },
    }]))).toThrow(/frontLayer\.normalMap\.texCoord.*TEXCOORD_3/);

    const tri = triangle('arbitrary-texture-uvs');
    const uv2 = new Float32Array(6);
    const uv3 = new Float32Array(6);
    expect(() => validateScene(sceneWith([{
      ...tri,
      uvSets: [tri.uvs, undefined, uv2, uv3],
      material: {
        ...MATERIAL,
        baseColorMap: { handle: {}, texCoord: 2 },
        frontLayer: {
          transmission: [1, 1, 1],
          normalMap: { handle: {}, texCoord: 3 },
        },
      },
    }]))).not.toThrow();

    expect(() => validateScene(sceneWith([{
      kind: 'analytic',
      id: 'textured-analytic',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: { ...MATERIAL, normalMap: { handle: {}, texCoord: 2 } },
      fallbackMesh: {
        positions: triangle().positions,
        normals: triangle().normals,
        uvs: triangle().uvs!,
      },
    }]))).toThrow(/normalMap\.texCoord.*fallbackMesh.*does not provide/);

    expect(() => validateScene(sceneWith([{
      kind: 'analytic',
      id: 'native-textured-analytic',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: { ...MATERIAL, normalMap: { handle: {}, texCoord: 2 } },
    }]))).not.toThrow();
  });

  it('rejects finite float64 values that overflow float32 while preserving attenuation Infinity', () => {
    expect(() => validateMaterialSpec({ ...MATERIAL, normalScale: 1e300 })).toThrow(
      /normalScale.*representable as float32/,
    );
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'overflow', position: [1e300, 0, 0],
        color: [1, 1, 1], intensity: 1,
      }],
    })).toThrow(/position\[0\].*representable as float32/);
    expect(() => validateMaterialSpec({ ...MATERIAL, opacity: 1e-50 })).toThrow(
      /opacity.*underflow to zero as float32/,
    );
    expect(() => validateMaterialSpec({ ...MATERIAL, ior: 1e-50 })).toThrow(
      /ior.*underflow to zero as float32/,
    );
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [{
        kind: 'point', id: 'underflow-intensity', position: [0, 0, 0],
        color: [1, 1, 1], intensity: 1e-50,
      }],
    })).toThrow(/intensity.*underflow to zero as float32/);
    expect(() => validateMaterialSpec({ ...MATERIAL, attenuationDistance: Infinity })).not.toThrow();
  });

  it('reserves only non-enumerable symbols for metadata', () => {
    const enumerableSymbol = { ...MATERIAL };
    Object.defineProperty(enumerableSymbol, Symbol('user-field'), {
      value: true,
      enumerable: true,
    });
    expect(() => validateMaterialSpec(enumerableSymbol)).toThrow(
      /Symbol\(user-field\).*enumerable symbol fields are not allowed/,
    );

    const nonEnumerableString = { ...MATERIAL };
    Object.defineProperty(nonEnumerableString, 'hiddenUserField', {
      value: true,
      enumerable: false,
    });
    expect(() => validateMaterialSpec(nonEnumerableString)).toThrow(
      /hiddenUserField.*contract fields must be enumerable/,
    );

    const provenance = { ...MATERIAL };
    Object.defineProperty(provenance, Symbol('vitrum.gltf.provenance'), {
      value: { source: 3 },
      enumerable: false,
    });
    expect(() => validateMaterialSpec(provenance)).not.toThrow();
  });

  it('validates strict dense and sparse array-container shapes without prototype lookup', () => {
    const primitivesWithExtra = Object.assign([triangle()], { ignoredTail: true });
    expect(() => validateScene(sceneWith(primitivesWithExtra))).toThrow(
      /scene\.primitives.*ignoredTail.*numeric indices/,
    );

    const emittersWithSymbol: SceneEmitter[] = [];
    Object.defineProperty(emittersWithSymbol, Symbol('emitter-list'), {
      value: true,
      enumerable: false,
    });
    expect(() => validateScene({
      ...sceneWith(),
      emitters: emittersWithSymbol,
    })).toThrow(/scene\.emitters.*Symbol\(emitter-list\).*numeric indices/);

    const uv0 = triangle().uvs!;
    const uv2 = new Float32Array(uv0);
    const sparseUvSets = new Array<Float32Array | undefined>(3);
    sparseUvSets[0] = uv0;
    sparseUvSets[2] = uv2;
    expect(() => validateScene(sceneWith([{
      ...triangle('sparse-own-uv-sets'),
      uvs: uv0,
      uvSets: sparseUvSets,
      material: { ...MATERIAL, baseColorMap: { handle: {}, texCoord: 2 } },
    }]))).not.toThrow();

    const accessorColorSets = [new Float32Array(9)];
    Object.defineProperty(accessorColorSets, '0', {
      get: () => new Float32Array(9),
      enumerable: true,
    });
    expect(() => validateScene(sceneWith([{
      ...triangle('accessor-color-sets'),
      colorSets: accessorColorSets,
    }]))).toThrow(/colorSets\[0\].*own enumerable data properties/);

    const inheritedInstances = new Array<ReturnType<typeof asMat4>>(1);
    const instancePrototype = Object.create(Array.prototype) as ReturnType<typeof asMat4>[];
    instancePrototype[0] = asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    Object.setPrototypeOf(inheritedInstances, instancePrototype);
    expect(() => validateScene(sceneWith([{
      ...triangle('inherited-instance'),
      kind: 'instanced-mesh',
      instances: inheritedInstances,
    }]))).toThrow(/instances.*dense.*own data property/);

    const thinFilmLayers = new Array<{ ior: number; thicknessNm: number }>(1);
    expect(() => validateMaterialSpec({
      ...MATERIAL,
      thinFilmStack: { layers: thinFilmLayers },
    })).toThrow(/thinFilmStack\.layers.*dense.*own data property/);

    const skinWeights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const morphUvSets = new Array<Array<Float32Array> | undefined>(3);
    morphUvSets[2] = [new Float32Array(6)];
    const validSparseMorph = {
      ...triangle('valid-sparse-morph'),
      kind: 'skinned-mesh' as const,
      uvSets: sparseUvSets,
      skinIndices: new Uint32Array(12),
      skinWeights,
      bones: identity,
      boneInverses: identity.slice(),
      morphTargets: [new Float32Array(9)],
      morphTargetUvSets: morphUvSets,
    };
    expect(() => validateScene(sceneWith([validSparseMorph]))).not.toThrow();

    const missingMorphTarget = new Array<Float32Array>(1);
    expect(() => validateScene(sceneWith([{
      ...validSparseMorph,
      morphTargets: missingMorphTarget,
    }]))).toThrow(/morphTargets.*dense.*own data property/);

    const missingInnerUvTarget = new Array<Float32Array>(1);
    const sparseOuterWithBadInner = new Array<Array<Float32Array> | undefined>(3);
    sparseOuterWithBadInner[2] = missingInnerUvTarget;
    expect(() => validateScene(sceneWith([{
      ...validSparseMorph,
      morphTargetUvSets: sparseOuterWithBadInner,
    }]))).toThrow(/morphTargetUvSets\[2\].*dense.*own data property/);
  });

  it('validates canonical sparse semantic keys across and beyond the native array-index ceiling', () => {
    const nativeCeilingIndex = 0xffff_fffe;
    const ordinaryPropertyIndex = 0x1_0000_0001;
    const uvSets: Array<Float32Array | undefined> = [];
    const colorSets: Array<Float32Array | undefined> = [];
    const morphTargetUvSets: Array<readonly Float32Array[] | undefined> = [];
    for (const index of [nativeCeilingIndex, ordinaryPropertyIndex]) {
      uvSets[index] = new Float32Array([0, 0, 1, 0, 0, 1]);
      colorSets[index] = new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]);
      morphTargetUvSets[index] = [new Float32Array(6)];
    }
    const identity = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    const primitive: ScenePrimitive = {
      ...triangle('high-sparse-semantic-sets'),
      kind: 'skinned-mesh',
      uvSets,
      colorSets,
      skinIndices: new Uint32Array(12),
      skinWeights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
      bones: identity,
      boneInverses: identity.slice(),
      morphTargets: [new Float32Array(9)],
      morphTargetUvSets,
      morphWeights: new Float32Array([0]),
      material: {
        ...MATERIAL,
        baseColorMap: { handle: {}, texCoord: ordinaryPropertyIndex },
      },
    };

    expect(() => validateScene(sceneWith([primitive]))).not.toThrow();
    expect(getPrimitiveUvSet(primitive, nativeCeilingIndex)).toBe(uvSets[nativeCeilingIndex]);
    expect(getPrimitiveUvSet(primitive, ordinaryPropertyIndex)).toBe(uvSets[ordinaryPropertyIndex]);
    expect(getPrimitiveColorSet(primitive, ordinaryPropertyIndex)).toBe(
      colorSets[ordinaryPropertyIndex],
    );

    const accessorSets: Array<Float32Array | undefined> = [];
    Object.defineProperty(accessorSets, String(ordinaryPropertyIndex), {
      get: () => new Float32Array(6),
      enumerable: true,
    });
    expect(() => validateScene(sceneWith([{
      ...triangle('high-accessor-set'),
      uvSets: accessorSets,
    }]))).toThrow(/uvSets.*own enumerable data properties/);

    const nonCanonicalSets: Array<Float32Array | undefined> = [];
    Object.defineProperty(nonCanonicalSets, '01', {
      value: new Float32Array(6),
      enumerable: true,
    });
    expect(() => validateScene(sceneWith([{
      ...triangle('noncanonical-set'),
      uvSets: nonCanonicalSets,
    }]))).toThrow(/uvSets.*"01".*numeric indices/);

    const unsafeSets: Array<Float32Array | undefined> = [];
    Object.defineProperty(unsafeSets, String(Number.MAX_SAFE_INTEGER + 1), {
      value: new Float32Array(6),
      enumerable: true,
    });
    expect(() => validateScene(sceneWith([{
      ...triangle('unsafe-set'),
      uvSets: unsafeSets,
    }]))).toThrow(/uvSets.*numeric indices/);
  });

  it('rejects degenerate vertex frames and non-semantic tangent handedness', () => {
    expect(() => validateScene(sceneWith([{
      ...triangle('zero-normal'),
      normals: new Float32Array(9),
    }]))).toThrow(/normals\[0\].*non-degenerate/);

    const validTangents = new Float32Array([
      1, 0, 0, 1,
      1, 0, 0, -1,
      1, 0, 0, 1,
    ]);
    expect(() => validateScene(sceneWith([{
      ...triangle('valid-tangent-frame'),
      tangents: validTangents,
    }]))).not.toThrow();

    const zeroTangent = validTangents.slice();
    zeroTangent[0] = 0;
    expect(() => validateScene(sceneWith([{
      ...triangle('zero-tangent'),
      tangents: zeroTangent,
    }]))).toThrow(/tangents\[0\].*non-degenerate/);

    const invalidHandedness = validTangents.slice();
    invalidHandedness[3] = 0;
    expect(() => validateScene(sceneWith([{
      ...triangle('bad-handedness'),
      tangents: invalidHandedness,
    }]))).toThrow(/tangents\[0\]\.w.*-1 or 1/);
  });

  it('rejects duplicate mesh-area ownership of one primitive with both paths', () => {
    expect(() => validateScene({
      ...sceneWith(),
      emitters: [
        { kind: 'mesh-area', id: 'first', meshId: 'triangle', color: [1, 1, 1], intensity: 1 },
        { kind: 'mesh-area', id: 'second', meshId: 'triangle', color: [1, 1, 1], intensity: 2 },
      ],
    })).toThrow(
      /scene\.emitters\[1\]\.meshId.*already claimed by scene\.emitters\[0\]\.meshId/,
    );
  });
});
