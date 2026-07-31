import { describe, expect, it } from 'vitest';
import { validateScene, type Scene } from '../scene/index.js';
import { patchEmitterInScene, patchPrimitiveInScene } from '../scene/patchScene.js';

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
        ]),
        normals: new Float32Array([
          0, 0, 1,
          0, 0, 1,
          0, 0, 1,
        ]),
        uvs: new Float32Array([
          0, 0,
          1, 0,
          0, 1,
        ]),
        material: {
          baseColor: [0.1, 0.2, 0.3],
          roughness: 0.5,
          metallic: 0.2,
        },
      },
      {
        kind: 'analytic',
        id: 'sphere-a',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: {
          baseColor: [1, 0, 0],
          roughness: 0.3,
          metallic: 0,
        },
      },
    ],
    emitters: [
      {
        kind: 'directional',
        id: 'sun',
        direction: [0, -1, 0],
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
    environment: { kind: 'none' },
  };
}

describe('patchScene helpers', () => {
  it('patches a primitive by id and leaves the original scene unmutated', () => {
    const scene = makeScene();
    const primitive = scene.primitives[0];
    if (primitive == null) throw new Error('missing primitive in test scene');

    const next = patchPrimitiveInScene(scene, 'mesh-a', {
      material: { ...primitive.material, roughness: 0.1 },
    });

    const nextPrimitive = next.primitives[0];
    if (nextPrimitive == null) throw new Error('missing patched primitive');
    expect(nextPrimitive.material.roughness).toBe(0.1);

    // Original scene is untouched (pure helper).
    expect(primitive.material.roughness).toBe(0.5);
    expect(next).not.toBe(scene);
    expect(next.primitives).not.toBe(scene.primitives);
    expect(next.emitters).toBe(scene.emitters);
  });

  it('deep-merges layered material patches without dropping nested normal descriptors', () => {
    const scene = makeScene();
    const primitive = scene.primitives[0];
    if (primitive == null) throw new Error('missing primitive in test scene');
    const frontNormal = { handle: { id: 'front-normal' } };
    const backNormal = { handle: { id: 'back-normal' } };
    const layeredScene: Scene = {
      ...scene,
      primitives: [{
        ...primitive,
        material: {
          ...primitive.material,
          frontLayer: {
            transmission: [1, 1, 1],
            roughness: 0.3,
            normalMap: frontNormal,
            normalScale: 0.75,
          },
          backLayer: {
            transmission: [0.9, 0.8, 0.7],
            roughness: 0.4,
            normalMap: backNormal,
            normalScale: 0.5,
          },
        },
      }],
    };

    const next = patchPrimitiveInScene(layeredScene, 'mesh-a', {
      material: {
        frontLayer: {
          transmission: [0.5, 0.6, 0.7],
        },
      },
    } as never);

    const patched = next.primitives[0];
    if (patched == null) throw new Error('missing patched primitive');
    expect(patched.material.frontLayer).toEqual({
      transmission: [0.5, 0.6, 0.7],
      roughness: 0.3,
      normalMap: frontNormal,
      normalScale: 0.75,
    });
    expect(patched.material.backLayer).toEqual({
      transmission: [0.9, 0.8, 0.7],
      roughness: 0.4,
      normalMap: backNormal,
      normalScale: 0.5,
    });
  });

  it('lets partial layered material patches explicitly clear nested fields', () => {
    const scene = makeScene();
    const primitive = scene.primitives[0];
    if (primitive == null) throw new Error('missing primitive in test scene');
    const normalMap = { handle: { id: 'front-normal' } };
    const layeredScene: Scene = {
      ...scene,
      primitives: [{
        ...primitive,
        material: {
          ...primitive.material,
          frontLayer: {
            transmission: [1, 1, 1],
            roughness: 0.3,
            normalMap,
            normalScale: 0.75,
          },
        },
      }],
    };

    const next = patchPrimitiveInScene(layeredScene, 'mesh-a', {
      material: {
        frontLayer: {
          normalMap: undefined,
        },
      },
    });

    const patched = next.primitives[0];
    if (patched == null) throw new Error('missing patched primitive');
    expect(patched.material.frontLayer).toEqual({
      transmission: [1, 1, 1],
      roughness: 0.3,
      normalMap: undefined,
      normalScale: 0.75,
    });
  });

  it('patches an emitter by id and leaves the original scene unmutated', () => {
    const scene = makeScene();
    const emitter = scene.emitters[0];
    if (emitter == null) throw new Error('missing emitter in test scene');

    const next = patchEmitterInScene(scene, 'sun', { intensity: 5 });
    const nextEmitter = next.emitters[0];
    if (nextEmitter == null) throw new Error('missing patched emitter');

    expect(nextEmitter.intensity).toBe(5);
    expect(emitter.intensity).toBe(2);
    expect(next.primitives).toBe(scene.primitives);
  });

  it('rejects emitter and material patches whose positive radiance is not representable in Float32', () => {
    const scene = makeScene();
    const maxFloat32 = Math.fround(3.4028234663852886e38);
    const minFloat32 = Math.fround(1.401298464324817e-45);

    expect(() => patchEmitterInScene(scene, 'sun', {
      color: [maxFloat32, 0, 0],
      intensity: 2,
    })).toThrow(/color.*intensity.*finite.*Float32/);
    expect(() => patchEmitterInScene(scene, 'sun', {
      color: [minFloat32, 0, 0],
      intensity: 0.5,
    })).toThrow(/color.*intensity.*underflow.*Float32/);

    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      material: {
        emissive: [maxFloat32, 0, 0],
        emissiveIntensity: 2,
      },
    })).toThrow(/emissive.*emissiveIntensity.*finite.*Float32/);
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      material: {
        emissive: [minFloat32, 0, 0],
        emissiveIntensity: 0.5,
      },
    })).toThrow(/emissive.*emissiveIntensity.*underflow.*Float32/);
  });

  it('throws when the primitive id is missing (with the unified updatePrimitive: prefix)', () => {
    const scene = makeScene();
    expect(() => patchPrimitiveInScene(scene, 'missing', {})).toThrow(
      /^updatePrimitive: .*not found/,
    );
  });

  it('throws when the emitter id is missing (with the unified updateEmitter: prefix)', () => {
    const scene = makeScene();
    expect(() => patchEmitterInScene(scene, 'missing', {})).toThrow(
      /^updateEmitter: .*not found/,
    );
  });

  it('throws when a primitive patch tries to change the id', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', { id: 'mesh-b' } as never),
    ).toThrow(/id cannot be changed/);
  });

  it('throws when an emitter patch tries to change the id', () => {
    const scene = makeScene();
    expect(() =>
      patchEmitterInScene(scene, 'sun', { id: 'moon' }),
    ).toThrow(/id cannot be changed/);
  });

  it('throws when a primitive patch tries to change the kind', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', { kind: 'analytic' } as never),
    ).toThrow(/kind cannot change/);
  });

  it('rejects every own primitive id/kind field, even unchanged or undefined', () => {
    const scene = makeScene();
    for (const patch of [
      { id: 'mesh-a' },
      { id: undefined },
    ]) {
      expect(() =>
        patchPrimitiveInScene(scene, 'mesh-a', patch as never),
      ).toThrow(/id cannot be changed or supplied/);
    }
    for (const patch of [
      { kind: 'mesh' },
      { kind: undefined },
    ]) {
      expect(() =>
        patchPrimitiveInScene(scene, 'mesh-a', patch as never),
      ).toThrow(/kind cannot change or be supplied/);
    }
  });

  it('throws when an emitter patch tries to change the kind', () => {
    const scene = makeScene();
    expect(() =>
      patchEmitterInScene(scene, 'sun', { kind: 'point' } as never),
    ).toThrow(/kind cannot change/);
  });

  it('validates analytic params and rejects a wrong-length shape param', () => {
    const scene = makeScene();
    // sphere expects 4 values; supply 3 → must throw via validateAnalyticParams.
    expect(() =>
      patchPrimitiveInScene(scene, 'sphere-a', {
        params: new Float32Array([0, 0, 0]),
      }),
    ).toThrow(/expects 4/);
  });

  it('throws when an analytic primitive resolves to missing shape/params', () => {
    // Defensive guard (patchScene.ts): a malformed analytic primitive lacking
    // `params`, patched on an unrelated field, must surface as an explicit
    // "requires shape and params" error rather than producing a broken Scene.
    const base = makeScene();
    const malformed = {
      kind: 'analytic',
      id: 'broken-analytic',
      shape: 'sphere',
      // params intentionally omitted
      material: { baseColor: [0, 1, 0], roughness: 0.4, metallic: 0 },
    };
    const scene: Scene = { ...base, primitives: [...base.primitives, malformed as never] };
    expect(() =>
      patchPrimitiveInScene(scene, 'broken-analytic', {
        material: { baseColor: [0, 0, 1], roughness: 0.2, metallic: 0 },
      } as never),
    ).toThrow(/requires shape and params/);
  });

  it('accepts a valid analytic param update', () => {
    const scene = makeScene();
    const next = patchPrimitiveInScene(scene, 'sphere-a', {
      params: new Float32Array([1, 2, 3, 4]),
    });
    const patched = next.primitives[1];
    if (patched == null || patched.kind !== 'analytic') {
      throw new Error('missing patched analytic primitive');
    }
    expect(Array.from(patched.params)).toEqual([1, 2, 3, 4]);
  });

  it('rejects analytic "params" on a mesh-like primitive', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', {
        params: new Float32Array([0, 0, 0, 1]),
      }),
    ).toThrow(/cannot accept analytic "params"/);
  });

  it('rejects analytic-only shape/fallbackMesh fields on a mesh-like primitive', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', {
        shape: 'sphere',
      } as never),
    ).toThrow(/cannot accept analytic "shape"/);

    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', {
        fallbackMesh: {
          positions: new Float32Array([0, 0, 0]),
          normals: new Float32Array([0, 0, 1]),
        },
      }),
    ).toThrow(/cannot accept analytic "fallbackMesh"/);
  });

  it('rejects unknown and cross-kind primitive patch fields without mutating the source', () => {
    const scene = makeScene();
    const retainedPrimitive = scene.primitives[0];
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      unknownPrimitivePatch: true,
    } as never)).toThrow(/unknownPrimitivePatch.*known contract field/);
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      instances: [],
    })).toThrow(/instances.*known contract field/);
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      material: { unknownMaterialPatch: true },
    } as never)).toThrow(/unknownMaterialPatch.*known contract field/);
    expect(scene.primitives[0]).toBe(retainedPrimitive);
  });

  it('rejects non-enumerable, symbol, and accessor primitive patch fields without invoking accessors', () => {
    const scene = makeScene();

    const nonEnumerable = {};
    Object.defineProperty(nonEnumerable, 'material', {
      enumerable: false,
      value: { roughness: 0.1 },
    });
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', nonEnumerable),
    ).toThrow(/material.*enumerable/);

    const symbolPatch = {};
    Object.defineProperty(symbolPatch, Symbol('hidden-mutation'), {
      enumerable: false,
      value: true,
    });
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', symbolPatch),
    ).toThrow(/symbol field.*not allowed/);

    let getterReads = 0;
    const accessorPatch = {};
    Object.defineProperty(accessorPatch, 'material', {
      enumerable: true,
      get() {
        getterReads += 1;
        return { roughness: 0.1 };
      },
    });
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', accessorPatch),
    ).toThrow(/material.*own data property/);
    expect(getterReads).toBe(0);
  });

  it('rejects inherited primitive patch fields without invoking prototype accessors', () => {
    const scene = makeScene();
    let getterReads = 0;
    const prototype = {};
    Object.defineProperty(prototype, 'material', {
      enumerable: true,
      get() {
        getterReads += 1;
        return { roughness: 0.1 };
      },
    });
    const inheritedPatch = Object.create(prototype);

    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', inheritedPatch),
    ).toThrow(/patch must be a plain data object/);
    expect(getterReads).toBe(0);
  });

  it('rejects nested material and layer accessors without invoking them', () => {
    const scene = makeScene();
    let materialGetterReads = 0;
    const materialPatch = {};
    Object.defineProperty(materialPatch, 'roughness', {
      enumerable: true,
      get() {
        materialGetterReads += 1;
        return 0.1;
      },
    });
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', { material: materialPatch }),
    ).toThrow(/patch material field "roughness" must be an own data property/);
    expect(materialGetterReads).toBe(0);

    let layerGetterReads = 0;
    const frontLayerPatch = {
      transmission: [0.8, 0.7, 0.6],
    };
    Object.defineProperty(frontLayerPatch, 'roughness', {
      enumerable: true,
      get() {
        layerGetterReads += 1;
        return 0.2;
      },
    });
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', {
        material: { frontLayer: frontLayerPatch },
      } as never),
    ).toThrow(/patch material\.frontLayer field "roughness" must be an own data property/);
    expect(layerGetterReads).toBe(0);
  });

  it('rejects non-object and array primitive patches at the runtime boundary', () => {
    const scene = makeScene();
    for (const patch of [null, undefined, 1, 'material', [], new Float32Array()]) {
      expect(() =>
        patchPrimitiveInScene(scene, 'mesh-a', patch as never),
      ).toThrow(/patch must be a non-array object/);
    }
  });

  it('rejects cross-kind fields beyond the analytic/mesh split', () => {
    const scene = makeScene();
    expect(() =>
      patchPrimitiveInScene(scene, 'mesh-a', {
        instances: [],
        transform: undefined,
      } as never),
    ).toThrow(/instances.*known contract field/);
    expect(() =>
      patchPrimitiveInScene(scene, 'sphere-a', {
        positions: new Float32Array([0, 0, 0]),
      }),
    ).toThrow(/positions.*known contract field/);
  });

  it('rejects unknown and cross-kind emitter patch fields without mutating the source', () => {
    const scene = makeScene();
    const retainedEmitter = scene.emitters[0];
    expect(() => patchEmitterInScene(scene, 'sun', {
      unknownEmitterPatch: true,
    } as never)).toThrow(/unknownEmitterPatch.*known contract field/);
    expect(() => patchEmitterInScene(scene, 'sun', {
      position: [0, 0, 0],
    } as never)).toThrow(/position.*known contract field/);
    expect(scene.emitters[0]).toBe(retainedEmitter);
  });

  it('validates only the patched primitive after an accepted snapshot', () => {
    const accepted = makeScene();
    // Model a host violating the readonly snapshot after engine acceptance:
    // the untouched node is now malformed. The hot patch path must neither
    // traverse nor accidentally bless it; a later full setScene still rejects
    // that snapshot through validateScene.
    const malformedUntouched = {
      ...accepted.primitives[1]!,
      params: new Float32Array([0]),
    } as Scene['primitives'][number];
    const scene: Scene = {
      ...accepted,
      primitives: [accepted.primitives[0]!, malformedUntouched],
    };
    const next = patchPrimitiveInScene(scene, 'mesh-a', {
      material: { roughness: 0.25 },
    });
    expect(next.primitives[0]!.material.roughness).toBe(0.25);
    expect(next.primitives[1]).toBe(malformedUntouched);
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      material: { roughness: Number.NaN },
    })).toThrow(/roughness.*finite/);
  });

  it('does not access unchanged geometry on a material-only patch', () => {
    const base = makeScene();
    validateScene(base);
    const source = base.primitives[0]!;
    const guarded = {
      ...source,
      indices: new Uint16Array([0, 1, 2]),
    } as Scene['primitives'][number];
    const geometry = {
      positions: source.kind === 'mesh' ? source.positions : undefined,
      normals: source.kind === 'mesh' ? source.normals : undefined,
      uvs: source.kind === 'mesh' ? source.uvs : undefined,
      indices: new Uint16Array([0, 1, 2]),
    };
    const reads: string[] = [];
    for (const key of ['positions', 'normals', 'uvs', 'indices'] as const) {
      Object.defineProperty(guarded, key, {
        configurable: true,
        enumerable: true,
        get() {
          reads.push(key);
          return geometry[key];
        },
      });
    }
    const scene: Scene = {
      ...base,
      primitives: [guarded, ...base.primitives.slice(1)],
    };

    const next = patchPrimitiveInScene(scene, 'mesh-a', {
      material: { roughness: 0.125 },
    });

    expect(next.primitives[0]!.material.roughness).toBe(0.125);
    expect(reads).toEqual([]);
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      material: { roughness: Number.NaN },
    })).toThrow(/roughness.*finite/);
    expect(reads).toEqual([]);
  });

  it('keeps material-to-UV cross-field validation on the fast path', () => {
    const scene = makeScene();
    expect(() => patchPrimitiveInScene(scene, 'mesh-a', {
      material: {
        baseColorMap: {
          handle: { id: 'base-color' },
          texCoord: 7,
        },
      },
    })).toThrow(/baseColorMap\.texCoord.*does not provide that UV stream/);
  });

  it('validates changed mesh-area references and unique ownership', () => {
    const base = makeScene();
    const secondMesh = {
      ...base.primitives[0]!,
      id: 'mesh-b',
    } as Scene['primitives'][number];
    const scene: Scene = {
      ...base,
      primitives: [...base.primitives, secondMesh],
      emitters: [
        {
          kind: 'mesh-area',
          id: 'area-a',
          meshId: 'mesh-a',
          color: [1, 1, 1],
          intensity: 1,
        },
        {
          kind: 'mesh-area',
          id: 'area-b',
          meshId: 'mesh-b',
          color: [1, 1, 1],
          intensity: 1,
        },
      ],
    };
    expect(() => patchEmitterInScene(scene, 'area-b', { meshId: 'missing' }))
      .toThrow(/references missing primitive/);
    expect(() => patchEmitterInScene(scene, 'area-b', { meshId: 'sphere-a' }))
      .toThrow(/mesh-like primitive/);
    expect(() => patchEmitterInScene(scene, 'area-b', { meshId: 'mesh-a' }))
      .toThrow(/duplicates mesh-area ownership/);
  });
});
