import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

const MATERIAL: MaterialSpec = {
  baseColor: [0.6, 0.6, 0.6],
  roughness: 1,
  metallic: 0,
};

function scene(): Scene {
  const primitive: MeshPrimitive = {
    kind: 'mesh',
    id: 'tri',
    positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array(6),
    indices: new Uint32Array([0, 1, 2]),
    material: MATERIAL,
  };
  return {
    primitives: [primitive],
    emitters: [
      {
        kind: 'point',
        id: 'old-light',
        position: [0, 1, 0],
        color: [1, 1, 1],
        intensity: 1,
      },
    ],
    environment: { kind: 'none' },
  };
}

describe('PTEngineWebGL2.updateLighting', () => {
  it('publishes emitters and environment together and advertises the method', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(scene());

    engine.updateLighting?.({
      emitters: [
        {
          kind: 'directional',
          id: 'sun',
          direction: [0, -1, 0],
          color: [1, 0.9, 0.8],
          intensity: 3,
        },
      ],
      environment: {
        kind: 'hdri',
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([0.25, 0.5, 1, 1]),
        },
        intensity: 0.5,
      },
    });

    expect(typeof engine.updateLighting).toBe('function');
    expect(engine.getScene!()?.emitters.map((emitter) => emitter.id)).toEqual(['sun']);
    expect(engine.getScene!()?.environment.kind).toBe('hdri');
    engine.dispose();
  });

  it('leaves both retained lighting domains unchanged when validation fails', async () => {
    const engine = await createPTEngine_WebGL2({ device: createMockGl() });
    engine.setScene(scene());
    const before = engine.getScene!();

    expect(() =>
      engine.updateLighting?.({
        emitters: [
          {
            kind: 'point',
            id: 'invalid-light',
            position: [0, 1, 0],
            color: [1, 1, 1],
            intensity: Number.NaN,
          },
        ],
        environment: {
          kind: 'procedural-sky',
          sunDirection: [0, 1, 0],
          intensity: 2,
        },
      }),
    ).toThrow();

    expect(engine.getScene!()).toBe(before);
    expect(engine.getScene!()?.emitters[0]?.id).toBe('old-light');
    expect(engine.getScene!()?.environment.kind).toBe('none');
    engine.dispose();
  });

  it('supports null environment and rejects unknown keys before retained-state mutation', async () => {
    const engine = await createPTEngine_WebGL2({
      device: createMockGl(),
    });
    engine.setScene(scene());
    const before = engine.getScene!();

    expect(() => engine.updateLighting?.({
      emitters: [],
      environment: {
        kind: 'procedural-sky',
        sunDirection: [0, 1, 0],
        intensity: 2,
      },
      vendorOnly: true,
    })).toThrow('updateLighting: unknown option key "vendorOnly"');
    expect(engine.getScene!()).toBe(before);
    expect(engine.getScene!()?.emitters[0]?.id).toBe('old-light');
    expect(engine.getScene!()?.environment.kind).toBe('none');

    engine.updateLighting?.({ environment: null });
    expect(engine.getScene!()?.environment.kind).toBe('none');
    engine.dispose();
  });
});
