import { describe, expect, it } from 'vitest';
import { Mesh } from 'three';
import { vitrumSceneToThree } from '../sceneToThree.js';

describe('vitrumSceneToThree layered userData', () => {
  it('stamps frontLayer/backLayer metadata for fork packing', () => {
    const scene = vitrumSceneToThree({
      primitives: [
        {
          kind: 'mesh',
          id: 'pane',
          positions: new Float32Array([
            -1, -1, 0,
            1, -1, 0,
            0, 1, 0,
          ]),
          normals: new Float32Array([
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
          ]),
          indices: new Uint32Array([0, 1, 2]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.2,
            metallic: 0,
            transmission: 1,
            ior: 1.52,
            frontLayer: {
              transmission: [0.2, 0.3, 0.4],
              roughness: 0.6,
            },
            backLayer: {
              transmission: [0.9, 0.9, 0.9],
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    } as never);

    const mesh = scene.children.find((x) => x instanceof Mesh) as Mesh | undefined;
    expect(mesh).toBeDefined();
    const material = mesh?.material as { userData?: Record<string, unknown> } | undefined;
    const front = material?.userData?.['vitrumFrontLayer'] as { transmission?: number[]; roughness?: number };
    const back = material?.userData?.['vitrumBackLayer'] as { transmission?: number[] };

    expect(front.transmission).toEqual([0.2, 0.3, 0.4]);
    expect(front.roughness).toBeCloseTo(0.6);
    expect(back.transmission).toEqual([0.9, 0.9, 0.9]);
  });
});
