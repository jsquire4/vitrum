import { describe, expect, it } from 'vitest';
import {
  maybeDisplaceMeshPositions,
  maybeMicrodisplaceMeshGeometry,
  resolveDisplacedGeometry,
} from '../index.js';

const POSITIONS = new Float32Array([
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
]);
const NORMALS = new Float32Array([
  0, 0, 1,
  0, 0, 1,
  0, 0, 1,
]);
const UV3 = new Float32Array([0, 0, 1, 0, 0, 0]);
const HEIGHT = {
  width: 2,
  height: 1,
  data: new Float32Array([0, 1]),
  channels: 1,
};

describe('arbitrary UV sets in shared displacement', () => {
  it('samples vertex displacement from the authored sparse texCoord', () => {
    const displaced = maybeDisplaceMeshPositions({
      primitiveId: 'uv3',
      material: {
        baseColor: [1, 1, 1],
        roughness: 0.5,
        metallic: 0,
        displacementMap: {
          handle: HEIGHT,
          texCoord: 3,
          wrapS: 'clamp-to-edge',
          wrapT: 'clamp-to-edge',
        },
      },
      positions: POSITIONS,
      normals: NORMALS,
      uvSets: [undefined, undefined, undefined, UV3],
    });

    expect(displaced).not.toBeNull();
    expect(displaced?.[2]).toBeCloseTo(0);
    expect(displaced?.[5]).toBeCloseTo(1);
  });

  it('interpolates every present UV set while microdisplacing', () => {
    const warnings: string[] = [];
    const diced = maybeMicrodisplaceMeshGeometry({
      primitiveId: 'uv3-diced',
      material: {
        baseColor: [1, 1, 1],
        roughness: 0.5,
        metallic: 0,
        displacementMap: {
          handle: HEIGHT,
          texCoord: 3,
          wrapS: 'clamp-to-edge',
          wrapT: 'clamp-to-edge',
        },
        displacementSubdivisions: 1,
      },
      positions: POSITIONS,
      normals: NORMALS,
      indices: new Uint32Array([0, 1, 2]),
      uvSets: [undefined, undefined, undefined, UV3],
      onWarning: (warning) => warnings.push(warning),
    });

    expect(warnings).toEqual([]);
    expect(diced).not.toBeNull();
    expect(diced?.uvSets?.[3]).toHaveLength((diced?.positions.length ?? 0) / 3 * 2);
    expect(Math.max(...(diced?.uvSets?.[3] ?? []))).toBeCloseTo(1);
  });

  it('preserves and samples sparse UV sets at and above the native array-index ceiling', () => {
    const nativeCeilingIndex = 0xffff_fffe;
    const ordinaryPropertyIndex = 0x1_0000_0001;
    const uvSets: Array<Float32Array | undefined> = [];
    uvSets[nativeCeilingIndex] = UV3;
    uvSets[ordinaryPropertyIndex] = UV3;
    const material = {
      baseColor: [1, 1, 1] as [number, number, number],
      roughness: 0.5,
      metallic: 0,
      displacementMap: {
        handle: HEIGHT,
        texCoord: ordinaryPropertyIndex,
        wrapS: 'clamp-to-edge' as const,
        wrapT: 'clamp-to-edge' as const,
      },
      displacementSubdivisions: 1,
    };

    const diced = maybeMicrodisplaceMeshGeometry({
      primitiveId: 'uv-high-diced',
      material,
      positions: POSITIONS,
      normals: NORMALS,
      indices: new Uint32Array([0, 1, 2]),
      uvSets,
    });

    expect(diced?.uvSets?.[nativeCeilingIndex]).toHaveLength(
      (diced?.positions.length ?? 0) / 3 * 2,
    );
    expect(diced?.uvSets?.[ordinaryPropertyIndex]).toHaveLength(
      (diced?.positions.length ?? 0) / 3 * 2,
    );
    const resolved = resolveDisplacedGeometry({
      id: 'uv-high-resolved',
      material,
      positions: POSITIONS,
      normals: NORMALS,
      indices: new Uint32Array([0, 1, 2]),
      uvSets,
    }, () => {});
    expect(resolved.baseUvSets?.[nativeCeilingIndex]).toBeInstanceOf(Float32Array);
    expect(resolved.baseUvSets?.[ordinaryPropertyIndex]).toBeInstanceOf(Float32Array);
  });
});
