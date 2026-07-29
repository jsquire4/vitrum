import { describe, expect, it } from 'vitest';
import {
  getPrimitiveActiveColorSet,
  validateScene,
  type MeshPrimitive,
  type Scene,
} from '../index.js';

const COLOR_0 = new Float32Array([
  1, 0, 0,
  1, 0, 0,
  1, 0, 0,
]);
const COLOR_2 = new Float32Array([
  0, 0, 1, 0.25,
  0, 0, 1, 0.5,
  0, 0, 1, 0.75,
]);

function primitive(
  vertexColorSet?: number | null,
): MeshPrimitive {
  const colorSets: Array<Float32Array | undefined> = [];
  colorSets[0] = COLOR_0;
  colorSets[2] = COLOR_2;
  return {
    kind: 'mesh',
    id: 'colored',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    colors: COLOR_0,
    colorSets,
    ...(vertexColorSet !== undefined ? { vertexColorSet } : {}),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
  };
}

function scene(mesh: MeshPrimitive): Scene {
  return {
    primitives: [mesh],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('primitive vertex-color lane selection', () => {
  it('defaults to COLOR_0 and selects arbitrary authored COLOR_n lanes', () => {
    expect(getPrimitiveActiveColorSet(primitive())).toBe(COLOR_0);
    expect(getPrimitiveActiveColorSet(primitive(2))).toBe(COLOR_2);
    expect(() => validateScene(scene(primitive(2)))).not.toThrow();
  });

  it('retains authored lanes while explicitly disabling their material multiplier', () => {
    const mesh = primitive(null);
    expect(getPrimitiveActiveColorSet(mesh)).toBeUndefined();
    expect(mesh.colorSets?.[2]).toBe(COLOR_2);
    expect(() => validateScene(scene(mesh))).not.toThrow();
  });

  it('rejects an absent, negative, fractional, or unsafe selected lane', () => {
    for (const value of [1, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(
        () => validateScene(scene(primitive(value))),
        String(value),
      ).toThrow(/vertexColorSet/);
    }
  });
});
