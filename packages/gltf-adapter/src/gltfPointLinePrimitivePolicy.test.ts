import { describe, expect, it } from 'vitest';
import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  gltfToScene,
  type GltfJson,
} from './index.js';

const POINT_LINE_MODES = [
  { mode: 0, name: 'POINTS' },
  { mode: 1, name: 'LINES' },
  { mode: 2, name: 'LINE_LOOP' },
  { mode: 3, name: 'LINE_STRIP' },
] as const;

function makePointLineModeGltf(): GltfJson {
  return {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      name: 'unsupported-topologies',
      primitives: POINT_LINE_MODES.map(({ mode }) => ({
        attributes: { POSITION: 0 },
        mode,
      })),
    }],
  };
}

describe('POINTS / line primitive policy', () => {
  it('reports each point/line topology as a structured unsupported compatibility issue', () => {
    const report = analyzeGltfAsset(makePointLineModeGltf());
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.byMode).toEqual({
      '0': 1,
      '1': 1,
      '2': 1,
      '3': 1,
    });
    expect(report.primitives.unsupportedModes).toEqual(['0', '1', '2', '3']);

    for (const { mode } of POINT_LINE_MODES) {
      expect(compatibility.issues).toContainEqual(expect.objectContaining({
        category: 'primitive',
        name: `mode:${mode}`,
        support: 'unsupported',
        message: `glTF primitive mode ${mode} has no core primitive representation.`,
      }));
    }

    expect(compatibility.unsupportedCount).toBe(POINT_LINE_MODES.length);
    expect(compatibility.isCompatible).toBe(false);
  });

  it('warns once per unsupported topology and skips every primitive', async () => {
    const { scene, warnings } = await gltfToScene(makePointLineModeGltf());

    expect(scene.primitives).toHaveLength(0);
    for (const { mode, name } of POINT_LINE_MODES) {
      expect(warnings).toContain(
        `[vitrum/gltf-adapter] Mesh "unsupported-topologies" primitive has unsupported ` +
          `mode ${mode} (${name}). Only TRIANGLES (4), TRIANGLE_STRIP (5) and ` +
          'TRIANGLE_FAN (6) are supported (core has no point/line primitive). ' +
          'This primitive is SKIPPED.',
      );
    }
    expect(warnings).toHaveLength(POINT_LINE_MODES.length);
  });
});
