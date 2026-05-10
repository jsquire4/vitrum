import { describe, it, expect } from 'vitest';
import type { Scene } from '@vitrum/core';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';

/** Serializable scene summary for regression tests (M5 golden). */
function summarizeScene(s: Scene): Record<string, unknown> {
  return {
    primitiveCount: s.primitives.length,
    meshCount: s.primitives.filter((p) => p.kind === 'mesh').length,
    emitterKinds: [...new Set(s.emitters.map((e) => e.kind))].sort(),
    environmentKind: s.environment.kind,
  };
}

describe('pt-webgl scene sync (golden summary)', () => {
  it('Cornell THREE → core Scene has stable topology', () => {
    const three = buildCornellBoxThreeScene();
    const scene = sceneFromThreeJS(three);
    expect(summarizeScene(scene)).toEqual({
      primitiveCount: 7,
      meshCount: 7,
      emitterKinds: ['rect-area'],
      environmentKind: 'none',
    });
  });
});
