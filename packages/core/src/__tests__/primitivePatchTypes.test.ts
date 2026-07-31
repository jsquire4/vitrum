import { describe, expect, it } from 'vitest';
import type { ScenePrimitivePatch } from '../scene/primitives.js';

describe('ScenePrimitivePatch type contract', () => {
  it('accepts bounded partial and explicit-clear updates', () => {
    const nestedLayerPatch: ScenePrimitivePatch = {
      material: {
        frontLayer: {
          roughness: 0.25,
        },
      },
    };
    const optionalMaterialClear: ScenePrimitivePatch = {
      material: {
        emissive: undefined,
      },
    };
    const optionalPrimitiveClear: ScenePrimitivePatch = {
      transform: undefined,
    };

    expect(nestedLayerPatch.material?.frontLayer?.roughness).toBe(0.25);
    expect(Object.hasOwn(optionalMaterialClear.material ?? {}, 'emissive')).toBe(true);
    expect(Object.hasOwn(optionalPrimitiveClear, 'transform')).toBe(true);
  });

  const compileOnlyRejectedAssignments = (): void => {
    // @ts-expect-error identity is selected by updatePrimitive(id, patch)
    const identityMutation: ScenePrimitivePatch = { id: 'replacement' };
    // @ts-expect-error the primitive discriminant cannot be patched
    const kindMutation: ScenePrimitivePatch = { kind: 'analytic' };
    // @ts-expect-error no primitive member owns both instances and transform
    const instanceTransformMix: ScenePrimitivePatch = {
      instances: [],
      transform: undefined,
    };
    // @ts-expect-error no primitive member owns both mesh positions and analytic shape
    const meshAnalyticMix: ScenePrimitivePatch = {
      positions: new Float32Array(),
      shape: 'sphere',
    };
    const requiredMaterialClear: ScenePrimitivePatch = {
      // @ts-expect-error required complete-material fields cannot be explicitly cleared
      material: { roughness: undefined },
    };

    void identityMutation;
    void kindMutation;
    void instanceTransformMix;
    void meshAnalyticMix;
    void requiredMaterialClear;
  };
  void compileOnlyRejectedAssignments;
});
