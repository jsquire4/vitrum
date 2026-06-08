import { describe, expect, it } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  partitionSceneBySupport,
  type BackendId,
  type SupportSets,
} from '@vitrum/core';
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';
import { auditPtWebglSceneForTlas } from '@vitrum/pt-webgl';
import { sceneFromThreeJS } from '@vitrum/three-bindings';

function supportSetsFor(backend: BackendId): SupportSets {
  const rec = BACKEND_PROMISE_LEDGER[backend];
  return {
    supportedPrimitiveKinds: new Set(rec.supportedPrimitiveKinds),
    supportedEmitterKinds: new Set(rec.supportedEmitterKinds),
    supportedAnalyticShapes: new Set(rec.supportedAnalyticShapes),
    supportedEnvironmentKinds: new Set(rec.supportedEnvironmentKinds),
  };
}

describe('two-engines-one-scene scene contract smoke', () => {
  it('builds one core scene accepted by pt-webgl and walkaround-hybrid gates without GPU', () => {
    const coreScene = sceneFromThreeJS(buildCornellBoxThreeScene());
    expect(coreScene.primitives.length).toBeGreaterThan(0);

    const ptWebgl = partitionSceneBySupport(coreScene, supportSetsFor('pt-webgl'));
    expect(ptWebgl.warnings).toEqual([]);
    expect(ptWebgl.supported.primitives).toHaveLength(coreScene.primitives.length);

    const ptWebglAudit = auditPtWebglSceneForTlas(ptWebgl.supported);
    expect(ptWebglAudit.meshLikePrimitiveCount).toBeGreaterThan(0);
    expect(ptWebglAudit.totalInstanceCount).toBe(0);

    const walkaround = partitionSceneBySupport(coreScene, supportSetsFor('walkaround-hybrid'));
    expect(walkaround.warnings).toEqual([]);
    expect(walkaround.supported.primitives.map((p) => p.id)).toEqual(
      ptWebgl.supported.primitives.map((p) => p.id),
    );
    expect(walkaround.supported.emitters.map((e) => e.id)).toEqual(
      ptWebgl.supported.emitters.map((e) => e.id),
    );
  });
});
