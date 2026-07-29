import { describe, expect, expectTypeOf, it } from 'vitest';
import {
  createWalkaroundEngine_Hybrid,
  type HybridEngine as PublicHybridEngine,
} from '../src/index.js';
import { HybridEngine as ConcreteHybridEngine } from '../src/HybridEngine.js';

describe('PPG durable-failure recovery public surface', () => {
  it('exposes the concrete recovery action through the package-root engine type', () => {
    expectTypeOf<PublicHybridEngine['requestPpgTrainingRecovery']>()
      .toEqualTypeOf<() => boolean>();
    expectTypeOf<Awaited<ReturnType<typeof createWalkaroundEngine_Hybrid>>>()
      .toHaveProperty('requestPpgTrainingRecovery');
    expect(typeof ConcreteHybridEngine.prototype.requestPpgTrainingRecovery).toBe('function');
    expectTypeOf<PublicHybridEngine['getPpgTrainingStatus']>()
      .toEqualTypeOf<
        () =>
          | 'unavailable'
          | 'disabled'
          | 'collecting'
          | 'readback'
          | 'retry-pending'
          | 'failed'
          | 'disposed'
      >();
    expect(typeof ConcreteHybridEngine.prototype.getPpgTrainingStatus).toBe('function');
  });
});
