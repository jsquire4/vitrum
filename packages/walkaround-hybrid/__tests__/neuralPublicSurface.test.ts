import { describe, expect, it } from 'vitest';
import * as publicApi from '../src/index.js';

describe('neural package-root surface', () => {
  it('does not expose uncertified random-weight fixtures as a host activation path', () => {
    expect(publicApi).not.toHaveProperty('buildRandomWeightsForSpec');
    expect(publicApi).toHaveProperty('loadWeightsFromArrayBuffer');
    expect(publicApi).toHaveProperty('assessNeuralCheckpointProductionReadiness');
  });
});
