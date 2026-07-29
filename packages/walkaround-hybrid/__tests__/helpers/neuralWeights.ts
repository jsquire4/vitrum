import {
  VITRUM_MODEL_MAGIC,
  type LayerWeights,
  type ModelWeights,
} from '../../src/neural/weights.js';

interface NeuralWeightFixtureSpec {
  readonly layers: readonly {
    readonly name: string;
    readonly weightLayout: 'OIKW' | 'IOKW' | 'none';
    readonly params: {
      readonly inC: number;
      readonly outC: number;
      readonly kH?: number;
      readonly kW?: number;
    };
  }[];
}

/**
 * Build deterministic, uncertified weights for neural unit/GPU tests.
 *
 * These weights intentionally have no checkpoint metadata and are not accepted
 * by the production `denoiser: 'neural'` path.
 */
export function buildRandomWeightsForSpec(
  spec: NeuralWeightFixtureSpec,
  seed: number = VITRUM_MODEL_MAGIC,
): ModelWeights {
  let state = (seed >>> 0) || 1;
  const nextUnit = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const nextRange = (lo: number, hi: number): number => lo + nextUnit() * (hi - lo);

  const layers: LayerWeights[] = spec.layers.map((layer) => {
    if (layer.weightLayout === 'none') {
      return {
        name: layer.name,
        weights: new Float32Array(0),
        biases: new Float32Array(0),
      };
    }

    const { inC, outC, kH = 1, kW = 1 } = layer.params;
    const count = inC * outC * kH * kW;
    const scale = Math.sqrt(2.0 / Math.max(1, inC * kH * kW));
    const weights = new Float32Array(count);
    for (let i = 0; i < count; i++) weights[i] = nextRange(-scale, scale);
    const biases = new Float32Array(outC);
    for (let i = 0; i < outC; i++) biases[i] = nextRange(-0.01, 0.01);
    return { name: layer.name, weights, biases };
  });

  return { layers };
}
