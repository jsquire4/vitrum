import { describe, expect, it } from 'vitest';

import {
  dispatchWorkgroupsFor,
  packLayerUniform,
  preflightTensorDims,
} from '../tensorDimSolver.js';
import { buildUNetSpec } from '../unetArchitecture.js';

describe('neural elementwise dispatch tiling', () => {
  it('tiles canonical 1080p work across rows without an index gap', () => {
    const spec = buildUNetSpec();
    const dims = preflightTensorDims(spec, 1920, 1080);
    const layer = spec.layers.find(candidate => candidate.name === 'enc1_relu');
    expect(layer).toBeDefined();
    const output = dims.get('enc1_relu_out');
    expect(output).toBeDefined();

    const maxGroups = 65_535;
    const groups = dispatchWorkgroupsFor('relu', output!, maxGroups);
    expect(groups).toEqual([65_535, 3, 1]);

    const uniforms = new Uint32Array(
      packLayerUniform(layer!, dims, 1080, 1920, maxGroups),
    );
    expect(uniforms[0]).toBe(1920 * 1080 * 24);
    expect(uniforms[1]).toBe(groups[0]);

    const lastIndexInRowZero = groups[0] * 256 - 1;
    const firstIndexInRowOne = groups[0] * 256;
    expect(firstIndexInRowOne).toBe(lastIndexInRowZero + 1);
    expect(firstIndexInRowOne).toBeLessThan(uniforms[0]!);
  });
});
