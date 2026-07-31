import { describe, expect, it } from 'vitest';
import {
  BMFR_BLOCK_FIT_SIZE_BYTES,
  BMFR_WGSL,
  householderLeastSquares,
} from '../src/index.js';

describe('BMFR direct QR implementation', () => {
  it('solves a highly correlated rectangular system without normal equations', () => {
    const rowCount = 64;
    const columnCount = 2;
    const matrix = new Float64Array(rowCount * columnCount);
    const rhs = new Float64Array(rowCount);
    const epsilon = 1e-7;
    for (let row = 0; row < rowCount; row += 1) {
      const a0 = 1;
      const a1 = 1 + epsilon * (row - (rowCount - 1) / 2);
      matrix[row * columnCount] = a0;
      matrix[row * columnCount + 1] = a1;
      rhs[row] = 3 * a0 - 2 * a1;
    }

    const solution = householderLeastSquares(
      matrix,
      rhs,
      rowCount,
      columnCount,
    );
    expect(solution[0]).toBeCloseTo(3, 3);
    expect(solution[1]).toBeCloseTo(-2, 3);
  });

  it('uses chunked Householder QR on A and augmented regularisation rows', () => {
    expect(BMFR_WGSL).toContain('fn qrReduceChunk(');
    expect(BMFR_WGSL).toContain('matrix[targetRow * F + col] = features[col]');
    expect(BMFR_WGSL).toContain(
      'let regularisationRoot = sqrt(max(0.0, bmfr_ubo.regularisation))',
    );
    expect(BMFR_WGSL).not.toContain('AᵀA');
    expect(BMFR_WGSL).not.toContain('normal matrix');
  });
});

describe('BMFR overlap and signed-depth semantics', () => {
  it('stores private block fits and resolves overlaps in a second entry point', () => {
    expect(BMFR_BLOCK_FIT_SIZE_BYTES).toBe(160);
    expect(BMFR_WGSL).toContain(
      '@group(0) @binding(4) var<storage, read_write> bmfr_blockFits',
    );
    expect(BMFR_WGSL).toContain('fn bmfrResolve(');
    expect(BMFR_WGSL).toContain('reconstructionSum / f32(contributionCount)');

    const fitSource = BMFR_WGSL.slice(0, BMFR_WGSL.indexOf('fn bmfrResolve('));
    expect(fitSource).not.toContain('textureStore(bmfr_out');
  });

  it('treats negative signed depth as valid transmissive geometry', () => {
    expect(BMFR_WGSL).toContain('let depth = abs(raw.w)');
    expect(BMFR_WGSL).toContain(
      'return vec4f(f32(coord.x), f32(coord.y), depth, depth)',
    );
    expect(BMFR_WGSL).toContain('if (position.w <= 0.0)');
  });

  it('preserves every validated positive world-position scale without an absolute floor', () => {
    expect(BMFR_WGSL.match(/1\.0 \/ bmfr_ubo\.positionScale/g)).toHaveLength(2);
    expect(BMFR_WGSL).not.toContain(
      'max(bmfr_ubo.positionScale, 1e-4)',
    );
  });

  it('bounds every rgba16float resolve write to a finite half domain', () => {
    expect(BMFR_WGSL).toContain('fn finiteHalfChannel(value: f32)');
    expect(BMFR_WGSL).toContain('return clamp(value, -65504.0, 65504.0)');
    expect(BMFR_WGSL.match(/vec4f\(finiteHalfRgb\(/g)).toHaveLength(2);
  });
});
