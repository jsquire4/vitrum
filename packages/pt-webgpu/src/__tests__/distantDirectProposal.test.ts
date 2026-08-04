import { describe, expect, it } from 'vitest';
import { REPRESENTED_PROPOSAL_BUCKET_COUNT } from '@vitrum/shared-samplers';
import { applyDistantDirectProposalPmf } from '../scene/distantDirectProposal.js';
import { PT_WEBGPU_MEDIUM_NEE_WGSL } from '../wgsl/pathTrace/mediumNee.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL } from '../wgsl/pathTrace/connectLite.wgsl.js';

function directionalRecords(weights: readonly number[]): Float32Array {
  const records = new Float32Array(weights.length * 8);
  for (let i = 0; i < weights.length; i += 1) {
    const base = i * 8;
    records.set([0, 1, 0, 0, weights[i]!, weights[i]!, weights[i]!, 0], base);
  }
  return records;
}

describe('represented WebGPU distant-direct proposal', () => {
  it('retains adversarial positive directional support and publishes the exact environment PMF', () => {
    const directional = directionalRecords([2 ** -30, 2 ** -30]);
    const environmentPmf = applyDistantDirectProposalPmf(
      directional,
      2,
      1,
      true,
    );
    const pmfs = [directional[7]!, directional[15]!, environmentPmf];

    expect(pmfs[0]).toBe(1 / REPRESENTED_PROPOSAL_BUCKET_COUNT);
    expect(pmfs[1]).toBe(1 / REPRESENTED_PROPOSAL_BUCKET_COUNT);
    expect(pmfs[2]).toBe(
      (REPRESENTED_PROPOSAL_BUCKET_COUNT - 2) /
        REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
    expect(Math.fround(pmfs[0]! + pmfs[1]! + pmfs[2]!)).toBe(1);
  });

  it('publishes an empty proposal for physically black distant sources', () => {
    const directional = directionalRecords([0, 0]);
    expect(applyDistantDirectProposalPmf(directional, 2, 0, true)).toBe(0);
    expect(directional[7]).toBe(0);
    expect(directional[15]).toBe(0);
  });

  it('rejects mismatched directional storage instead of publishing a partial proposal', () => {
    expect(() =>
      applyDistantDirectProposalPmf(new Float32Array(7), 1, 0, false),
    ).toThrow(/expected 8/);
  });

  it('assigns exact 24-bit CDF endpoints to the next positive interval', () => {
    const bucketCount = REPRESENTED_PROPOSAL_BUCKET_COUNT;
    const pmfs = [1 / bucketCount, 1 / bucketCount, (bucketCount - 2) / bucketCount];
    const select = (bucket: number): number => {
      const xi = bucket / bucketCount;
      let cumulative = 0;
      for (let i = 0; i < pmfs.length; i += 1) {
        cumulative = Math.fround(cumulative + pmfs[i]!);
        if (xi < cumulative) return i;
      }
      return pmfs.length - 1;
    };

    expect(select(0)).toBe(0);
    expect(select(1)).toBe(1); // xi == CDF[1], never the prior interval
    expect(select(2)).toBe(2); // xi == CDF[2]
    expect(select(bucketCount - 1)).toBe(2);
    expect(PT_WEBGPU_MEDIUM_NEE_WGSL).toContain(
      'if (pmf > 0.0 && pickTarget < cumulative)',
    );
  });

  it('uses a last-CDF-not-greater binary-search invariant for full and lite environments', () => {
    // These searches keep `lo` at the final endpoint <= xi, then select the
    // interval beginning at lo. Thus <= implements the first-CDF-strictly-
    // greater-than-xi rule,
    // including equality at every represented 24-bit boundary.
    expect(PT_WEBGPU_PATH_TRACE_CONNECT_WGSL).toContain(
      'if (environmentMapCdf[mid] <= xi) { lo = mid; } else { hi = mid; }',
    );
    expect(PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL).toContain(
      'if (cdfMid <= xi) { lo = mid; } else { hi = mid; }',
    );
  });
});
