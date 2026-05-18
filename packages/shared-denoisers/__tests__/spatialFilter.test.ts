/**
 * spatialFilter.test.ts — Defensive tests for Sprint 6 WGSL spatial filter.
 *
 * These tests do NOT execute WGSL on a GPU.  Instead they verify:
 *  1. The exported constant is a non-empty string (guards against module
 *     truncation or empty-export bugs).
 *  2. The string contains the expected @compute entry-point function.
 *  3. All expected uniform/texture binding declarations are present.
 *  4. The hexagonal offset array has exactly 37 × 2 = 74 entries.
 *
 * GPU-side execution is verified at integration time when a WebGPU consumer
 * dispatches the kernel and checks pixel output.
 */

import { describe, it, expect } from 'vitest';
import { SPATIAL_FILTER_WGSL } from '../src/wgsl/spatialFilter.wgsl.js';

describe('SPATIAL_FILTER_WGSL', () => {
  it('is a non-empty string', () => {
    expect(typeof SPATIAL_FILTER_WGSL).toBe('string');
    expect(SPATIAL_FILTER_WGSL.length).toBeGreaterThan(0);
  });

  it('declares the @compute entry-point "spatialFilterMain"', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('fn spatialFilterMain(');
  });

  it('declares the @compute attribute with 16×16 workgroup', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('@compute @workgroup_size(16, 16, 1)');
  });

  it('declares inputColor binding', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('var inputColor');
  });

  it('declares outputColor storage binding', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('var outputColor');
  });

  it('declares gbufferNormal binding', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('var gbufferNormal');
  });

  it('declares gbufferDepth binding', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('var gbufferDepth');
  });

  it('declares SpatialFilterUBO uniform struct', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('struct SpatialFilterUBO');
    expect(SPATIAL_FILTER_WGSL).toContain('sigmaColor');
    expect(SPATIAL_FILTER_WGSL).toContain('sigmaNormal');
    expect(SPATIAL_FILTER_WGSL).toContain('sigmaDepth');
  });

  it('declares NUM_TAPS = 37', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('NUM_TAPS: i32 = 37');
  });

  it('OFFSETS array has 74 entries (37 taps × 2 coordinates)', () => {
    // Extract the OFFSETS array content from the WGSL string.
    // We look for `array<i32, 74>` to confirm the declared size.
    expect(SPATIAL_FILTER_WGSL).toContain('array<i32, 74>');
  });

  it('uses textureStore to write the output', () => {
    expect(SPATIAL_FILTER_WGSL).toContain('textureStore(outputColor');
  });

  it('implements bilateral edge-stopping (color + normal + depth weights)', () => {
    // These identifiers confirm the three edge-stopping weight components.
    expect(SPATIAL_FILTER_WGSL).toContain('wc'); // color weight
    expect(SPATIAL_FILTER_WGSL).toContain('wn'); // normal weight
    expect(SPATIAL_FILTER_WGSL).toContain('wz'); // depth weight
  });
});
