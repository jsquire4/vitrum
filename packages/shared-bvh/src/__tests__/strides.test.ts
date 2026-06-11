/**
 * strides.test.ts — Pin tests for BVH_NODE_FLOATS, VERTEX_STRIDE_F32, MAT4_STRIDE_F32.
 *
 * Goals:
 *   1. Assert constants equal their expected literal values (regression guard).
 *   2. Structural assertion: build a real BVH via buildArrayBvh and verify the
 *      node buffer length is a multiple of BVH_NODE_FLOATS, and that node-count
 *      arithmetic via BVH_NODE_FLOATS matches the single-node case exactly.
 *   3. Verify expandIndicesToStride4 + collapseIndicesToStride3 round-trip.
 */

import { describe, expect, it } from 'vitest';
import {
  BVH_NODE_FLOATS,
  VERTEX_STRIDE_F32,
  MAT4_STRIDE_F32,
  expandIndicesToStride4,
  collapseIndicesToStride3,
} from '../index.js';
import { buildArrayBvh } from '../index.js';

// ── 1. Literal value pins ─────────────────────────────────────────────────────

describe('stride constants — literal value pins', () => {
  it('BVH_NODE_FLOATS === 8', () => {
    expect(BVH_NODE_FLOATS).toBe(8);
  });

  it('VERTEX_STRIDE_F32 === 4', () => {
    expect(VERTEX_STRIDE_F32).toBe(4);
  });

  it('MAT4_STRIDE_F32 === 16', () => {
    expect(MAT4_STRIDE_F32).toBe(16);
  });

  it('BVH_NODE_FLOATS * 4 === 32 (bytes per node)', () => {
    // 8 × 4-byte words = 32 bytes; this is the number that was the source of
    // the merged-mode 16B-vs-32B black-GI bug (fixed 0bedd92).
    expect(BVH_NODE_FLOATS * 4).toBe(32);
  });
});

// ── 2. Structural BVH build assertion ────────────────────────────────────────

describe('BVH_NODE_FLOATS — structural BVH alignment', () => {
  /**
   * Build a minimal stride-4 BVH (single triangle) and verify node layout.
   * buildArrayBvh always emits exactly one node for a single-triangle input.
   */
  function makeSingleTriBvh() {
    // 3 vertices in vec4f layout (VERTEX_STRIDE_F32 = 4 floats/vertex)
    const positions = new Float32Array([
      0, 0, 0, 0,  // v0
      1, 0, 0, 0,  // v1
      0, 1, 0, 0,  // v2
    ]);
    const indices = new Uint32Array([0, 1, 2, 0]); // stride-4: xyz + pad
    const triMaterialIds = new Uint32Array([0]);
    return buildArrayBvh(positions, indices, triMaterialIds);
  }

  it('single-triangle BVH: node buffer length is a multiple of BVH_NODE_FLOATS', () => {
    const bvh = makeSingleTriBvh();
    expect(bvh.bvhNodes.length % BVH_NODE_FLOATS).toBe(0);
  });

  it('single-triangle BVH: node count via BVH_NODE_FLOATS is 1', () => {
    const bvh = makeSingleTriBvh();
    const nodeCount = bvh.bvhNodes.length / BVH_NODE_FLOATS;
    expect(nodeCount).toBe(1);
  });

  it('multi-triangle BVH: node buffer length is always a multiple of BVH_NODE_FLOATS', () => {
    // 6 triangles in a 3×2 grid — forces interior nodes.
    const positions = new Float32Array([
      0,0,0,0, 1,0,0,0, 0,1,0,0,
      1,0,0,0, 2,0,0,0, 1,1,0,0,
      0,1,0,0, 1,1,0,0, 0,2,0,0,
      1,1,0,0, 2,1,0,0, 1,2,0,0,
      0,2,0,0, 1,2,0,0, 0,3,0,0,
      1,2,0,0, 2,2,0,0, 1,3,0,0,
    ]);
    const indices = new Uint32Array([
      0,1,2,0, 3,4,5,0, 6,7,8,0,
      9,10,11,0, 12,13,14,0, 15,16,17,0,
    ]);
    const mats = new Uint32Array(6);
    const bvh = buildArrayBvh(positions, indices, mats);
    expect(bvh.bvhNodes.length % BVH_NODE_FLOATS).toBe(0);
    // Node count must be consistent with the length.
    const nodeCount = bvh.bvhNodes.length / BVH_NODE_FLOATS;
    expect(nodeCount).toBeGreaterThanOrEqual(1);
    // Cross-check: each node is 8 words; node count * 8 = total length.
    expect(nodeCount * BVH_NODE_FLOATS).toBe(bvh.bvhNodes.length);
  });
});

// ── 3. collapseIndicesToStride3 round-trip with expandIndicesToStride4 ───────

describe('collapseIndicesToStride3 + expandIndicesToStride4 round-trip', () => {
  it('collapse is the inverse of expand (no payload)', () => {
    const stride3 = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);  // 3 triangles
    const stride4 = expandIndicesToStride4(stride3);
    const collapsed = collapseIndicesToStride3(stride4);
    expect(Array.from(collapsed)).toEqual(Array.from(stride3));
  });

  it('collapse is the inverse of expand (with payload — payload is discarded)', () => {
    const stride3 = new Uint32Array([10, 20, 30, 40, 50, 60]);  // 2 triangles
    const payload = (t: number) => t + 99;
    const stride4 = expandIndicesToStride4(stride3, payload);
    // stride4.w lanes are non-zero (99, 100); collapse should drop them.
    const collapsed = collapseIndicesToStride3(stride4);
    expect(Array.from(collapsed)).toEqual(Array.from(stride3));
  });

  it('expand then collapse preserves vertex indices for arbitrary input', () => {
    const triCount = 10;
    const stride3 = new Uint32Array(triCount * 3);
    for (let i = 0; i < stride3.length; i++) stride3[i] = i * 7 + 3;  // arbitrary
    const stride4 = expandIndicesToStride4(stride3);
    const collapsed = collapseIndicesToStride3(stride4);
    expect(collapsed.length).toBe(stride3.length);
    for (let i = 0; i < stride3.length; i++) {
      expect(collapsed[i]).toBe(stride3[i]);
    }
  });

  it('collapse of an empty array returns an empty array', () => {
    const empty = new Uint32Array(0);
    expect(collapseIndicesToStride3(empty).length).toBe(0);
  });

  it('expand of an empty array returns an empty array', () => {
    const empty = new Uint32Array(0);
    expect(expandIndicesToStride4(empty).length).toBe(0);
  });

  it('collapseIndicesToStride3 output length is always triCount * 3', () => {
    const stride4 = new Uint32Array([1, 2, 3, 0,  4, 5, 6, 0]);  // 2 triangles
    const out = collapseIndicesToStride3(stride4);
    expect(out.length).toBe(6);
  });
});
