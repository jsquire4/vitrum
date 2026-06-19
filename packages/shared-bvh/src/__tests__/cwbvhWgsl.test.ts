import { describe, expect, it } from 'vitest';
import {
  CWBVH_CHILDREN,
  CWBVH_CHILD_BOUNDS_PACKED_U32,
  CWBVH_INTERSECT_CORE_WGSL,
  CWBVH_INTERSECT_STACK_DEPTH,
  CWBVH_INTERSECT_WGSL,
} from '../index.js';

describe('CWBVH_INTERSECT_WGSL', () => {
  it('exports closest-hit and any-hit traversal entry points', () => {
    expect(CWBVH_INTERSECT_CORE_WGSL).toContain('fn cwbvhIntersectFirstHitRangeFromRoot(');
    expect(CWBVH_INTERSECT_CORE_WGSL).toContain('fn cwbvhIntersectFirstHitFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhIntersectFirstHitRangeFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhIntersectFirstHitFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhIntersectFirstHit(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhIntersectAnyFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhIntersectAny(');
    expect(CWBVH_INTERSECT_WGSL).toContain('struct CwbvhIntersectionResult');
    expect(CWBVH_INTERSECT_WGSL).toContain('struct CwbvhChildMeta');
  });

  it('exposes a helper-free core module for renderer compositions', () => {
    expect(CWBVH_INTERSECT_CORE_WGSL).toContain('struct CwbvhIntersectionResult');
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain('fn safeInvDir(d: vec3f) -> vec3f');
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain('fn mollerTrumboreCore(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn safeInvDir(d: vec3f) -> vec3f');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn mollerTrumboreCore(');
  });

  it('supports explicit forest roots while preserving root-zero wrappers', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('rootNode: u32');
    expect(CWBVH_INTERSECT_WGSL).toContain('nodeCount == 0u || rootNode >= nodeCount');
    expect(CWBVH_INTERSECT_WGSL).toContain('stack[stackPtr] = rootNode;');
    expect(CWBVH_INTERSECT_WGSL).toContain('return cwbvhIntersectFirstHitFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toContain('return cwbvhIntersectAnyFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toMatch(/nodeCount,\s+0u,\s+skipGlass,/);
  });

  it('pins WGSL constants to the TypeScript CWBVH layout constants', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain(`const CWBVH_CHILDREN: u32 = ${CWBVH_CHILDREN}u;`);
    expect(CWBVH_INTERSECT_WGSL).toContain(
      `const CWBVH_CHILD_BOUNDS_PACKED_U32: u32 = ${CWBVH_CHILD_BOUNDS_PACKED_U32}u;`,
    );
    expect(CWBVH_INTERSECT_WGSL).toContain(
      `const CWBVH_INTERSECT_STACK_DEPTH: u32 = ${CWBVH_INTERSECT_STACK_DEPTH}u;`,
    );
  });

  it('decodes the six-u16 CPU child bounds from three packed u32 words', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhUnpackLo16(word: u32) -> u32');
    expect(CWBVH_INTERSECT_WGSL).toContain('return word & 0xffffu;');
    expect(CWBVH_INTERSECT_WGSL).toContain('return (word >> 16u) & 0xffffu;');
    expect(CWBVH_INTERSECT_WGSL).toContain('let w0 = (*cwbvhChildBoundsPacked)[base + 0u];');
    expect(CWBVH_INTERSECT_WGSL).toContain('let w1 = (*cwbvhChildBoundsPacked)[base + 1u];');
    expect(CWBVH_INTERSECT_WGSL).toContain('let w2 = (*cwbvhChildBoundsPacked)[base + 2u];');
    expect(CWBVH_INTERSECT_WGSL).toContain('65535.0');
  });

  it('uses shared ray math and preserves glass-skip semantics', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('fn safeInvDir(d: vec3f) -> vec3f');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn mollerTrumboreCore(');
    expect(CWBVH_INTERSECT_WGSL).toContain('let trans4 = (idxEntry.w >> 4u) & 0xFu;');
    expect(CWBVH_INTERSECT_WGSL).toContain('if (trans4 > 4u)');
  });
});
