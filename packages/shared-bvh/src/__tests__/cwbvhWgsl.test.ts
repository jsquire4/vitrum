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
    expect(CWBVH_INTERSECT_WGSL).toContain('struct CwbvhAnyHitResult');
    expect(CWBVH_INTERSECT_WGSL).toContain('struct CwbvhChildMeta');
  });

  it('exposes a helper-free core module for renderer compositions', () => {
    expect(CWBVH_INTERSECT_CORE_WGSL).toContain('struct CwbvhIntersectionResult');
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain('fn safeInvDir(d: vec3f) -> vec3f');
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain('fn mollerTrumboreCore(');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn safeInvDir(d: vec3f) -> vec3f');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn mollerTrumboreCore(');
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain('ptr<storage');
    expect(CWBVH_INTERSECT_WGSL).toContain(
      'fn cwbvhLoadNodeBounds(index: u32) -> CwbvhNodeBounds',
    );
  });

  it('supports explicit forest roots while preserving root-zero wrappers', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('rootNode: u32');
    expect(CWBVH_INTERSECT_WGSL).toContain('if (nodeCount == 0u)');
    expect(CWBVH_INTERSECT_WGSL).toContain('if (rootNode >= nodeCount)');
    expect(CWBVH_INTERSECT_WGSL).toContain('stack[stackPtr] = rootNode;');
    expect(CWBVH_INTERSECT_WGSL).toContain('return cwbvhIntersectFirstHitFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toContain('return cwbvhIntersectAnyFromRoot(');
    expect(CWBVH_INTERSECT_WGSL).toMatch(/nodeCount,\s+0u,\s+skipGlass,/);
  });

  it('keeps COMPLETE, STACK_OVERFLOW, and INVALID_LAYOUT distinct for any-hit callers', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('const CWBVH_STATUS_COMPLETE: u32 = 0u;');
    expect(CWBVH_INTERSECT_WGSL).toContain('const CWBVH_STATUS_STACK_OVERFLOW: u32 = 1u;');
    expect(CWBVH_INTERSECT_WGSL).toContain('const CWBVH_STATUS_INVALID_LAYOUT: u32 = 2u;');
    expect(CWBVH_INTERSECT_WGSL).toContain(') -> CwbvhAnyHitResult');
    expect(CWBVH_INTERSECT_WGSL).toContain(
      'return cwbvhAnyHitResult(CWBVH_STATUS_STACK_OVERFLOW, false);',
    );
    expect(CWBVH_INTERSECT_WGSL).toContain(
      'return cwbvhAnyHitResult(CWBVH_STATUS_INVALID_LAYOUT, false);',
    );
    expect(CWBVH_INTERSECT_WGSL).toContain('if (tri.hit && tri.t > triEps && tri.t < tMax)');
  });

  it('rejects corrupt live children while leaving zeroed padding outside childCount unread', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('fn cwbvhBoundsAreValid(');
    expect(CWBVH_INTERSECT_WGSL).toContain('all(abs(bmin) <= finiteMax)');
    expect(CWBVH_INTERSECT_WGSL).toContain('for (var slot = 0u; slot < count; slot = slot + 1u)');
    expect(CWBVH_INTERSECT_WGSL).not.toMatch(
      /if \(childInfo\.kind == CWBVH_CHILD_EMPTY\) \{\s*continue;/,
    );
    expect(CWBVH_INTERSECT_WGSL.match(/childInfo\.triCount == 0u/g)).toHaveLength(2);
    expect(CWBVH_INTERSECT_WGSL.match(/CWBVH_STATUS_INVALID_LAYOUT/g)?.length ?? 0).toBeGreaterThan(10);
  });

  it('validates storage capacities without overflow-prone node-count products', () => {
    expect(CWBVH_INTERSECT_CORE_WGSL.match(
      /nodeCount > cwbvhChildBoundsWordCount\(\) \/ \(CWBVH_CHILDREN \* CWBVH_CHILD_BOUNDS_PACKED_U32\)/g,
    )).toHaveLength(2);
    expect(CWBVH_INTERSECT_CORE_WGSL.match(
      /nodeCount > cwbvhChildMetaCount\(\) \/ CWBVH_CHILDREN/g,
    )).toHaveLength(2);
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain(
      'cwbvhChildBoundsWordCount() < nodeCount *',
    );
    expect(CWBVH_INTERSECT_CORE_WGSL).not.toContain(
      'cwbvhChildMetaCount() < nodeCount *',
    );
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
    expect(CWBVH_INTERSECT_WGSL).toContain('let w0 = cwbvhLoadChildBoundsWord(base + 0u);');
    expect(CWBVH_INTERSECT_WGSL).toContain('let w1 = cwbvhLoadChildBoundsWord(base + 1u);');
    expect(CWBVH_INTERSECT_WGSL).toContain('let w2 = cwbvhLoadChildBoundsWord(base + 2u);');
    expect(CWBVH_INTERSECT_WGSL).toContain('65535.0');
  });

  it('uses shared ray math and preserves glass-skip semantics', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('fn safeInvDir(d: vec3f) -> vec3f');
    expect(CWBVH_INTERSECT_WGSL).toContain('fn mollerTrumboreCore(');
    expect(CWBVH_INTERSECT_WGSL).toContain(
      'fn cwbvhPackedMaterialHasTransmission(packedMaterial: u32) -> bool',
    );
    expect(CWBVH_INTERSECT_WGSL).toContain(
      'if (cwbvhPackedMaterialHasTransmission(idxEntry.w))',
    );
  });

  it('decodes position.w UV payloads as packed f16 pairs', () => {
    expect(CWBVH_INTERSECT_WGSL).toContain('unpack2x16float(bitcast<u32>(pa4.w))');
    expect(CWBVH_INTERSECT_WGSL).not.toContain('unpack2x16unorm(bitcast<u32>(pa4.w))');
  });
});
