/**
 * WS2 — structural consumption checks for the many-light light tree.
 *
 * These guard against the two failure modes the wave plan flags:
 *   1. "computed-but-unconsumed" — the group(3) `lightTree` buffer must actually
 *      be READ by `sampleLightTree` in the composed FULL shader.
 *   2. A stray uniform-pick path left behind in NEE — the full kernel's light
 *      selection must route through `sampleLightTree` (gated), not an
 *      unconditional `floor(rand·lightCount)` compensated by `·f32(lightCount)`.
 *
 * Plus the compile-time gating invariants: the light-tree WGSL + group(3)
 * binding appear ONLY in the FULL trace, never in the LITE trace (lite keeps the
 * uniform pick on its own separate kernel).
 */

import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

describe('WS2 — light-tree WGSL consumption + gating', () => {
  it('FULL trace declares the group(3) light-tree storage buffer EXACTLY ONCE', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(
      /@group\(3\)\s*@binding\(0\)\s*var<storage,\s*read>\s*lightTree\s*:\s*array<f32>/,
    );
    // No duplicate declarations of the traversal symbols (the WGSL is composed
    // once into the full trace — a second copy would be a redefinition error).
    expect((PT_WEBGPU_TRACE_WGSL.match(/fn sampleLightTree\(/g) ?? []).length).toBe(1);
    expect((PT_WEBGPU_TRACE_WGSL.match(/struct LightTreeSample/g) ?? []).length).toBe(1);
    // Exactly one ACTUAL group(3) binding declaration (the prose comment may also
    // mention @group(3), so match the var<storage> decl specifically).
    expect(
      (PT_WEBGPU_TRACE_WGSL.match(/@group\(3\)\s*@binding\(0\)\s*var<storage/g) ?? []).length,
    ).toBe(1);
  });

  it('FULL trace CONSUMES the light-tree buffer (sampleLightTree reads lightTree[])', () => {
    // The traversal function must exist AND be called in the kernel.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleLightTree(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn sampleCanonicalDirectLight(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let selected = sampleLightTree(');
    // And the buffer is actually dereferenced inside the traversal (not dead).
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/lightTree\[base \+ 1u\]/); // power read
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/lightTree\[base \+ 2u\]/); // leftChild read
  });

  it('FULL trace gates the tree pick at COMPILE-consumed runtime flags (not an unconditional path)', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'params.lightTreeEnabled != 0u && params.lightTreeNodeCount > 0u;',
    );
    // The selection reciprocal comes from the tree on the active path and from
    // the uniform fallback otherwise.
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'u32(selected.emitterIndex), 1.0 / selected.pdf,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'return DirectLightSelection(index, f32(lightCount));',
    );
    // EVERY NEE branch compensates the selection OUTSIDE the MIS heuristic by
    // multiplying its contribution by directLightingScale. Ordinary renders use
    // lightSelectInvPdf; inverse summed-direct renders pin the scale to 1.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var lightSelection: DirectLightSelection;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'lightSelection = sampleDistantDirectLight(',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'lightSelection = sampleCanonicalDirectLight(',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/\* directLightingScale;/);
  });

  it('FULL trace accumulates a fully-normalized directLi (no leftover unconditional uniform compensation)', () => {
    // Each branch self-normalizes (·lightSelectInvPdf), so the accumulation is a
    // bare add.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('radiance = radiance + directLi;');
    // The pre-WS2 unconditional `directLi * f32(lightCount)` must be gone.
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('radiance = radiance + directLi * f32(lightCount);');
  });

  it('LITE trace contains NEITHER the light-tree binding NOR the traversal (uniform pick only)', () => {
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('@group(3)');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('var<storage, read> lightTree');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('fn sampleLightTree(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('lightSelectInvPdf');
    // Lite keeps its own uniform pick (separate kernelLite.wgsl) and shares the
    // inverse summed-direct scale switch.
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      'let directLightingScale = select(f32(lightCount), 1.0, sumDirectLighting);',
    );
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('radiance = radiance + directLi * directLightingScale;');
  });
});
