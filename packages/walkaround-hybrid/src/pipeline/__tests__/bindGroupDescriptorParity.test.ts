/**
 * Builder ↔ layout parity test for the descriptor-table BGL families
 * (T9-stepB). Mechanically prevents the lockstep-drift the old dual-table
 * (hand-written layout factory + hand-written builder) invited: it drives the
 * REAL cached-BGL factory and the REAL named builder for every table family
 * through a recording stub device, then asserts their binding-index sets are
 * identical.
 *
 * Because both sides now derive from `bindGroupDescriptors.ts`, this also
 * pins the table itself: if a binding is added/removed in the descriptor, both
 * the layout and the builder change in lockstep and the test stays green; if a
 * future edit reintroduces a divergent hand-written list, the counts mismatch
 * and this fails.
 */

import { describe, expect, it } from 'vitest';

import {
  BIND_GROUP_TABLE,
  bglEntriesFor,
  type BindGroupTableId,
} from '../bindGroupDescriptors.js';
import {
  getFrameBindGroupLayout,
  getSceneBindGroupLayout,
  getUboBindGroupLayout,
  getCompositeBindGroupLayout,
  getSampleBudgetBindGroupLayout,
  getResolveBindGroupLayout,
  getCbPrefillBindGroupLayout,
  getMotionVectorsBindGroupLayout,
  getGTAOBindGroupLayout,
  getGTAOUpsampleBindGroupLayout,
  getTemporalGiBindGroupLayout,
  getSpatialGiBindGroupLayout,
  getIndirectCombineBindGroupLayout,
  getIndirectTemporalAccumBindGroupLayout,
  getTransparentOitBindGroupLayout,
  type BGLCache,
} from '../bindGroupLayouts.js';
import {
  buildFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
  buildCompositeBindGroup,
  buildSampleBudgetBindGroup,
  buildResolveBindGroup,
  buildCbPrefillBindGroup,
  buildMotionVectorsBindGroup,
  buildGTAOBindGroup,
  buildGTAOUpsampleBindGroup,
  buildTemporalGiBindGroup,
  buildSpatialGiBindGroup,
  buildIndirectTemporalAccumBindGroup,
  buildIndirectCombineBindGroup,
  buildTransparentOitBindGroup,
} from '../bindGroupBuilders.js';

// ── Recording stub device ────────────────────────────────────────────────────

interface CapturedLayout {
  readonly label: string;
  readonly bindings: number[];
}
interface CapturedBindGroup {
  readonly label: string;
  readonly bindings: number[];
}

function makeStubDevice() {
  const layouts: CapturedLayout[] = [];
  const bindGroups: CapturedBindGroup[] = [];
  const device = {
    createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
      const entries = [...(desc.entries)];
      layouts.push({
        label: desc.label ?? '',
        bindings: entries.map((e) => e.binding),
      });
      // Return a tagged token so the builder's layout arg is non-null.
      return { __bgl: desc.label } as unknown as GPUBindGroupLayout;
    },
    createBindGroup(desc: GPUBindGroupDescriptor): GPUBindGroup {
      const entries = [...(desc.entries)];
      bindGroups.push({
        label: desc.label ?? '',
        bindings: entries.map((e) => e.binding),
      });
      return { __bg: desc.label } as unknown as GPUBindGroup;
    },
  } as unknown as GPUDevice;
  return { device, layouts, bindGroups };
}

/** N distinct dummy GPUBindingResources (the parity test ignores resource
 *  shape — it only checks binding-index correspondence). Retained for bind-group parity tests. */
function _dummyResources(n: number): GPUBindingResource[] {
  return Array.from({ length: n }, (_, i) => ({ buffer: { __i: i } } as unknown as GPUBindingResource));
}
const view = {} as GPUTextureView;
const buf = {} as GPUBuffer;
const sampler = {} as GPUSampler;

/**
 * Drives the REAL named builder for each table family with the correct number
 * of arguments. Returns the captured bind-group binding indices.
 */
const BUILDER_DRIVERS: Record<BindGroupTableId, (d: GPUDevice, c: BGLCache) => GPUBindGroup> = {
    frame: (d, c) => buildFrameBindGroup(d, c, {
      reservoirCurrentBuffer: buf, reservoirPreviousBuffer: buf, reservoirSpatialBuffer: buf,
    hdrColorTexture: { createView: () => view } as unknown as GPUTexture,
    nearestSampler: sampler,
    gNormalDepthTexture: { createView: () => view } as unknown as GPUTexture,
    reservoirGiCurrentBuffer: buf,
    hdrIndirectTexture: { createView: () => view } as unknown as GPUTexture,
    hdrTotalTexture: { createView: () => view } as unknown as GPUTexture,
    albedoTexture: { createView: () => view } as unknown as GPUTexture,
    svgfCurrentObjectIdTexture: { createView: () => view } as unknown as GPUTexture,
    }),
    scene: (d, c) => buildSceneBindGroup(d, c, {
      sceneStorageArenaBuffers: [buf, buf, buf],
      // WS1 — beer is a uint texture (binding 5).
      bvhBeerTextureView: view, bvhEmissiveTextureView: view,
      bvhRoughMetalTextureView: view,
      materialTextureAtlasView: view, baseColorMapMetaTextureView: view,
      bvhTangentTextureView: view,
      bvhVertexColorTextureView: view,
      analyticLightsTextureView: view,
    // B3 — directional IBL env resources (bindings 15-19).
    envMapTextureView: view, envMarginalTextureView: view, envConditionalTextureView: view,
    envPdfTextureView: view,
    envParamsBuffer: buf,
  }),
  ubo: (d, c) => buildUboBindGroup(d, c, buf, view, view),
  composite: (d, c) => buildCompositeBindGroup(d, c, view, buf),
  sampleBudget: (d, c) => buildSampleBudgetBindGroup(d, c, view, view, buf, buf),
  resolve: (d, c) => buildResolveBindGroup(d, c, buf, view, view, view, view),
  cbPrefill: (d, c) => buildCbPrefillBindGroup(d, c, buf, view, view, view, view),
  motionVectors: (d, c) => buildMotionVectorsBindGroup(d, c, view, view, buf),
  gtao: (d, c) => buildGTAOBindGroup(d, c, view, view, buf, view),
  gtaoUpsample: (d, c) => buildGTAOUpsampleBindGroup(d, c, view, view, view, buf),
  temporalGi: (d, c) => buildTemporalGiBindGroup(d, c, buf, buf, buf),
  spatialGi: (d, c) => buildSpatialGiBindGroup(d, c, buf, buf, buf, 'spatial-gi-bg-1'),
  indirectTemporalAccum: (d, c) => buildIndirectTemporalAccumBindGroup(
    d, c, view, view, view, view,
  ),
  indirectCombine: (d, c) => buildIndirectCombineBindGroup(d, c, view, view, view, view),
  transparentOit: (d, c) => buildTransparentOitBindGroup(
    d, c, view, view, sampler, buf, buf, buf, view, view,
  ),
};

const LAYOUT_FACTORIES: Record<BindGroupTableId, (d: GPUDevice, c: BGLCache) => GPUBindGroupLayout> = {
  frame: getFrameBindGroupLayout,
  scene: getSceneBindGroupLayout,
  ubo: getUboBindGroupLayout,
  composite: getCompositeBindGroupLayout,
  sampleBudget: getSampleBudgetBindGroupLayout,
  resolve: getResolveBindGroupLayout,
  cbPrefill: getCbPrefillBindGroupLayout,
  motionVectors: getMotionVectorsBindGroupLayout,
  gtao: getGTAOBindGroupLayout,
  gtaoUpsample: getGTAOUpsampleBindGroupLayout,
  temporalGi: getTemporalGiBindGroupLayout,
  spatialGi: getSpatialGiBindGroupLayout,
  indirectTemporalAccum: getIndirectTemporalAccumBindGroupLayout,
  indirectCombine: getIndirectCombineBindGroupLayout,
  transparentOit: getTransparentOitBindGroupLayout,
};

describe('bind-group descriptor parity (T9-stepB)', () => {
  it('every table family is covered by both a layout factory and a builder driver', () => {
    for (const entry of BIND_GROUP_TABLE) {
      expect(LAYOUT_FACTORIES[entry.id], `layout factory for '${entry.id}'`).toBeDefined();
      expect(BUILDER_DRIVERS[entry.id], `builder driver for '${entry.id}'`).toBeDefined();
    }
  });

  it.each(BIND_GROUP_TABLE.map((e) => e.id))(
    "'%s': layout entry count + binding indices match the builder's",
    (id) => {
      const tableBindings = bglEntriesFor(id).map((e) => e.binding);

      // Layout side — real cached-BGL factory.
      const { device: ldev, layouts } = makeStubDevice();
      LAYOUT_FACTORIES[id](ldev, {});
      expect(layouts).toHaveLength(1);
      const layoutBindings = layouts[0]!.bindings;

      // Builder side — real named builder.
      const { device: bdev, bindGroups } = makeStubDevice();
      BUILDER_DRIVERS[id](bdev, {});
      expect(bindGroups).toHaveLength(1);
      const builderBindings = bindGroups[0]!.bindings;

      // Core assertion: same count …
      expect(builderBindings.length).toBe(layoutBindings.length);
      // … and identical binding indices in the same order.
      expect(builderBindings).toEqual(layoutBindings);
      // … and both agree with the descriptor table.
      expect(layoutBindings).toEqual(tableBindings);
    },
  );

  it('table binding indices are strictly ascending and duplicate-free', () => {
    for (const entry of BIND_GROUP_TABLE) {
      const bindings = entry.entries.map((e) => e.binding);
      expect(bindings.every((binding) => Number.isInteger(binding) && binding >= 0))
        .toBe(true);
      expect(bindings, `family '${entry.id}'`).toEqual(
        [...new Set(bindings)].sort((a, b) => a - b),
      );
    }
  });

  it('every inert/placeholder binding retains its rationale note', () => {
    // The shade-only outputs (10/12/13/14/15) and the ubo adaptive-tier slot (2)
    // are load-bearing-but-inert in peer passes; their notes are the only record
    // of why they must stay bound. Guard them.
    const frame = BIND_GROUP_TABLE.find((e) => e.id === 'frame')!;
    for (const b of [10, 12, 13, 14, 15]) {
      const e = frame.entries.find((x) => x.binding === b)!;
      expect(e.note, `frame binding ${b} note`).toBeTruthy();
    }
    const ubo = BIND_GROUP_TABLE.find((e) => e.id === 'ubo')!;
    expect(ubo.entries.find((x) => x.binding === 2)!.note).toMatch(/inert|risGi/i);
  });
});
