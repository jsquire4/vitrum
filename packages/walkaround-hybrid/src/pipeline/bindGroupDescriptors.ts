/**
 * Bind-group descriptor table — single source of truth for the *uniform*
 * BGL families (T9-stepB, "staged-both" design).
 *
 * # Why this file exists
 *
 * Before this table, every bind group lived as TWO hand-written lists that had
 * to stay in lockstep:
 *   1. a layout factory in `bindGroupLayouts.ts` (binding → entry type), and
 *   2. a builder in `bindGroupBuilders.ts` (binding → concrete resource).
 *
 * Drift between the two is silent until a `createBindGroup` validation error
 * at runtime. This descriptor table captures each binding's *kind* once and
 * generates BOTH sides from it:
 *   - {@link bglEntriesFor} → the `GPUBindGroupLayoutEntry[]` the cached BGL
 *     factory in `bindGroupLayouts.ts` consumes.
 *   - {@link buildBindGroupFromTable} → a generic `buildBindGroup(id,
 *     resources[])` used by the named builders in `bindGroupBuilders.ts`.
 *
 * The companion parity test (`__tests__/bindGroupDescriptorParity.test.ts`)
 * asserts the generated entry count + binding indices match the layout, so
 * the lockstep-drift the dual-table invited is now mechanically impossible.
 *
 * # Scope — what's in the table vs escape-hatched
 *
 * Only the families whose builder is a *straight entries-list* (one resource
 * per binding, no per-call mutation) live here:
 *   frame · scene · ubo · gtao · gtaoUpsample · temporalGi · spatialGi ·
 *   indirectCombine · indirectTemporalAccum · motionVectors · resolve ·
 *   transparentOit · sampleBudget · composite
 *
 * The three NON-uniform builders stay HAND-WRITTEN in `bindGroupBuilders.ts`
 * (they do work no descriptor can express):
 *   - `buildAtrousBindGroup` / `buildAccumBindGroup` — lazily create + writeBuffer
 *     a per-builder UBO (`if (!uboRef.buf)`),
 *   - `buildHybridLayersBindGroup` — null→placeholder texture fallback.
 * The `layout:'auto'` denoiser / PPG pipelines also stay outside this system
 * (their layout is owned by the shader, not a cached BGL).
 *
 * # CRITICAL — dead-binding rationale is preserved
 *
 * Several frame/ubo bindings are inert placeholders that MUST stay bound for
 * layout compatibility across pipelines that don't reference them (e.g. the
 * frame G-buffer slots 0-4, and the shade-only output slots 10/12/13/14/15).
 * Their rationale lives in each entry's `note` field below — do NOT prune any
 * inert binding; dropping it from the layout breaks every pipeline that shares
 * the BGL.
 */

/**
 * Discriminated tag for a single binding's GPU resource type. The string
 * encodes both the WebGPU layout-entry shape and (for storage textures) the
 * format, so one tag round-trips to a `GPUBindGroupLayoutEntry`.
 */
type BindingKind =
  | 'storage-ro'                  // buffer: { type: 'read-only-storage' }
  | 'storage-rw'                  // buffer: { type: 'storage' }
  | 'uniform'                     // buffer: { type: 'uniform' }
  | 'tex'                         // texture: { sampleType: 'unfilterable-float' }
  | 'tex-array'                   // texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' }
  | 'tex:uint'                    // texture: { sampleType: 'uint' }
  | 'sampler:nf'                  // sampler: { type: 'non-filtering' }
  | 'storage-tex:rgba16float'     // storageTexture: write-only rgba16float
  | 'storage-tex:rgba32float'     // storageTexture: write-only rgba32float
  | 'storage-tex:r32uint';        // storageTexture: write-only r32uint

interface BindingDescriptor {
  readonly binding: number;
  readonly kind: BindingKind;
  /** Optional rationale — load-bearing for inert / placeholder bindings. */
  readonly note?: string;
}

/** Identifier for a table-backed BGL family. Must match a `BGLCache` key. */
export type BindGroupTableId =
  | 'frame'
  | 'scene'
  | 'ubo'
  | 'gtao'
  | 'gtaoUpsample'
  | 'temporalGi'
  | 'spatialGi'
  | 'indirectCombine'
  | 'indirectTemporalAccum'
  | 'transparentOit'
  | 'motionVectors'
  | 'resolve'
  | 'sampleBudget'
  | 'composite'
  | 'cbPrefill';

export interface BindGroupTableEntry {
  readonly id: BindGroupTableId;
  /** GPU debug label suffix (`<label>-bgl` / `<label>-bg`). */
  readonly label: string;
  /** Shader stage visibility for every binding in this family. */
  readonly visibility: 'compute' | 'fragment';
  readonly entries: readonly BindingDescriptor[];
}

// ── The table ────────────────────────────────────────────────────────────────

export const BIND_GROUP_TABLE: readonly BindGroupTableEntry[] = [
  {
    id: 'frame',
    label: 'frame',
    visibility: 'compute',
    entries: [
      // Slots 0-4 — G-buffer textures. shade.wgsl declares ALL FIVE as
      // gDepth/gNormal/gAlbedo/gRough/motionVec at @group(0)@binding(0..4).
      // In primary-ray-cast mode no pre-pass writes them, so they are bound
      // to a shared 1×1 placeholder texture — but the BGL entry MUST remain
      // because shade compiles them and WebGPU validates the BGL against the
      // pipeline's shader interface. ris/temporal/spatial declare a subset of
      // the frame BGL and never reference bindings 0-4; they are inert for
      // those passes but bound for layout compat so the same BGL is shared.
      //
      // IMPORTANT: Do NOT remove these entries. shade.wgsl reads gDepth (for
      // primary-hit distance) and gNormal (for shade normals) in future
      // G-buffer-fill mode, and gRough/gAlbedo/motionVec are reserved for
      // the same upgrade. The 1×1 placeholder bound today makes the shader
      // a no-op for those reads without requiring a separate pipeline variant.
      { binding: 0, kind: 'tex', note: 'gDepth — shade.wgsl @binding(0); 1×1 placeholder in primary-ray-cast mode' },
      { binding: 1, kind: 'tex', note: 'gNormal — shade.wgsl @binding(1); 1×1 placeholder in primary-ray-cast mode' },
      { binding: 2, kind: 'tex', note: 'gAlbedo — shade.wgsl @binding(2); 1×1 placeholder in primary-ray-cast mode' },
      { binding: 3, kind: 'tex', note: 'gRough — shade.wgsl @binding(3); 1×1 placeholder in primary-ray-cast mode' },
      { binding: 4, kind: 'tex', note: 'motionVec — shade.wgsl @binding(4); 1×1 placeholder in primary-ray-cast mode' },
      { binding: 5, kind: 'storage-rw', note: 'reservoirCurrent' },
      { binding: 6, kind: 'storage-ro', note: 'reservoirPrevious' },
      { binding: 7, kind: 'storage-rw', note: 'reservoirSpatial' },
      { binding: 8, kind: 'storage-tex:rgba16float', note: 'hdrColor (write)' },
      { binding: 9, kind: 'sampler:nf', note: 'nearest sampler' },
      // Slot 10 — gNormalDepth. Only shade writes it (normal.xyz + primary-hit
      // distance.w); ris/temporal/spatial declare it via the BGL but never
      // reference the symbol. Inert for them; bound to the same texture in
      // every dispatch for layout compat.
      { binding: 10, kind: 'storage-tex:rgba16float', note: 'gNormalDepth (shade-write only; inert for ris/temporal/spatial)' },
      // Slot 11 — Sprint 16 half-res ReSTIR-GI reservoir. risGi writes, shade
      // reads; other DI passes declare it via the BGL but never reference it.
      { binding: 11, kind: 'storage-rw', note: 'reservoirGiCurrent (risGi-write / shade-read; inert for DI passes)' },
      // Slot 12 — Sprint 18 indirect-channel HDR output. Only shade writes it;
      // bound to all frame-BGL pipelines for layout compat.
      { binding: 12, kind: 'storage-tex:rgba16float', note: 'hdrIndirect (shade-write only; inert for ris/temporal/spatial/risGi)' },
      // Slot 13 — Sprint 18 follow-up total-radiance output (welford input).
      // Only shade writes it; inert/placeholder for the other frame-BGL passes.
      { binding: 13, kind: 'storage-tex:rgba16float', note: 'hdrTotal (shade-write only; welford reads it)' },
      // Slot 14 — Item 24 albedo demodulation (Schied 2017 §4.1). shade writes
      // visible-point albedo; indirectCombine reads it. Inert for ris/temporal/
      // spatial/risGi but must stay bound for frame-BGL layout compat.
      { binding: 14, kind: 'storage-tex:rgba16float', note: 'albedo (shade-write only; indirectCombine reads it)' },
      // Slot 15 — SVGF-real object ID. shade writes the current frame's stable
      // visible object/primitive/triangle ID; SVGF reprojection reads it via the
      // denoiser-private bind group. Inert for RIS/temporal/spatial/risGi.
      { binding: 15, kind: 'storage-tex:r32uint', note: 'SVGF current object ID (shade-write only; svgf-real reads it)' },
    ],
  },
  {
    id: 'scene',
    label: 'scene',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'storage-ro', note: 'bvhNodes' },
      { binding: 1, kind: 'storage-ro', note: 'bvhIndex (vec4u: [0..2]=indices, [3]=RGBA8 raw attCol)' },
      { binding: 2, kind: 'storage-ro', note: 'bvhPositions' },
      { binding: 3, kind: 'storage-ro', note: 'emitters' },
      { binding: 4, kind: 'storage-ro', note: 'emitterCdf' },
      // WS1 (2026-05-29) — bvh_beer (per-tri Beer-Lambert visible color, RGBA8
      // packed u32) moved from a storage buffer to a `texture_2d<u32>` (r32uint).
      // Textures do NOT count against maxStorageBuffersPerShaderStage, so this
      // swap frees a storage slot for `bvh_normal` (binding 11) while the scene
      // group's storage count stays at the 16-storage shade-pass floor. Only the
      // shade pass references binding 5 (lo_emit); the other primary passes
      // declare a subset of the layout, so the texture is shade-only.
      { binding: 5, kind: 'tex:uint', note: 'bvh_beer (Beer-Lambert visible color, r32uint texture; shade-only)' },
      { binding: 6, kind: 'storage-ro', note: 'tlasNodes' },
      { binding: 7, kind: 'storage-ro', note: 'tlasInstanceIndices' },
      { binding: 8, kind: 'storage-ro', note: 'tlasBlasRoots' },
      { binding: 9, kind: 'storage-ro', note: 'tlasInstanceWorldToLocal (mat4 cols)' },
      { binding: 10, kind: 'storage-ro', note: 'tlasInstanceLocalToWorld' },
      // WS1 (2026-05-29) — per-vertex world-space normals (stride-4 vec4f, .w
      // unused). Barycentric-blended in shade/ris/risGi/risGiNrc to produce a
      // SMOOTH shading normal (was faceted geometric). Data is the same
      // `shared.normals` already exposed as SceneBVHBuffers.emitterNormals. The
      // GPU-skin kernel writes its inverse-transpose normals here at
      // `baseVertex+vi` (the merged-BVH world-space slot). 1 storage buffer; the
      // bvh_beer→texture swap above keeps the net storage count unchanged.
      { binding: 11, kind: 'storage-ro', note: 'bvh_normal (per-vertex world-space smooth normals)' },
      // Camera-visible + GI-suffix emitters (2026-05-30; GI suffix 2026-06-20) —
      // per-triangle HDR emissive radiance
      // Le, rgba16float texture (texture, not storage — keeps the scene group at
      // the 16-storage floor, same rationale as bvh_beer). Shade/transparent OIT
      // read it for camera-visible glow; ReSTIR-GI suffix shading reads it before
      // applying readable emissive maps at hit UV.
      { binding: 12, kind: 'tex', note: 'bvh_emissive (per-tri HDR emissive Le, rgba16float texture; shade + OIT + ReSTIR-GI suffix)' },
      // H41 — analytic point/spot emitters for camera-visible NEE in opaque shade
      // and transparent OIT (separate from the RIS area-emitter pool). The
      // rgba32float texture starts with a header texel whose .x lane is the light
      // count, so zero-light scenes bind a valid placeholder while the loops stay
      // no-op.
      { binding: 13, kind: 'tex', note: 'analytic_lights (H41 point/spot NEE; rgba32float texture; shade + transparent OIT)' },
      // B1 (road-to-100) — per-triangle roughness+metalness lane (r32uint
      // texture, one u32 per triangle: bits[31:24]=rough×255, [23:16]=metal×255).
      // Read by ris/risGi/risGiNrc/restirCastPrimary/shade via
      // decodeRoughMetal(triIndex) to drive the GGX BRDF + glossy/metal GI target.
      // A texture (not a storage buffer) so it does NOT count against the
      // maxStorageBuffersPerShaderStage=16 shade-pass floor.
      { binding: 14, kind: 'tex:uint', note: 'bvh_material (per-tri roughness+metalness, r32uint texture)' },
      // B3 (road-to-100) — directional IBL. The equirect radiance map + the PBRT
      // 2D-distribution importance CDFs are TEXTURES (not storage buffers) so the
      // scene group stays at the 16-storage shade-pass floor (same rationale as
      // bvh_beer/bvh_material). A 1×1 placeholder + envParams.hasEnv=0 is bound for
      // non-HDRI scenes (the WGSL falls back to the scalar sky → byte-identical).
      // Only ris/risGi/shade reference these; the other passes declare a subset.
      { binding: 15, kind: 'tex', note: 'env_map (directional IBL radiance .rgb + per-texel solid-angle pdf .a, rgba16float)' },
      { binding: 16, kind: 'tex', note: 'env_marginal (1×H inverse-CDF, r32float; random→row v)' },
      { binding: 17, kind: 'tex', note: 'env_conditional (W×H inverse-CDF, r32float; random→col u)' },
      { binding: 18, kind: 'sampler:nf', note: 'env_sampler (declared for completeness; lookups use textureLoad)' },
      { binding: 19, kind: 'uniform', note: 'EnvParams { hasEnv, width, height, rotationY, intensity } — own uniform (WalkaroundUBO is frozen at 416B)' },
      // Phase-3D material-map atlas and per-triangle metadata.
      // Both are textures (not storage buffers) so the scene group stays inside
      // the full-tier WebGPU storage-buffer floor.
      { binding: 20, kind: 'tex-array', note: 'materialTextureAtlas (readable material maps as RGBA32F array layers)' },
      { binding: 21, kind: 'tex', note: 'baseColorMapMeta (per-triangle map layer/wrap/transform/coverage metadata)' },
      { binding: 22, kind: 'tex', note: 'bvh_tangent (per-vertex authored/generated tangent.xyzw, rgba32float texture)' },
      { binding: 23, kind: 'tex', note: 'bvh_vertex_color (per-vertex COLOR_0 rgba, rgba32float texture)' },
    ],
  },
  {
    id: 'ubo',
    label: 'ubo',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'uniform', note: 'WalkaroundUBO (256 bytes)' },
      { binding: 1, kind: 'tex', note: 'Sprint 15 — full-res GTAO occlusion factor (rgba16float), 1-frame lagged' },
      // Slot 2 — Sprint 9 adaptive-sampling tier (r32uint, sample-budget output).
      // risGi reads it to scale M_GI per pixel; ris/temporal/spatial/shade
      // declare the slot for layout compat but do not reference the symbol.
      { binding: 2, kind: 'tex:uint', note: 'adaptive-sampling tier (risGi reads; inert for ris/temporal/spatial/shade)' },
    ],
  },
  {
    id: 'gtao',
    label: 'gtao',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'gNormalDepth' },
      { binding: 1, kind: 'storage-tex:rgba16float', note: 'aoHalf out (E1 multi-bounce: bumped from r16float)' },
      { binding: 2, kind: 'uniform', note: 'GTAOUniforms' },
      { binding: 3, kind: 'tex', note: 'E1 — hdrAlbedoOut (Jiménez 2016 §5.2 multi-bounce term)' },
    ],
  },
  {
    id: 'gtaoUpsample',
    label: 'gtao-upsample',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'aoHalf in (per-channel multi-bounce AO)' },
      { binding: 1, kind: 'tex', note: 'gNormalDepth' },
      { binding: 2, kind: 'storage-tex:rgba16float', note: 'aoFull out (per-channel Jiménez 2016 §5.2 AO in .rgb)' },
      { binding: 3, kind: 'uniform', note: 'GTAOUniforms (audit B3 — bilateralDepthSigma)' },
    ],
  },
  {
    id: 'temporalGi',
    label: 'temporal-gi',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'storage-rw', note: 'reservoirGiCurrent' },
      { binding: 1, kind: 'storage-ro', note: 'reservoirGiPrevious' },
      { binding: 2, kind: 'uniform', note: 'WalkaroundUBO' },
    ],
  },
  {
    id: 'spatialGi',
    label: 'spatial-gi',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'storage-ro', note: 'input reservoir' },
      { binding: 1, kind: 'storage-rw', note: 'output reservoir' },
      { binding: 2, kind: 'uniform', note: 'WalkaroundUBO' },
    ],
  },
  {
    id: 'indirectCombine',
    label: 'indirect-combine',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'denoisedDirect' },
      { binding: 1, kind: 'tex', note: 'hdrIndirect' },
      { binding: 2, kind: 'storage-tex:rgba16float', note: 'combinedOut' },
      { binding: 3, kind: 'tex', note: 'Item 24 — albedo demodulation (Schied 2017 §4.1)' },
    ],
  },
  {
    id: 'indirectTemporalAccum',
    label: 'indirect-temporal-accum',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'currentRaw (hdrIndirectTexture)' },
      { binding: 1, kind: 'tex', note: 'prevAccum (previous frame accumulator output)' },
      { binding: 2, kind: 'storage-tex:rgba16float', note: 'outAccum (this frame accumulator output)' },
    ],
  },
  {
    id: 'transparentOit',
    label: 'transparent-oit',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'opaque/background combined radiance' },
      { binding: 1, kind: 'storage-tex:rgba16float', note: 'transparent-composited radiance out' },
    ],
  },
  {
    id: 'motionVectors',
    label: 'motion-vectors',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'gNormalDepth in' },
      { binding: 1, kind: 'storage-tex:rgba32float', note: 'motion out' },
      { binding: 2, kind: 'uniform', note: 'WalkaroundUBO' },
    ],
  },
  {
    id: 'resolve',
    label: 'resolve',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'uniform', note: 'ResolveUniforms (screen size + frame parity)' },
      { binding: 1, kind: 'tex', note: 'current radiance' },
      { binding: 2, kind: 'tex', note: 'previous radiance' },
      { binding: 3, kind: 'tex', note: 'motion vectors' },
      { binding: 4, kind: 'storage-tex:rgba16float', note: 'resolved out' },
    ],
  },
  {
    id: 'sampleBudget',
    label: 'sample-budget',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'uniform', note: 'SampleBudgetUniforms (thresholds + screen size)' },
      { binding: 1, kind: 'tex', note: 'variance source (rgba32float, welford)' },
      { binding: 2, kind: 'storage-tex:r32uint', note: 'tier output' },
      { binding: 3, kind: 'uniform', note: 'SampleCountUniforms (sample-count counter)' },
    ],
  },
  {
    id: 'composite',
    label: 'composite',
    visibility: 'fragment',
    entries: [
      { binding: 0, kind: 'tex', note: 'final blit source' },
      { binding: 1, kind: 'sampler:nf', note: 'composite sampler' },
      { binding: 2, kind: 'uniform', note: 'CompositeUniforms (tonemapMode, exposure, outputColorSpace)' },
    ],
  },
  {
    id: 'cbPrefill',
    label: 'cb-prefill',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'uniform', note: 'CbPrefillUniforms (screenW/H, frameParity, pad)' },
      { binding: 1, kind: 'tex', note: 'readAccum — previous-frame accumulated radiance' },
      { binding: 2, kind: 'tex', note: 'motionVectors — rgba32float NDC motion' },
      { binding: 3, kind: 'storage-tex:rgba16float', note: 'hdrColorTexture — gap-pixel fill output' },
    ],
  },
];

const TABLE_BY_ID: ReadonlyMap<BindGroupTableId, BindGroupTableEntry> = new Map(
  BIND_GROUP_TABLE.map((e) => [e.id, e]),
);

/** Look up a table entry by id; throws on unknown id (registration error). */
function getTableEntry(id: BindGroupTableId): BindGroupTableEntry {
  const e = TABLE_BY_ID.get(id);
  if (!e) throw new Error(`[bindGroupDescriptors] unknown table id: ${id}`);
  return e;
}

// ── kind → WebGPU shape mappers ──────────────────────────────────────────────

function visibilityFlag(v: 'compute' | 'fragment'): number {
  // Use the runtime global when present (real WebGPU env); otherwise fall back
  // to the spec-fixed bit constants so the layout factory can also run in a
  // non-WebGPU test environment (FRAGMENT = 0x2, COMPUTE = 0x4 per the WebGPU
  // GPUShaderStage flags). The numeric value is what createBindGroupLayout
  // validates against, so the fallback is byte-identical to the global.
  const stage: { FRAGMENT: number; COMPUTE: number } =
    (globalThis as { GPUShaderStage?: { FRAGMENT: number; COMPUTE: number } }).GPUShaderStage
    ?? { FRAGMENT: 0x2, COMPUTE: 0x4 };
  return v === 'fragment' ? stage.FRAGMENT : stage.COMPUTE;
}

/** Map a {@link BindingKind} to the resource-specific part of a layout entry. */
function layoutResourceFor(kind: BindingKind): Omit<GPUBindGroupLayoutEntry, 'binding' | 'visibility'> {
  switch (kind) {
    case 'storage-ro':
      return { buffer: { type: 'read-only-storage' } };
    case 'storage-rw':
      return { buffer: { type: 'storage' } };
    case 'uniform':
      return { buffer: { type: 'uniform' } };
    case 'tex':
      return { texture: { sampleType: 'unfilterable-float' } };
    case 'tex-array':
      return { texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } };
    case 'tex:uint':
      return { texture: { sampleType: 'uint' } };
    case 'sampler:nf':
      return { sampler: { type: 'non-filtering' } };
    case 'storage-tex:rgba16float':
      return { storageTexture: { access: 'write-only', format: 'rgba16float' } };
    case 'storage-tex:rgba32float':
      return { storageTexture: { access: 'write-only', format: 'rgba32float' } };
    case 'storage-tex:r32uint':
      return { storageTexture: { access: 'write-only', format: 'r32uint' } };
  }
}

/**
 * Generate the `GPUBindGroupLayoutEntry[]` for a table family. The cached BGL
 * factory in `bindGroupLayouts.ts` calls this so the layout entries are
 * derived from the same table the builder uses.
 */
export function bglEntriesFor(id: BindGroupTableId): GPUBindGroupLayoutEntry[] {
  const t = getTableEntry(id);
  const vis = visibilityFlag(t.visibility);
  return t.entries.map((d) => ({
    binding: d.binding,
    visibility: vis,
    ...layoutResourceFor(d.kind),
  }));
}

/**
 * Generic positional bind-group builder. Resources are supplied in binding
 * order (index 0 → first descriptor entry, …). Throws if the resource count
 * does not match the table's binding count — a loud, immediate failure beats a
 * silent partial bind group.
 *
 * The named builders in `bindGroupBuilders.ts` (`buildFrameBindGroup`, …) are
 * thin wrappers that assemble the positional array and delegate here, so both
 * the layout and the bind group flow from this one table.
 */
export function buildBindGroupFromTable(
  device: GPUDevice,
  id: BindGroupTableId,
  layout: GPUBindGroupLayout,
  resources: readonly GPUBindingResource[],
  /** Override the default `<label>-bg` GPU debug label (e.g. the spatialGi
   *  ping-pong builder distinguishes its two bind groups by label). */
  labelOverride?: string,
): GPUBindGroup {
  const t = getTableEntry(id);
  if (resources.length !== t.entries.length) {
    throw new Error(
      `[bindGroupDescriptors] '${id}' expects ${t.entries.length} resources, ` +
        `got ${resources.length}`,
    );
  }
  // Map over `resources` (not the table) so each element is a non-undefined
  // GPUBindingResource under noUncheckedIndexedAccess; the length check above
  // guarantees a 1:1 correspondence with the table's binding indices.
  return device.createBindGroup({
    label: labelOverride ?? `${t.label}-bg`,
    layout,
    entries: resources.map((resource, i) => ({ binding: t.entries[i]!.binding, resource })),
  });
}
