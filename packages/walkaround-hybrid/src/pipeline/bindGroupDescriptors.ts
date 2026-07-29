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
 * # CRITICAL — shared-layout binding rationale is preserved
 *
 * Several frame/ubo bindings are consumed by only a subset of the pipelines
 * that share the layout (for example, shade-only output slots 10/12/13/14/15).
 * Their rationale lives in each entry's `note` field below — do not prune a
 * pass-local binding merely because another pipeline using the same BGL does
 * not reference it.
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
  | 'tex:f'                       // texture: { sampleType: 'float' }
  | 'tex-array'                   // texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' }
  | 'tex:uint'                    // texture: { sampleType: 'uint' }
  | 'sampler:nf'                  // sampler: { type: 'non-filtering' }
  | 'sampler:f'                   // sampler: { type: 'filtering' }
  | 'storage-tex:rgba16float'     // storageTexture: write-only rgba16float
  | 'storage-tex:rgba32float'     // storageTexture: write-only rgba32float
  | 'storage-tex:r32uint';        // storageTexture: write-only r32uint

interface BindingDescriptor {
  readonly binding: number;
  readonly kind: BindingKind;
  /** Expected minimum byte size for buffer bindings with fixed WGSL structs. */
  readonly minSizeBytes?: number;
  /** Optional rationale — load-bearing for shared-layout bindings. */
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

// Fixed uniform binding sizes mirrored from the WGSL structs. These keep the
// descriptor table honest and make stale placeholder/cached-resource mistakes
// fail with an actionable table+binding error before backend validation does.
const WALKAROUND_UBO_BYTES = 432;
const ENV_PARAMS_BYTES = 32;
const GTAO_UBO_BYTES = 96;
const UBO_16_BYTES = 16;

// ── The table ────────────────────────────────────────────────────────────────

export const BIND_GROUP_TABLE: readonly BindGroupTableEntry[] = [
  {
    id: 'frame',
    label: 'frame',
    visibility: 'compute',
    entries: [
      // Primary visibility is generated by the RIS/shade BVH ray casts, so the
      // frame layout intentionally begins at binding 5. The earlier draft
      // G-buffer inputs at bindings 0-4 are not declared or bound.
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
      // The complete scene-storage contract is carried by three versioned raw
      // arenas. Keeping geometry, TLAS instances, and lighting in distinct
      // shards preserves targeted publication while every scene-consuming
      // pipeline stays inside WebGPU's guaranteed eight-storage-buffer floor.
      { binding: 0, kind: 'storage-ro', note: 'versioned scene geometry arena (BVH nodes/index/positions/normals)' },
      { binding: 1, kind: 'storage-ro', note: 'versioned scene TLAS arena (nodes/indices/roots/transforms)' },
      { binding: 2, kind: 'storage-ro', note: 'versioned scene lighting arena (emitters/CDF)' },
      // WS1 (2026-05-29) — bvh_beer (per-tri Beer-Lambert visible color, RGBA8
      // packed u32) moved from a storage buffer to a `texture_2d<u32>` (r32uint).
      // Textures do NOT count against maxStorageBuffersPerShaderStage, so this
      // swap frees a storage slot while the versioned scene group stays at
      // three storage arenas. Only the
      // shade pass references binding 5 (lo_emit); the other primary passes
      // declare a subset of the layout, so the texture is shade-only.
      { binding: 5, kind: 'tex:uint', note: 'bvh_beer (Beer-Lambert visible color, r32uint texture; shade-only)' },
      // Camera-visible + GI-suffix emitters (2026-05-30; GI suffix 2026-06-20) —
      // per-triangle HDR emissive radiance
      // Le, rgba16float texture (texture, not storage — keeps the scene group at
      // three storage arenas, same rationale as bvh_beer). Shade/transparent OIT
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
      // A texture (not a storage buffer) so it does not consume one of the
      // eight guaranteed storage-buffer bindings.
      { binding: 14, kind: 'tex:uint', note: 'bvh_material (per-tri roughness+metalness, r32uint texture)' },
      // B3 (road-to-100) — directional IBL. The equirect radiance map + the PBRT
      // 2D-distribution importance CDFs are TEXTURES (not storage buffers) so the
      // scene group stays at three versioned storage arenas (same rationale as
      // bvh_beer/bvh_material). A 1×1 placeholder + envParams.hasEnv=0 is bound for
      // non-HDRI scenes (the WGSL falls back to the scalar sky → byte-identical).
      // Only ris/risGi/shade reference these; the other passes declare a subset.
      { binding: 15, kind: 'tex', note: 'env_map (directional IBL radiance .rgb + per-texel solid-angle pdf .a, rgba16float)' },
      { binding: 16, kind: 'tex', note: 'env_marginal (1×H inverse-CDF, r32float; random→row v)' },
      { binding: 17, kind: 'tex', note: 'env_conditional (W×H inverse-CDF, r32float; random→col u)' },
      { binding: 18, kind: 'sampler:nf', note: 'env_sampler (declared for completeness; lookups use textureLoad)' },
      { binding: 19, kind: 'uniform', minSizeBytes: ENV_PARAMS_BYTES, note: 'EnvParams { hasEnv, width, height, rotationY, intensity } — own uniform (WalkaroundUBO is frozen at 432B)' },
      // Phase-3D material-map atlas and per-triangle metadata.
      // Both are textures (not storage buffers) so the scene group stays inside
      // the full-tier WebGPU storage-buffer floor.
      { binding: 20, kind: 'tex-array', note: 'materialTextureAtlas (CPU pixels or nominal GPU sources as mipmapped RGBA32F array layers)' },
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
      { binding: 0, kind: 'uniform', minSizeBytes: WALKAROUND_UBO_BYTES, note: 'WalkaroundUBO (432 bytes)' },
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
      { binding: 2, kind: 'uniform', minSizeBytes: GTAO_UBO_BYTES, note: 'GTAOUniforms' },
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
      { binding: 3, kind: 'uniform', minSizeBytes: GTAO_UBO_BYTES, note: 'GTAOUniforms (audit B3 — bilateralDepthSigma)' },
    ],
  },
  {
    id: 'temporalGi',
    label: 'temporal-gi',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'storage-rw', note: 'reservoirGiCurrent' },
      { binding: 1, kind: 'storage-ro', note: 'reservoirGiPrevious' },
      { binding: 2, kind: 'uniform', minSizeBytes: WALKAROUND_UBO_BYTES, note: 'WalkaroundUBO' },
    ],
  },
  {
    id: 'spatialGi',
    label: 'spatial-gi',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'storage-ro', note: 'input reservoir' },
      { binding: 1, kind: 'storage-rw', note: 'output reservoir' },
      { binding: 2, kind: 'uniform', minSizeBytes: WALKAROUND_UBO_BYTES, note: 'WalkaroundUBO' },
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
      { binding: 2, kind: 'tex', note: 'motion vectors (previous-current pixel delta)' },
      { binding: 3, kind: 'storage-tex:rgba16float', note: 'outAccum (this frame accumulator output)' },
    ],
  },
  {
    id: 'transparentOit',
    label: 'transparent-oit',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex:f', note: 'DDGI irradiance atlas' },
      { binding: 1, kind: 'tex:f', note: 'DDGI visibility atlas' },
      { binding: 2, kind: 'sampler:f', note: 'DDGI linear sampler' },
      { binding: 3, kind: 'uniform', note: 'DDGIGridUniform' },
      { binding: 4, kind: 'storage-ro', note: 'RC cascade-0 radiance' },
      { binding: 5, kind: 'uniform', note: 'RCParams' },
      { binding: 6, kind: 'tex', note: 'opaque/background combined radiance' },
      { binding: 7, kind: 'storage-tex:rgba16float', note: 'transparent-composited radiance out' },
    ],
  },
  {
    id: 'motionVectors',
    label: 'motion-vectors',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'tex', note: 'gNormalDepth in' },
      { binding: 1, kind: 'storage-tex:rgba32float', note: 'motion out' },
      { binding: 2, kind: 'uniform', minSizeBytes: WALKAROUND_UBO_BYTES, note: 'WalkaroundUBO' },
    ],
  },
  {
    id: 'resolve',
    label: 'resolve',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'uniform', minSizeBytes: UBO_16_BYTES, note: 'ResolveUniforms (screen size + frame parity)' },
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
      { binding: 0, kind: 'uniform', minSizeBytes: UBO_16_BYTES, note: 'SampleBudgetUniforms (thresholds + screen size)' },
      { binding: 1, kind: 'tex', note: 'variance source (rgba32float, welford)' },
      { binding: 2, kind: 'storage-tex:r32uint', note: 'tier output' },
      { binding: 3, kind: 'uniform', minSizeBytes: UBO_16_BYTES, note: 'SampleCountUniforms (sample-count counter)' },
    ],
  },
  {
    id: 'composite',
    label: 'composite',
    visibility: 'fragment',
    entries: [
      { binding: 0, kind: 'tex', note: 'final blit source' },
      { binding: 1, kind: 'uniform', minSizeBytes: UBO_16_BYTES, note: 'CompositeUniforms (tonemapMode, exposure, outputColorSpace)' },
    ],
  },
  {
    id: 'cbPrefill',
    label: 'cb-prefill',
    visibility: 'compute',
    entries: [
      { binding: 0, kind: 'uniform', minSizeBytes: UBO_16_BYTES, note: 'CbPrefillUniforms (screenW/H, frameParity, pad)' },
      { binding: 1, kind: 'tex', note: 'readAccum — previous-frame accumulated radiance' },
      { binding: 2, kind: 'tex', note: 'motionVectors — rgba32float previous-current pixel delta' },
      { binding: 3, kind: 'tex', note: 'current shaded-radiance snapshot for spatial reconstruction' },
      { binding: 4, kind: 'storage-tex:rgba16float', note: 'hdrColorTexture — gap-pixel fill output' },
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
function layoutResourceFor(
  kind: BindingKind,
  minSizeBytes?: number,
): Omit<GPUBindGroupLayoutEntry, 'binding' | 'visibility'> {
  const buffer = (
    type: GPUBufferBindingType,
  ): Omit<GPUBindGroupLayoutEntry, 'binding' | 'visibility'> => ({
    buffer: {
      type,
      ...(minSizeBytes !== undefined ? { minBindingSize: minSizeBytes } : {}),
    },
  });
  switch (kind) {
    case 'storage-ro':
      return buffer('read-only-storage');
    case 'storage-rw':
      return buffer('storage');
    case 'uniform':
      return buffer('uniform');
    case 'tex':
      return { texture: { sampleType: 'unfilterable-float' } };
    case 'tex:f':
      return { texture: { sampleType: 'float' } };
    case 'tex-array':
      return { texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } };
    case 'tex:uint':
      return { texture: { sampleType: 'uint' } };
    case 'sampler:nf':
      return { sampler: { type: 'non-filtering' } };
    case 'sampler:f':
      return { sampler: { type: 'filtering' } };
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
    ...layoutResourceFor(d.kind, d.minSizeBytes),
  }));
}

function bindingResourceSize(resource: GPUBindingResource): number | undefined {
  if (resource == null || typeof resource !== 'object' || !('buffer' in resource)) {
    return undefined;
  }
  const binding = resource;
  const buffer = binding.buffer as GPUBuffer & { readonly size?: number; readonly label?: string };
  if (typeof binding.size === 'number') return binding.size;
  if (typeof buffer.size !== 'number') return undefined;
  const offset = typeof binding.offset === 'number' ? binding.offset : 0;
  return Math.max(0, buffer.size - offset);
}

function bindingResourceLabel(resource: GPUBindingResource): string {
  if (resource == null || typeof resource !== 'object' || !('buffer' in resource)) {
    return '';
  }
  const buffer = (resource).buffer as GPUBuffer & { readonly label?: string };
  return buffer.label ?? '';
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
  for (let i = 0; i < resources.length; i += 1) {
    const descriptor = t.entries[i]!;
    const minSizeBytes = descriptor.minSizeBytes;
    if (minSizeBytes === undefined) continue;
    const resource = resources[i]!;
    const actualSizeBytes = bindingResourceSize(resource);
    if (actualSizeBytes !== undefined && actualSizeBytes < minSizeBytes) {
      const label = bindingResourceLabel(resource);
      throw new RangeError(
        `[bindGroupDescriptors] '${id}' binding ${descriptor.binding} requires ` +
          `${minSizeBytes} bytes for ${descriptor.note ?? descriptor.kind}, ` +
          `but received ${actualSizeBytes} bytes` +
          `${label ? ` from '${label}'` : ''}.`,
      );
    }
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
