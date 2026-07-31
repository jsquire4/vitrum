/**
 * RCDispatcher — per-frame cascade compute scheduler (raw WebGPU).
 *
 * Extracted from `_staging/legacy-source/src/rendering/scene/walkaround/cascadeDispatch.ts`.
 *
 * TSL → raw WebGPU conversion (RD-12).  See TSL_TO_RAW_MAPPING.md for the
 * full mapping rationale.  Summary:
 *   - TSL `storage()` → explicit `GPUBindGroupLayoutEntry` + `GPUBindGroup`
 *   - TSL `compute(fn, [64])` → `GPUComputePipeline` + `passEncoder.dispatchWorkgroups(ceil(n/64))`
 *   - TSL `instanceIndex` → `@builtin(global_invocation_id)` in WGSL entry point
 *   - `wgslFn()` bodies → pre-assembled WGSL module strings (see rc/wgsl/)
 *
 * The converted class:
 *   1. Allocates one `GPUComputePipeline` per cascade pass (cast: 5, merge: 4 = 9 total).
 *   2. Creates one bind group per pass from the caller-supplied GPU buffer handles.
 *   3. Dispatches passes in order: cast C0→C4, then merge C3→C0.
 *
 * Verification status: structural (TypeScript compile + binding-shape unit tests)
 * PLUS behavioral — the RC merged-BVH path is CPU-brute-force-oracle-validated
 * (tree-shape-invariant, 100% vs ground truth) and exercised by the rcEnabled
 * GPU smoke added with the F-RC1 stride fix (781f66f); the cascade-zero light-
 * model gaps were resolved 2026-06-07 (596c341 RC-has-energy gate + 1e893fa
 * probe-cast emitter NEE, converged A/B 999 dB). See README.md for residual risk.
 *
 * Dispatch counts — preserved from source TSL compute() arguments:
 *   Cast pass k:  totalRays = probes[0]*probes[1]*probes[2]*rays
 *                 workgroups = ceil(totalRays / 64)
 *   Merge pass k: totalLower = lowerDim.probes[0]*lowerDim.probes[1]*lowerDim.probes[2]*lowerDim.rays
 *                 workgroups = ceil(totalLower / 64)
 *
 * Workgroup size: 64 — matches `.compute(totalRays, [64])` in original.
 */

import { CASCADE_DIMS, validateCascadeDims, type CascadeDim } from './cascadePyramid.js';
import {
  assertRcEnvironmentRadianceF32,
  assertRcEnvironmentScaleF32,
} from './environmentRadianceScale.js';
import { PROBE_RAY_CAST_WGSL } from './wgsl/probeRayCast.wgsl.js';
import { CASCADE_MERGE_WGSL } from './wgsl/cascadeMerge.wgsl.js';
import { CascadeUniformsOffset } from './cascadeUniformsLayout.generated.js';

// ─── Internal types ───────────────────────────────────────────────────────────

interface CastPassHandles {
  pipeline:  GPUComputePipeline;
  /** Uniform buffer (GPUBuffer wrapping a Float32Array aligned to CascadeUniforms). */
  cascadeParamsBuf: GPUBuffer;
  /** CPU-side backing for the uniform buffer — updated each frame. */
  cascadeParamsRaw: Float32Array<ArrayBuffer>;
  /** Workgroup dispatch count = ceil(totalRays / 64). */
  dispatchX:  number;
}

interface MergePassHandles {
  pipeline:  GPUComputePipeline;
  /** Uniform buffer for MergeUniforms. */
  cascadeParamsBuf: GPUBuffer;
  /** Workgroup dispatch count = ceil(totalLower / 64). */
  dispatchX:  number;
}

interface DispatchHandles {
  castPasses:   CastPassHandles[];
  mergePasses:  MergePassHandles[];
  /** The env texture view bound in every cast pass at binding 6. */
  envTextureView: GPUTextureView;
  /** The env sampler bound in every cast pass at binding 7. */
  envSampler:     GPUSampler;
  /** Bind groups: one array<GPUBindGroup> per cast pass (index k), one per merge pass. */
  castBindGroups:  GPUBindGroup[];
  mergeBindGroups: GPUBindGroup[];
  /** Owned placeholder env texture when caller provided none. */
  placeholderEnvTexture?: GPUTexture;
  /** Owned placeholder material atlas textures when caller provided none. */
  placeholderMaterialAtlasTexture?: GPUTexture;
  placeholderMaterialMetaTexture?: GPUTexture;
  /** Owned 1x1 zero tangent texture when caller provided none. */
  placeholderTangentTexture?: GPUTexture;
  /** Owned 1x1 white vertex-color texture when caller provided none. */
  placeholderVertexColorTexture?: GPUTexture;
  /** Packed read-only material/TLAS arena consumed by the cast shader. */
  sceneArenaBuf: GPUBuffer;
  /** Dirty source ranges copied incrementally into {@link sceneArenaBuf}. */
  sceneArenaCopies: SceneArenaCopy[];
  materialArenaVersion: number;
  tlasArenaVersion: number;
  /** Every destroyable resource allocated while building this candidate. */
  ownedResources: OwnedGpuResource[];
}

interface OwnedGpuResource {
  destroy(): void;
}

type OwnGpuResource = <T extends OwnedGpuResource>(resource: T) => T;

interface SceneArenaCopy {
  source: GPUBuffer;
  readonly destinationOffset: number;
  readonly size: number;
  readonly category: 'material' | 'tlas';
  dirty: boolean;
}

/**
 * The cast shader stays within WebGPU's guaranteed minimum of eight storage
 * buffers per shader stage. Hosts therefore never need to negotiate an
 * optional adapter limit just to enable RC.
 */
export const RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE = 8;

/** Runtime dielectric-interface budget accepted by the bounded RC glass walk. */
export const RC_MIN_TRANSMITTED_INTERFACE_BUDGET = 1;
export const RC_MAX_TRANSMITTED_INTERFACE_BUDGET = 8;
export const RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET = 8;

const RC_SCENE_ARENA_HEADER_WORDS = 16;
const RC_SCENE_ARENA_HEADER_BYTES = RC_SCENE_ARENA_HEADER_WORDS * 4;
const RC_CASCADE_UNIFORM_WORDS = 40;
const RC_CASCADE_UNIFORM_BYTES = RC_CASCADE_UNIFORM_WORDS * 4;
const RC_EMITTER_STRIDE_BYTES = 80;
const RC_ALIAS_STRIDE_BYTES = 16;
const RC_LIGHT_HEADER_BYTES = 16;
const RC_LIGHT_STRIDE_BYTES = 64;
const RC_LIGHTS_BUFFER_BYTES = RC_LIGHT_HEADER_BYTES;
const UINT32_MAX = 0xffff_ffff;

interface DispatchBindingSignature {
  readonly device: GPUDevice;
  readonly bvhMode: 'merged' | 'tlas';
  readonly bvhNodesBuf: GPUBuffer;
  readonly bvhNodesOffset: number;
  readonly bvhNodesSize: number | null;
  readonly bvhIndicesBuf: GPUBuffer;
  readonly bvhIndicesOffset: number;
  readonly bvhIndicesSize: number | null;
  readonly bvhPositionsBuf: GPUBuffer;
  readonly bvhPositionsOffset: number;
  readonly bvhPositionsSize: number | null;
  readonly bvhNormalsBuf: GPUBuffer;
  readonly bvhNormalsOffset: number;
  readonly bvhNormalsSize: number | null;
  readonly cascadeBufs: readonly GPUBuffer[];
  readonly probeOriginWorld: readonly [number, number, number];
  readonly roomSize: readonly [number, number, number];
  readonly envTextureView: GPUTextureView | null;
  readonly envSampler: GPUSampler | null;
  readonly materialTextureAtlasView: GPUTextureView | null;
  readonly materialMapMetaTextureView: GPUTextureView | null;
  readonly bvhTangentTextureView: GPUTextureView | null;
  readonly bvhVertexColorTextureView: GPUTextureView | null;
  readonly emittersBuf: GPUBuffer | null;
  readonly emittersOffset: number;
  readonly emittersSize: number | null;
  readonly emitterDataOffset: number;
  readonly emitterAliasOffset: number;
  readonly lightsBuf: GPUBuffer | null;
  readonly lightsOffset: number;
  readonly lightsSize: number | null;
}

// ─── Public interface ─────────────────────────────────────────────────────────

/**
 * Raw-GPU dispatch options. Hosts that own a `GPUDevice` + raw `GPUBuffer`
 * handles (e.g. `HybridEngine`, which runs the WGSL shade pipeline directly
 * and does not use THREE's WebGPU renderer backend) pass this directly to
 * `RCDispatcher.dispatchFrameRaw`.
 *
 * History: the W8 Phase 1B refactor (2026-05-18) split this out from a
 * THREE-tied `RCDispatchOpts` that extracted GPU handles via
 * `StorageBufferAttribute.__gpuBuffer` reach-through. The legacy THREE-tied
 * path was dropped on 2026-05-18 once `RCSubsystem` (the in-engine consumer)
 * was confirmed to be the only call site and was already using this raw
 * variant. The `__gpuBuffer` accessor is gone from the dispatcher.
 */
export interface RCDispatchOptsRaw {
  /** Raw WebGPU device — caller-owned. */
  device:             GPUDevice;

  /**
   * BVH GPU buffers. The optional byte ranges let an owning renderer bind
   * RC directly to subranges of its canonical scene-storage arena instead of
   * allocating a second geometry BVH. Offsets must satisfy the device storage
   * binding alignment; sizes default to the remainder of the buffer.
   */
  bvhNodesBuf:        GPUBuffer;
  bvhNodesOffset?:    number;
  bvhNodesSize?:      number;
  bvhIndicesBuf:      GPUBuffer;
  bvhIndicesOffset?:  number;
  bvhIndicesSize?:    number;
  bvhPositionsBuf:    GPUBuffer;
  bvhPositionsOffset?: number;
  bvhPositionsSize?:   number;
  /** Packed normal.xyz plus UV1 in .w, matching the shared ReSTIR BVH layout. */
  bvhNormalsBuf:      GPUBuffer;
  bvhNormalsOffset?:  number;
  bvhNormalsSize?:    number;
  /** Must include `GPUBufferUsage.COPY_SRC`; packed into RC's scene arena. */
  materialsBuf:       GPUBuffer;
  /** Must include `GPUBufferUsage.COPY_SRC`; packed into RC's scene arena. */
  triMaterialIdBuf:   GPUBuffer;
  /** Increment when material/triangle-material data is mutated in place. A
   *  replacement buffer identity is detected automatically. */
  materialArenaVersion?: number;

  /** One cascade-output `GPUBuffer` per cascade, same order as
   *  {@link CASCADE_DIMS}. The dispatcher writes into these (cast) and
   *  reads/writes parents (merge). */
  cascadeBufs:        readonly GPUBuffer[];

  /** Cascade geometry — plain world-space origin and room-size tuples. */
  probeOriginWorld:   readonly [number, number, number];
  roomSize:           readonly [number, number, number];

  /** Sun direction (world space, normalised) and RGB tint. */
  sunDirection:       readonly [number, number, number];
  sunColor:           readonly [number, number, number];
  /** When true, direct RC sun lighting skips the sun visibility ray. */
  sunCastShadowDisabled?: boolean;
  /** Finite directional emitter cone radius in radians. */
  sunAngularRadius?: number;

  /** Environment equirectangular texture — caller supplies a pre-created
   *  view + sampler. Pass `null` to use the dispatcher's 1×1 black placeholder. */
  envTextureView?:    GPUTextureView | null;
  envSampler?:        GPUSampler | null;
  /**
   * H6 world-to-unrotated-map Y rotation in radians. The shader evaluates
   * `RY(-envRotationY) * worldDirection` before equirectangular lookup.
   * Defaults to 0.
   */
  envRotationY?:      number;
  /**
   * Linear radiance multiplier for the environment map. The bound map is
   * expected to contain unit-intensity texels. Defaults to 1.
   */
  envIntensity?:      number;
  /**
   * Constant sky radiance used on last-cascade misses when no directional
   * environment is active. Defaults to black for raw callers.
   */
  scalarSkyRadiance?: readonly [number, number, number];
  /**
   * Whether the supplied texture/sampler pair contains a live directional
   * environment. Defaults to true when a pair is supplied and false otherwise,
   * preserving existing raw-call behavior while allowing a bindable black
   * placeholder to select {@link scalarSkyRadiance}.
   */
  hasDirectionalEnvironment?: boolean;

  /** Material texture atlas + per-triangle map metadata for UV-varying
   *  material-backed emitter radiance. When omitted, RC falls back to the
   *  scalar `EmitterTri.Le` path for every emitter sample. */
  materialTextureAtlasView?: GPUTextureView | null;
  materialMapMetaTextureView?: GPUTextureView | null;
  /** Authored/generated tangent.xyzw texture matching the main walkaround
   *  scene binding. When omitted, mapped normal/bump paths fall back to a
   *  derived UV-gradient tangent frame. */
  bvhTangentTextureView?: GPUTextureView | null;
  /** Per-vertex COLOR_0 rgba texture matching the main walkaround scene
   *  binding. When omitted, RC uses opaque white so raw callers preserve the
   *  historical scalar/material-map alpha behavior. */
  bvhVertexColorTextureView?: GPUTextureView | null;

  frameSeed:          number;
  /** Möller–Trumbore coplanarity threshold. Default 1e-5. */
  triIntersectEpsilon?: number;
  /** Maximum dielectric interfaces crossed by one transmitted probe ray.
   *  Integer in [1, 8]. Defaults to 8. Thin sheets consume two interfaces. */
  transmittedInterfaceBudget?: number;

  /** C2 — TLAS traversal (ReSTIR-shared buffers). Omit for merged-only RC BVH. */
  bvhMode?: 'merged' | 'tlas';
  tlasNodeCount?: number;
  /** TLAS buffers must include `GPUBufferUsage.COPY_SRC`; RC packs them into
   *  one read-only arena to stay within WebGPU's guaranteed binding limits. */
  tlasNodesBuf?: GPUBuffer;
  tlasInstanceIndicesBuf?: GPUBuffer;
  tlasBlasRootsBuf?: GPUBuffer;
  tlasInstanceWorldToLocalBuf?: GPUBuffer;
  tlasInstanceLocalToWorldBuf?: GPUBuffer;
  /** Increment when any TLAS buffer is mutated in place. Replacement buffer
   *  identities are detected automatically. */
  tlasArenaVersion?: number;

  /** Rect-area emitter NEE (2026-06-07). The packed `array<EmitterTri>` buffer
   *  (80 bytes/tri — share the main pipeline's `BvhBufferHost._emitterBuffer`)
   *  + its triangle count. Omit (or count 0) to keep RC's prior light model
   *  (sun + emissive geometry + env); the dispatcher binds an 80-byte zero
   *  placeholder so the bind group stays valid. */
  emittersBuf?: GPUBuffer;
  emittersOffset?: number;
  emittersSize?: number;
  /** Byte offsets relative to the bound emitter window. */
  emitterDataOffset?: number;
  /** Defaults to the first byte after the resolved emitter-data range. */
  emitterAliasOffset?: number;
  emitterCount?: number;

  /** Runtime-sized RCLight records plus represented-PMF alias entries.
   * Omit to bind a header-only zero placeholder. */
  lightsBuf?: GPUBuffer;
  lightsOffset?: number;
  lightsSize?: number;
  lightCount?: number;
}

// ─── Uniform data builders ────────────────────────────────────────────────────
// These are the exact same packing functions as the original cascadeDispatch.ts.

/**
 * Named options bag for {@link buildCascadeUniformDataInto} (D13.2).
 *
 * Field order and packing are identical to the old positional-params signature;
 * this object form is used internally to eliminate the 13-param call.
 */
export interface CascadeUniformInputs {
  readonly probeOriginWorld: readonly [number, number, number];
  readonly roomSize:         readonly [number, number, number];
  readonly sunDir:           readonly [number, number, number];
  readonly sunColor:         readonly [number, number, number];
  readonly sunCastShadowDisabled: boolean;
  readonly sunAngularRadius: number;
  readonly envIntensity:     number;
  /** H6 world-to-unrotated-map Y rotation. Defaults to 0. */
  readonly envRotationY?:    number;
  /** Scalar sky radiance selected when no directional environment is active. */
  readonly scalarSkyRadiance?: readonly [number, number, number];
  readonly hasDirectionalEnvironment?: boolean;
  readonly frameSeed:        number;
  /** E2: Möller–Trumbore coplanarity threshold (was local WGSL const). */
  readonly triIntersectEpsilon: number;
  /** Bounded transmitted-dielectric interface count. Defaults to 8. */
  readonly transmittedInterfaceBudget?: number;
  readonly bvhMode:          number;
  readonly tlasNodeCount:    number;
  readonly emitterCount:     number;
  /** Runtime punctual/directional analytic light count. */
  readonly lightCount:       number;
  readonly emitterDataWordOffset?: number;
  readonly emitterAliasWordOffset?: number;
  /** Per-instance cascade dimensions. Defaults to {@link CASCADE_DIMS}. */
  readonly dims?:            readonly CascadeDim[];
}

interface ResolvedEmitterLayout {
  readonly count: number;
  readonly dataOffset: number;
  readonly aliasOffset: number;
  readonly dataBytes: number;
  readonly aliasBytes: number;
  readonly dataWordOffset: number;
  readonly aliasWordOffset: number;
}

/**
 * Resolve the one emitter-buffer layout used by validation and every shader
 * uniform writer. In particular, an omitted alias offset follows the resolved
 * data offset and data extent; it must never validate at one address and reach
 * WGSL as zero.
 */
export function resolveEmitterLayout(
  opts: Pick<
    RCDispatchOptsRaw,
    'emitterCount' | 'emitterDataOffset' | 'emitterAliasOffset'
  >,
): ResolvedEmitterLayout {
  const count = opts.emitterCount ?? 0;
  const dataOffset = opts.emitterDataOffset ?? 0;
  const dataBytes = count * RC_EMITTER_STRIDE_BYTES;
  const aliasBytes = count * RC_ALIAS_STRIDE_BYTES;
  const defaultAliasOffset = dataOffset + dataBytes;
  if (
    !Number.isSafeInteger(dataBytes) ||
    !Number.isSafeInteger(aliasBytes) ||
    !Number.isSafeInteger(defaultAliasOffset)
  ) {
    throw new Error('[RCDispatcher] emitter sampling ranges exceed Number.MAX_SAFE_INTEGER.');
  }
  const aliasOffset = opts.emitterAliasOffset ?? defaultAliasOffset;
  return {
    count,
    dataOffset,
    aliasOffset,
    dataBytes,
    aliasBytes,
    dataWordOffset: dataOffset / 4,
    aliasWordOffset: aliasOffset / 4,
  };
}

function assertU32(value: unknown, path: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0 || (value as number) > UINT32_MAX) {
    throw new Error(`${path} must be an unsigned 32-bit integer; received ${String(value)}`);
  }
}

function assertFiniteF32(value: unknown, path: string): asserts value is number {
  if (
    typeof value !== 'number' ||
    !Number.isFinite(value) ||
    !Number.isFinite(Math.fround(value))
  ) {
    throw new Error(`${path} must be a finite f32; received ${String(value)}`);
  }
}

function assertFiniteVec3(
  value: unknown,
  path: string,
  predicate?: (component: number) => boolean,
  predicateDescription?: string,
): asserts value is readonly [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${path} must be a [x, y, z] tuple`);
  }
  const tuple: readonly unknown[] = value;
  for (let axis = 0; axis < 3; axis += 1) {
    const component = tuple[axis];
    assertFiniteF32(component, `${path}[${axis}]`);
    if (predicate && !predicate(component)) {
      throw new Error(`${path}[${axis}] must be ${predicateDescription ?? 'valid'}`);
    }
  }
}

function validateCascadeUniformInputs(
  k: number,
  inputs: CascadeUniformInputs,
): readonly CascadeDim[] {
  const dims = validateCascadeDims(inputs.dims ?? CASCADE_DIMS, 'CascadeUniformInputs.dims');
  if (!Number.isSafeInteger(k) || k < 0 || k >= dims.length) {
    throw new Error(`cascade index must be an integer in [0, ${dims.length}); received ${String(k)}`);
  }
  assertFiniteVec3(inputs.probeOriginWorld, 'probeOriginWorld');
  assertFiniteVec3(inputs.roomSize, 'roomSize', component => component > 0, 'positive');
  assertFiniteVec3(inputs.sunDir, 'sunDir');
  const sunLength = Math.hypot(inputs.sunDir[0], inputs.sunDir[1], inputs.sunDir[2]);
  if (!Number.isFinite(sunLength) || Math.abs(sunLength - 1) > 1e-3) {
    throw new Error(`sunDir must be normalized; length=${String(sunLength)}`);
  }
  assertFiniteVec3(inputs.sunColor, 'sunColor', component => component >= 0, 'nonnegative');
  if (typeof inputs.sunCastShadowDisabled !== 'boolean') {
    throw new Error('sunCastShadowDisabled must be boolean');
  }
  assertFiniteF32(inputs.sunAngularRadius, 'sunAngularRadius');
  if (inputs.sunAngularRadius < 0 || inputs.sunAngularRadius > Math.PI) {
    throw new Error('sunAngularRadius must be in [0, PI]');
  }
  assertRcEnvironmentScaleF32(inputs.envIntensity, 'envIntensity');
  assertFiniteF32(inputs.envRotationY ?? 0, 'envRotationY');
  assertRcEnvironmentRadianceF32(
    inputs.scalarSkyRadiance ?? [0, 0, 0],
    'scalarSkyRadiance',
  );
  if (typeof (inputs.hasDirectionalEnvironment ?? false) !== 'boolean') {
    throw new Error('hasDirectionalEnvironment must be boolean');
  }
  assertU32(inputs.frameSeed, 'frameSeed');
  assertFiniteF32(inputs.triIntersectEpsilon, 'triIntersectEpsilon');
  if (inputs.triIntersectEpsilon <= 0) {
    throw new Error('triIntersectEpsilon must be positive');
  }
  const transmittedInterfaceBudget = inputs.transmittedInterfaceBudget
    ?? RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET;
  assertU32(transmittedInterfaceBudget, 'transmittedInterfaceBudget');
  if (
    transmittedInterfaceBudget < RC_MIN_TRANSMITTED_INTERFACE_BUDGET ||
    transmittedInterfaceBudget > RC_MAX_TRANSMITTED_INTERFACE_BUDGET
  ) {
    throw new Error(
      `transmittedInterfaceBudget must be an integer in ` +
      `[${RC_MIN_TRANSMITTED_INTERFACE_BUDGET}, ${RC_MAX_TRANSMITTED_INTERFACE_BUDGET}]; ` +
      `received ${String(transmittedInterfaceBudget)}`,
    );
  }
  if (inputs.bvhMode !== 0 && inputs.bvhMode !== 1) {
    throw new Error(`bvhMode must be exactly 0 or 1; received ${String(inputs.bvhMode)}`);
  }
  assertU32(inputs.tlasNodeCount, 'tlasNodeCount');
  assertU32(inputs.emitterCount, 'emitterCount');
  assertU32(inputs.lightCount, 'lightCount');
  assertU32(inputs.emitterDataWordOffset ?? 0, 'emitterDataWordOffset');
  assertU32(inputs.emitterAliasWordOffset ?? 0, 'emitterAliasWordOffset');
  return dims;
}

/** Write CascadeUniforms into an existing Float32Array (avoids realloc per frame).
 *
 * sunDir / sunColor / cascade geometry are plain `readonly [number, number, number]`
 * tuples — no `THREE.Vector3` / `THREE.Color` coupling.
 *
 * B3b (2026-05-19) — `dims` parameter replaces the module-level
 * `CASCADE_DIMS` lookup. Per-instance dims flow from
 * `HybridEngineOptions.cascadeDims` through `RCDispatcher.constructor`.
 */
export function buildCascadeUniformDataInto(
  d: Float32Array,
  k: number,
  inputs: CascadeUniformInputs,
): void {
  if (!(d instanceof Float32Array) || d.length < RC_CASCADE_UNIFORM_WORDS) {
    throw new Error(
      `CascadeUniforms destination must be a Float32Array of at least ` +
      `${RC_CASCADE_UNIFORM_WORDS} words`,
    );
  }
  const {
    probeOriginWorld,
    roomSize,
    sunDir,
    sunColor,
    sunCastShadowDisabled,
    sunAngularRadius,
    envIntensity,
    envRotationY = 0,
    scalarSkyRadiance = [0, 0, 0],
    hasDirectionalEnvironment = false,
    frameSeed,
    triIntersectEpsilon,
    transmittedInterfaceBudget = RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
    bvhMode,
    tlasNodeCount,
    emitterCount,
    lightCount,
    emitterDataWordOffset = 0,
    emitterAliasWordOffset = 0,
  } = inputs;
  const dims = validateCascadeUniformInputs(k, inputs);
  const dim = dims[k]!;
  const rayGridSize = Math.round(Math.sqrt(dim.rays));
  const o = probeOriginWorld;
  const s = roomSize;
  // CascadeUniforms layout is generated from the WGSL struct by
  // tools/generate-wgsl-layouts.mjs (cascadeUniformsLayout.generated.ts). Each
  // field is indexed by its named byte offset / 4 (all fields are 4-aligned) so
  // adding/moving a field is a single-site edit in the codegen field list — the
  // magic slot indices (ui[29..31] etc.) and the drift-prone ASCII layout
  // comment are gone. The buffer remains exactly 40 words (160 bytes).
  const O = CascadeUniformsOffset;
  d.fill(0, 0, RC_CASCADE_UNIFORM_WORDS);
  const ui = new Uint32Array(d.buffer, d.byteOffset, d.length);
  d[O.probeOriginWorld / 4 + 0] = o[0];
  d[O.probeOriginWorld / 4 + 1] = o[1];
  d[O.probeOriginWorld / 4 + 2] = o[2];
  d[O.roomSize / 4 + 0] = s[0];
  d[O.roomSize / 4 + 1] = s[1];
  d[O.roomSize / 4 + 2] = s[2];
  ui[O.probeCount / 4 + 0] = dim.probes[0];
  ui[O.probeCount / 4 + 1] = dim.probes[1];
  ui[O.probeCount / 4 + 2] = dim.probes[2];
  ui[O.raysPerProbe / 4] = dim.rays;
  ui[O.rayGridSize / 4] = rayGridSize;
  d[O.intervalNear / 4] = dim.intervalNear;
  d[O.intervalFar / 4] = dim.intervalFar;
  ui[O.cascadeIndex / 4] = k;
  d[O.sunDirection / 4 + 0] = sunDir[0];
  d[O.sunDirection / 4 + 1] = sunDir[1];
  d[O.sunDirection / 4 + 2] = sunDir[2];
  d[O.sunAngularRadius / 4] = sunAngularRadius;
  d[O.sunColor / 4 + 0] = sunColor[0];
  d[O.sunColor / 4 + 1] = sunColor[1];
  d[O.sunColor / 4 + 2] = sunColor[2];
  d[O.envIntensity / 4] = envIntensity;
  ui[O.frameSeed / 4] = frameSeed;
  ui[O.lastCascade / 4] = dims.length - 1;
  d[O.triIntersectEpsilon / 4] = triIntersectEpsilon;  // E2: UBO-plumbed (was local const)
  ui[O.bvhMode / 4] = bvhMode;
  ui[O.tlasNodeCount / 4] = tlasNodeCount;
  ui[O.emitterCount / 4] = emitterCount;         // RC emitter NEE
  ui[O.lightCount / 4] = lightCount;
  ui[O.sunCastShadowDisabled / 4] = sunCastShadowDisabled ? 1 : 0; // directional castShadow:false
  ui[O.emitterDataWordOffset / 4] = emitterDataWordOffset;
  ui[O.emitterAliasWordOffset / 4] = emitterAliasWordOffset;
  ui[O.transmittedInterfaceBudget / 4] = transmittedInterfaceBudget;
  d[O.envRotationY / 4] = envRotationY;
  d[O.scalarSkyRadiance / 4 + 0] = scalarSkyRadiance[0];
  d[O.scalarSkyRadiance / 4 + 1] = scalarSkyRadiance[1];
  d[O.scalarSkyRadiance / 4 + 2] = scalarSkyRadiance[2];
  ui[O.hasDirectionalEnv / 4] = hasDirectionalEnvironment ? 1 : 0;
}

function buildMergeUniformData(
  lowerDim: CascadeDim,
  upperDim: CascadeDim,
  probeOriginWorld: readonly [number, number, number],
  roomSize:         readonly [number, number, number],
): Float32Array {
  // MergeUniforms layout (matches WGSL struct in cascadeMerge.wgsl.ts):
  // lowerProbeCount(3u), lowerRayCount(u)
  // upperProbeCount(3u), upperRayCount(u)
  // lowerRayGridSize(u), upperRayGridSize(u), _pad0(2u)
  // probeOriginWorld(3f), _pad1(f)
  // roomSize(3f), _pad2(f)
  // Total: 20 float/uint values = 80 bytes
  const d = new Float32Array(20);
  const ui = new Uint32Array(d.buffer);
  const o = probeOriginWorld;
  const s = roomSize;
  ui[0]  = lowerDim.probes[0]; ui[1]  = lowerDim.probes[1]; ui[2]  = lowerDim.probes[2];
  ui[3]  = lowerDim.rays;
  ui[4]  = upperDim.probes[0]; ui[5]  = upperDim.probes[1]; ui[6]  = upperDim.probes[2];
  ui[7]  = upperDim.rays;
  ui[8]  = Math.round(Math.sqrt(lowerDim.rays));
  ui[9]  = Math.round(Math.sqrt(upperDim.rays));
  ui[10] = 0; ui[11] = 0;
  d[12]  = o[0]; d[13] = o[1]; d[14] = o[2]; d[15] = 0;
  d[16]  = s[0]; d[17] = s[1]; d[18] = s[2]; d[19] = 0;
  return d;
}

function assertNonnegativeSafeInteger(value: unknown, path: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${path} must be a nonnegative safe integer; received ${String(value)}`);
  }
}

function validateBuffer(
  buffer: GPUBuffer | null | undefined,
  label: string,
  stride: number,
  requiredUsage: number,
): GPUBuffer {
  if (buffer == null || typeof buffer !== 'object') {
    throw new Error(`[RCDispatcher] ${label} is required.`);
  }
  const size = buffer.size;
  if (!Number.isSafeInteger(size) || size < stride || size % stride !== 0) {
    throw new Error(
      `[RCDispatcher] ${label}.size=${String(size)} must be a positive safe-integer ` +
      `multiple of ${stride}.`,
    );
  }
  if (typeof buffer.usage === 'number' && (buffer.usage & requiredUsage) !== requiredUsage) {
    throw new Error(`[RCDispatcher] ${label} is missing required GPUBufferUsage flags.`);
  }
  return buffer;
}

interface ValidatedStorageBinding {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
  readonly resource: GPUBufferBinding;
}

function validateStorageBinding(
  device: GPUDevice,
  buffer: GPUBuffer | null | undefined,
  label: string,
  stride: number,
  requiredUsage: number,
  offsetValue: number | undefined,
  sizeValue: number | undefined,
): ValidatedStorageBinding {
  const checked = validateBuffer(buffer, label, 4, requiredUsage);
  const offset = offsetValue ?? 0;
  assertNonnegativeSafeInteger(offset, `${label}Offset`);
  const alignment = device.limits?.minStorageBufferOffsetAlignment;
  if (typeof alignment === 'number' && offset % alignment !== 0) {
    throw new Error(
      `[RCDispatcher] ${label}Offset must satisfy ` +
      `minStorageBufferOffsetAlignment=${alignment}.`,
    );
  }
  const size = sizeValue ?? checked.size - offset;
  assertNonnegativeSafeInteger(size, `${label}Size`);
  if (
    size < stride || size % stride !== 0 ||
    offset > checked.size - size
  ) {
    throw new Error(
      `[RCDispatcher] ${label} binding [${offset}, ${offset + size}) must be ` +
      `in bounds and have a positive size divisible by ${stride}.`,
    );
  }
  return {
    buffer: checked,
    offset,
    size,
    resource: { buffer: checked, offset, size },
  };
}

function validateSceneArenaSources(opts: RCDispatchOptsRaw): {
  arenaBytes: number;
  sourceBuffers: readonly GPUBuffer[];
} {
  const sources: GPUBuffer[] = [
    validateBuffer(opts.materialsBuf, 'materialsBuf', 64, GPUBufferUsage.COPY_SRC),
    validateBuffer(opts.triMaterialIdBuf, 'triMaterialIdBuf', 4, GPUBufferUsage.COPY_SRC),
  ];
  const mode = opts.bvhMode ?? 'merged';
  if (mode === 'tlas') {
    sources.push(
      validateBuffer(opts.tlasNodesBuf, 'tlasNodesBuf', 32, GPUBufferUsage.COPY_SRC),
      validateBuffer(opts.tlasInstanceIndicesBuf, 'tlasInstanceIndicesBuf', 4, GPUBufferUsage.COPY_SRC),
      validateBuffer(opts.tlasBlasRootsBuf, 'tlasBlasRootsBuf', 4, GPUBufferUsage.COPY_SRC),
      validateBuffer(opts.tlasInstanceWorldToLocalBuf, 'tlasInstanceWorldToLocalBuf', 64, GPUBufferUsage.COPY_SRC),
      validateBuffer(opts.tlasInstanceLocalToWorldBuf, 'tlasInstanceLocalToWorldBuf', 64, GPUBufferUsage.COPY_SRC),
    );
  }

  let arenaBytes = RC_SCENE_ARENA_HEADER_BYTES;
  for (const source of sources) {
    if (arenaBytes > Number.MAX_SAFE_INTEGER - source.size) {
      throw new Error('[RCDispatcher] packed scene arena size exceeds the safe-integer range.');
    }
    arenaBytes += source.size;
    if (arenaBytes / 4 > UINT32_MAX) {
      throw new Error('[RCDispatcher] packed scene arena exceeds its u32 word-offset encoding.');
    }
  }
  return { arenaBytes, sourceBuffers: sources };
}

function assertLimit(
  actual: number | undefined,
  required: number,
  name: string,
): void {
  if (typeof actual === 'number' && actual < required) {
    throw new Error(`[RCDispatcher] ${name}=${actual}; RC requires at least ${required}.`);
  }
}

function validateDispatchOptsRaw(
  opts: RCDispatchOptsRaw,
  dims: readonly CascadeDim[],
): void {
  if (opts == null || typeof opts !== 'object' || opts.device == null) {
    throw new Error('[RCDispatcher] dispatch options must include a GPUDevice.');
  }
  const mode = opts.bvhMode ?? 'merged';
  if (mode !== 'merged' && mode !== 'tlas') {
    throw new Error(`[RCDispatcher] bvhMode must be "merged" or "tlas"; received ${String(mode)}.`);
  }
  assertU32(opts.materialArenaVersion ?? 0, 'materialArenaVersion');
  assertU32(opts.tlasArenaVersion ?? 0, 'tlasArenaVersion');
  assertU32(opts.tlasNodeCount ?? 0, 'tlasNodeCount');
  assertU32(opts.emitterCount ?? 0, 'emitterCount');
  assertU32(opts.lightCount ?? 0, 'lightCount');
  const emitterLayout = resolveEmitterLayout(opts);

  const hasEnvView = opts.envTextureView != null;
  const hasEnvSampler = opts.envSampler != null;
  if (hasEnvView !== hasEnvSampler) {
    throw new Error('[RCDispatcher] envTextureView and envSampler must be supplied together.');
  }
  const hasDirectionalEnvironment =
    opts.hasDirectionalEnvironment ?? (hasEnvView && hasEnvSampler);
  if (hasDirectionalEnvironment === true && !hasEnvView) {
    throw new Error(
      '[RCDispatcher] hasDirectionalEnvironment=true requires envTextureView and envSampler.',
    );
  }

  validateCascadeUniformInputs(0, {
    probeOriginWorld: opts.probeOriginWorld,
    roomSize: opts.roomSize,
    sunDir: opts.sunDirection,
    sunColor: opts.sunColor,
    sunCastShadowDisabled: opts.sunCastShadowDisabled ?? false,
    sunAngularRadius: opts.sunAngularRadius ?? 0,
    envIntensity: opts.envIntensity ?? 1,
    envRotationY: opts.envRotationY ?? 0,
    scalarSkyRadiance: opts.scalarSkyRadiance ?? [0, 0, 0],
    hasDirectionalEnvironment,
    frameSeed: opts.frameSeed,
    triIntersectEpsilon: opts.triIntersectEpsilon ?? 1e-5,
    transmittedInterfaceBudget: opts.transmittedInterfaceBudget
      ?? RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
    bvhMode: mode === 'tlas' ? 1 : 0,
    tlasNodeCount: opts.tlasNodeCount ?? 0,
    emitterCount: emitterLayout.count,
    lightCount: opts.lightCount ?? 0,
    emitterDataWordOffset: emitterLayout.dataWordOffset,
    emitterAliasWordOffset: emitterLayout.aliasWordOffset,
    dims,
  });

  const hasAtlas = opts.materialTextureAtlasView != null;
  const hasAtlasMeta = opts.materialMapMetaTextureView != null;
  if (hasAtlas !== hasAtlasMeta) {
    throw new Error(
      '[RCDispatcher] materialTextureAtlasView and materialMapMetaTextureView must be supplied together.',
    );
  }

  const bvhNodes = validateStorageBinding(
    opts.device, opts.bvhNodesBuf, 'bvhNodesBuf', 32, GPUBufferUsage.STORAGE,
    opts.bvhNodesOffset, opts.bvhNodesSize,
  );
  const bvhIndices = validateStorageBinding(
    opts.device, opts.bvhIndicesBuf, 'bvhIndicesBuf', 16, GPUBufferUsage.STORAGE,
    opts.bvhIndicesOffset, opts.bvhIndicesSize,
  );
  const bvhPositions = validateStorageBinding(
    opts.device, opts.bvhPositionsBuf, 'bvhPositionsBuf', 16, GPUBufferUsage.STORAGE,
    opts.bvhPositionsOffset, opts.bvhPositionsSize,
  );
  const bvhNormals = validateStorageBinding(
    opts.device, opts.bvhNormalsBuf, 'bvhNormalsBuf', 16, GPUBufferUsage.STORAGE,
    opts.bvhNormalsOffset, opts.bvhNormalsSize,
  );
  if (bvhPositions.size !== bvhNormals.size) {
    throw new Error('[RCDispatcher] bvhPositionsBuf and bvhNormalsBuf must have equal vertex capacity.');
  }

  const cascadeBufsAreArray: boolean = Array.isArray(opts.cascadeBufs);
  if (!cascadeBufsAreArray || opts.cascadeBufs.length !== dims.length) {
    throw new Error(`[RCDispatcher] cascadeBufs must contain exactly ${dims.length} buffers.`);
  }
  const cascadeBuffers = opts.cascadeBufs.map((buffer, index) => {
    const dim = dims[index]!;
    const requiredBytes = dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays * 16;
    const checked = validateBuffer(buffer, `cascadeBufs[${index}]`, 16, GPUBufferUsage.STORAGE);
    if (checked.size < requiredBytes) {
      throw new Error(
        `[RCDispatcher] cascadeBufs[${index}].size=${checked.size}; requires ${requiredBytes} bytes.`,
      );
    }
    return checked;
  });

  const { arenaBytes, sourceBuffers } = validateSceneArenaSources(opts);
  const triCount = bvhIndices.size / 16;
  if (opts.triMaterialIdBuf.size / 4 < triCount) {
    throw new Error('[RCDispatcher] triMaterialIdBuf does not cover every geometry triangle.');
  }

  if (mode === 'tlas') {
    if ((opts.tlasNodeCount ?? 0) === 0) {
      throw new Error('[RCDispatcher] bvhMode="tlas" requires a positive tlasNodeCount.');
    }
    const tlasNodeCapacity = opts.tlasNodesBuf!.size / 32;
    if ((opts.tlasNodeCount ?? 0) > tlasNodeCapacity) {
      throw new Error('[RCDispatcher] tlasNodeCount exceeds tlasNodesBuf capacity.');
    }
    const instanceCount = opts.tlasBlasRootsBuf!.size / 4;
    if (
      opts.tlasInstanceIndicesBuf!.size / 4 !== instanceCount ||
      opts.tlasInstanceWorldToLocalBuf!.size / 64 !== instanceCount ||
      opts.tlasInstanceLocalToWorldBuf!.size / 64 !== instanceCount
    ) {
      throw new Error('[RCDispatcher] TLAS instance-index/root/transform capacities must match.');
    }
  } else {
    if ((opts.tlasNodeCount ?? 0) !== 0) {
      throw new Error('[RCDispatcher] merged bvhMode requires tlasNodeCount=0.');
    }
    if (
      opts.tlasNodesBuf != null || opts.tlasInstanceIndicesBuf != null ||
      opts.tlasBlasRootsBuf != null || opts.tlasInstanceWorldToLocalBuf != null ||
      opts.tlasInstanceLocalToWorldBuf != null
    ) {
      throw new Error('[RCDispatcher] TLAS buffers are only valid when bvhMode="tlas".');
    }
  }

  const emitterCount = emitterLayout.count;
  let emitterBindingBytes = 16;
  if (opts.emittersBuf == null) {
    if (
      emitterCount !== 0 || opts.emittersOffset != null || opts.emittersSize != null ||
      opts.emitterDataOffset != null || opts.emitterAliasOffset != null
    ) {
      throw new Error('[RCDispatcher] emitter count/range/alias fields require emittersBuf.');
    }
  } else {
    const emitterBuffer = validateBuffer(opts.emittersBuf, 'emittersBuf', 4, GPUBufferUsage.STORAGE);
    const offset = opts.emittersOffset ?? 0;
    assertNonnegativeSafeInteger(offset, 'emittersOffset');
    const storageAlignment = opts.device.limits?.minStorageBufferOffsetAlignment;
    if (typeof storageAlignment === 'number' && offset % storageAlignment !== 0) {
      throw new Error(`[RCDispatcher] emittersOffset must satisfy minStorageBufferOffsetAlignment=${storageAlignment}.`);
    }
    emitterBindingBytes = opts.emittersSize ?? (emitterBuffer.size - offset);
    assertNonnegativeSafeInteger(emitterBindingBytes, 'emittersSize');
    if (emitterBindingBytes < 16 || offset > emitterBuffer.size - emitterBindingBytes) {
      throw new Error('[RCDispatcher] emitter sampling binding range must be in bounds and at least 16 bytes.');
    }
    const dataOffset = emitterLayout.dataOffset;
    const aliasOffset = emitterLayout.aliasOffset;
    assertNonnegativeSafeInteger(dataOffset, 'emitterDataOffset');
    assertNonnegativeSafeInteger(aliasOffset, 'emitterAliasOffset');
    if (dataOffset % 4 !== 0 || aliasOffset % 4 !== 0) {
      throw new Error('[RCDispatcher] emitter data/alias offsets must be 4-byte aligned.');
    }
    const dataBytes = emitterLayout.dataBytes;
    const aliasBytes = emitterLayout.aliasBytes;
    if (dataOffset > emitterBindingBytes - dataBytes || aliasOffset > emitterBindingBytes - aliasBytes) {
      throw new Error(
        `[RCDispatcher] emitterCount=${emitterCount} requires ${dataBytes} data bytes and ` +
        `${aliasBytes} alias bytes inside the ${emitterBindingBytes}-byte bound window.`,
      );
    }
    if (emitterCount > 0 && dataOffset < aliasOffset + aliasBytes && aliasOffset < dataOffset + dataBytes) {
      throw new Error('[RCDispatcher] emitter data and alias ranges must not overlap.');
    }
  }

  const lightCount = opts.lightCount ?? 0;
  let lightsBindingBytes = RC_LIGHTS_BUFFER_BYTES;
  if (opts.lightsBuf == null) {
    if (lightCount !== 0 || opts.lightsOffset != null || opts.lightsSize != null) {
      throw new Error('[RCDispatcher] light count/range fields require lightsBuf.');
    }
  } else {
    const lights = validateBuffer(opts.lightsBuf, 'lightsBuf', 16, GPUBufferUsage.STORAGE);
    const offset = opts.lightsOffset ?? 0;
    assertNonnegativeSafeInteger(offset, 'lightsOffset');
    const storageAlignment = opts.device.limits?.minStorageBufferOffsetAlignment;
    if (typeof storageAlignment === 'number' && offset % storageAlignment !== 0) {
      throw new Error(`[RCDispatcher] lightsOffset must satisfy minStorageBufferOffsetAlignment=${storageAlignment}.`);
    }
    const requiredBytes = RC_LIGHT_HEADER_BYTES
      + lightCount * (RC_LIGHT_STRIDE_BYTES + RC_ALIAS_STRIDE_BYTES);
    if (!Number.isSafeInteger(requiredBytes)) {
      throw new Error('[RCDispatcher] runtime light buffer size exceeds Number.MAX_SAFE_INTEGER.');
    }
    lightsBindingBytes = opts.lightsSize ?? (lights.size - offset);
    assertNonnegativeSafeInteger(lightsBindingBytes, 'lightsSize');
    if (lightsBindingBytes !== requiredBytes || offset > lights.size - lightsBindingBytes) {
      throw new Error(
        `[RCDispatcher] lightCount=${lightCount} requires an exact ${requiredBytes}-byte bound range; ` +
        `received ${lightsBindingBytes} bytes.`,
      );
    }
  }

  const limits = opts.device.limits;
  assertLimit(limits?.maxStorageBuffersPerShaderStage, RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE, 'maxStorageBuffersPerShaderStage');
  assertLimit(limits?.maxSampledTexturesPerShaderStage, 5, 'maxSampledTexturesPerShaderStage');
  assertLimit(limits?.maxSamplersPerShaderStage, 1, 'maxSamplersPerShaderStage');
  assertLimit(limits?.maxUniformBufferBindingSize, RC_CASCADE_UNIFORM_BYTES, 'maxUniformBufferBindingSize');
  assertLimit(limits?.maxComputeInvocationsPerWorkgroup, 64, 'maxComputeInvocationsPerWorkgroup');
  assertLimit(limits?.maxComputeWorkgroupSizeX, 64, 'maxComputeWorkgroupSizeX');

  const directStorageSizes = [
    bvhNodes.size, bvhIndices.size, bvhPositions.size, bvhNormals.size,
    ...cascadeBuffers.map(buffer => buffer.size),
    arenaBytes, emitterBindingBytes, lightsBindingBytes, 80,
  ];
  const maxStorageBinding = Math.max(...directStorageSizes);
  assertLimit(limits?.maxStorageBufferBindingSize, maxStorageBinding, 'maxStorageBufferBindingSize');

  const allBufferSizes = [
    ...sourceBuffers.map(buffer => buffer.size),
    bvhNodes.buffer.size, bvhIndices.buffer.size,
    bvhPositions.buffer.size, bvhNormals.buffer.size,
    ...cascadeBuffers.map(buffer => buffer.size), arenaBytes,
    ...(opts.emittersBuf ? [opts.emittersBuf.size] : []),
    ...(opts.lightsBuf ? [opts.lightsBuf.size] : []),
  ];
  assertLimit(limits?.maxBufferSize, Math.max(...allBufferSizes), 'maxBufferSize');

  const maxDispatchX = Math.max(
    ...dims.map(dim => Math.ceil(dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays / 64)),
  );
  assertLimit(limits?.maxComputeWorkgroupsPerDimension, maxDispatchX, 'maxComputeWorkgroupsPerDimension');
}

function sameVec3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function destroyOwnedResource(
  resource: { destroy(): void } | null | undefined,
): void {
  try { resource?.destroy(); } catch { /* continue releasing independent owners */ }
}

function sameBufferArray(a: readonly GPUBuffer[], b: readonly GPUBuffer[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Single-source field registry for the binding signature (D16-8).
 *
 * Each entry captures one binding-relevant field: how to extract it from
 * `RCDispatchOptsRaw` (applying the default a missing/optional field falls back
 * to), and how to compare two captured values. `bindingSignature` builds the
 * cached snapshot by iterating this list; `sameBindingSignature` diffs by
 * iterating the same list. Adding/removing a binding is a **single-site edit**
 * here — the two functions can no longer drift out of sync (the H1/H41
 * offset-drift class this replaces). `bindingFieldRegistryFields` is exported
 * for a pin test asserting the exact field set is unchanged.
 *
 * `kind` selects the comparator: `'ref'` = identity (`===`, incl. GPUBuffer /
 * texture view / sampler / string), `'vec3'` = per-component tuple equality,
 * `'bufferArray'` = ordered GPUBuffer[] equality.
 */
type BindingFieldKind = 'ref' | 'vec3' | 'bufferArray';

interface BindingField<K extends keyof DispatchBindingSignature> {
  readonly key: K;
  readonly kind: BindingFieldKind;
  readonly extract: (opts: RCDispatchOptsRaw) => DispatchBindingSignature[K];
}

function bf<K extends keyof DispatchBindingSignature>(
  key: K,
  kind: BindingFieldKind,
  extract: (opts: RCDispatchOptsRaw) => DispatchBindingSignature[K],
): BindingField<K> {
  return { key, kind, extract };
}

// One entry per binding-relevant field. ORDER + membership define the signature.
const BINDING_FIELDS: readonly BindingField<keyof DispatchBindingSignature>[] = [
  bf('device', 'ref', (o) => o.device),
  bf('bvhMode', 'ref', (o) => o.bvhMode ?? 'merged'),
  bf('bvhNodesBuf', 'ref', (o) => o.bvhNodesBuf),
  bf('bvhNodesOffset', 'ref', (o) => o.bvhNodesOffset ?? 0),
  bf('bvhNodesSize', 'ref', (o) => o.bvhNodesSize ?? null),
  bf('bvhIndicesBuf', 'ref', (o) => o.bvhIndicesBuf),
  bf('bvhIndicesOffset', 'ref', (o) => o.bvhIndicesOffset ?? 0),
  bf('bvhIndicesSize', 'ref', (o) => o.bvhIndicesSize ?? null),
  bf('bvhPositionsBuf', 'ref', (o) => o.bvhPositionsBuf),
  bf('bvhPositionsOffset', 'ref', (o) => o.bvhPositionsOffset ?? 0),
  bf('bvhPositionsSize', 'ref', (o) => o.bvhPositionsSize ?? null),
  bf('bvhNormalsBuf', 'ref', (o) => o.bvhNormalsBuf),
  bf('bvhNormalsOffset', 'ref', (o) => o.bvhNormalsOffset ?? 0),
  bf('bvhNormalsSize', 'ref', (o) => o.bvhNormalsSize ?? null),
  bf('cascadeBufs', 'bufferArray', (o) => [...o.cascadeBufs]),
  bf('probeOriginWorld', 'vec3', (o) => [...o.probeOriginWorld] as [number, number, number]),
  bf('roomSize', 'vec3', (o) => [...o.roomSize] as [number, number, number]),
  bf('envTextureView', 'ref', (o) => o.envTextureView ?? null),
  bf('envSampler', 'ref', (o) => o.envSampler ?? null),
  bf('materialTextureAtlasView', 'ref', (o) => o.materialTextureAtlasView ?? null),
  bf('materialMapMetaTextureView', 'ref', (o) => o.materialMapMetaTextureView ?? null),
  bf('bvhTangentTextureView', 'ref', (o) => o.bvhTangentTextureView ?? null),
  bf('bvhVertexColorTextureView', 'ref', (o) => o.bvhVertexColorTextureView ?? null),
  bf('emittersBuf', 'ref', (o) => o.emittersBuf ?? null),
  bf('emittersOffset', 'ref', (o) => o.emittersOffset ?? 0),
  bf('emittersSize', 'ref', (o) => o.emittersSize ?? null),
  bf('emitterDataOffset', 'ref', (o) => resolveEmitterLayout(o).dataOffset),
  bf('emitterAliasOffset', 'ref', (o) => resolveEmitterLayout(o).aliasOffset),
  bf('lightsBuf', 'ref', (o) => o.lightsBuf ?? null),
  bf('lightsOffset', 'ref', (o) => o.lightsOffset ?? 0),
  bf('lightsSize', 'ref', (o) => o.lightsSize ?? null),
] as const;

/** The binding-signature field keys, in order — exported for the pin test. */
export const bindingFieldRegistryFields: readonly (keyof DispatchBindingSignature)[] =
  BINDING_FIELDS.map((f) => f.key);

function bindingSignature(opts: RCDispatchOptsRaw): DispatchBindingSignature {
  // Iterate the registry so the snapshot always covers exactly BINDING_FIELDS.
  const sig = {} as Record<keyof DispatchBindingSignature, unknown>;
  for (const field of BINDING_FIELDS) {
    sig[field.key] = field.extract(opts);
  }
  return sig as unknown as DispatchBindingSignature;
}

function sameBindingSignature(
  a: DispatchBindingSignature | null,
  b: DispatchBindingSignature,
): boolean {
  if (a == null) return false;
  for (const field of BINDING_FIELDS) {
    const av = a[field.key];
    const bv = b[field.key];
    switch (field.kind) {
      case 'vec3':
        if (!sameVec3(
          av as readonly [number, number, number],
          bv as readonly [number, number, number],
        )) return false;
        break;
      case 'bufferArray':
        if (!sameBufferArray(av as readonly GPUBuffer[], bv as readonly GPUBuffer[])) return false;
        break;
      default:
        if (av !== bv) return false;
        break;
    }
  }
  return true;
}

// ─── RCDispatcher class ───────────────────────────────────────────────────────

/**
 * Manages the RC cascade compute pipeline (raw WebGPU).
 *
 * Lifecycle:
 *   1. Construct with `new RCDispatcher()`.
 *   2. Call `dispatchFrameRaw(opts)` each frame.  Handles lazy init internally.
 *   3. Call `dispose()` to release GPU resources.
 *
 * History: this used to expose a THREE-tied `dispatchFrame(opts: RCDispatchOpts)`
 * that reached into `StorageBufferAttribute.__gpuBuffer` (renderer-internal) to
 * extract raw `GPUBuffer` handles. That path was dropped 2026-05-18 once
 * `RCSubsystem` was confirmed to be the only consumer and was already calling
 * the raw entry directly.
 */
export class RCDispatcher {
  private _handles: DispatchHandles | null = null;
  private _bindingSignature: DispatchBindingSignature | null = null;
  private _castShaderModule:  GPUShaderModule | null = null;
  private _mergeShaderModule: GPUShaderModule | null = null;
  private _shaderDevice: GPUDevice | null = null;
  private _lastError: Error | null = null;
  /** B3b (2026-05-19) — per-instance cascade dimensions. Defaults to the
   *  Cornell-tuned `CASCADE_DIMS`; hosts override via constructor for
   *  non-Cornell aspect ratios / scene scales. */
  private readonly _cascadeDims: readonly CascadeDim[];

  constructor(cascadeDims: readonly CascadeDim[] = CASCADE_DIMS) {
    this._cascadeDims = validateCascadeDims(cascadeDims, 'RCDispatcher cascadeDims');
  }

  get lastError(): Error | null {
    return this._lastError;
  }

  /**
   * Dispatch the cascade compute pipeline for one frame.
   * Pipelines and bind groups are compiled/created lazily on the first call
   * (or after `dispose()`).
   *
   * Bind groups are reused while binding-relevant inputs are stable. When the
   * raw caller swaps buffer sets, env bindings, bvhMode, device, cascade output
   * buffers, or cascade bounds, the dispatcher releases cached bind groups and
   * rebuilds them before dispatching.
   */
  dispatchFrameRaw(opts: RCDispatchOptsRaw): void {
    validateDispatchOptsRaw(opts, this._cascadeDims);
    const device = opts.device;
    const signature = bindingSignature(opts);
    let handles = this._handles;
    let needsBuild = handles == null || !sameBindingSignature(this._bindingSignature, signature);
    if (!needsBuild && handles && !this._refreshSceneArenaSources(handles, opts)) {
      needsBuild = true;
    }
    if (needsBuild) {
      try {
        const candidate = this._buildHandlesRaw(device, opts);
        const previous = this._handles;
        this._handles = candidate;
        this._bindingSignature = signature;
        this._lastError = null;
        handles = candidate;
        this._releaseHandleResources(previous);
      } catch (err: unknown) {
        const error = err instanceof Error ? err : new Error(String(err));
        this._lastError = error;
        throw new Error(`[RCDispatcher] buildHandlesRaw failed: ${error.message}`);
      }
    }
    if (!handles) throw new Error('[RCDispatcher] internal error: dispatch handles were not published.');
    this._lastError = null;

    // Update per-frame uniforms for each cast pass.
    const dims = this._cascadeDims;
    const emitterLayout = resolveEmitterLayout(opts);
    for (let k = 0; k < dims.length; k++) {
      const pass = handles.castPasses[k]!;
      buildCascadeUniformDataInto(pass.cascadeParamsRaw, k, {
        probeOriginWorld: opts.probeOriginWorld,
        roomSize:         opts.roomSize,
        sunDir:           opts.sunDirection,
        sunColor:         opts.sunColor,
        sunCastShadowDisabled: opts.sunCastShadowDisabled === true,
        sunAngularRadius: opts.sunAngularRadius ?? 0,
        envIntensity:     opts.envIntensity ?? 1,
        envRotationY:     opts.envRotationY ?? 0,
        scalarSkyRadiance: opts.scalarSkyRadiance ?? [0, 0, 0],
        hasDirectionalEnvironment:
          opts.hasDirectionalEnvironment ??
          (opts.envTextureView != null && opts.envSampler != null),
        frameSeed:        opts.frameSeed,
        triIntersectEpsilon: opts.triIntersectEpsilon ?? 1e-5,
        transmittedInterfaceBudget: opts.transmittedInterfaceBudget
          ?? RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
        bvhMode:          opts.bvhMode === 'tlas' ? 1 : 0,
        tlasNodeCount:    opts.tlasNodeCount ?? 0,
        emitterCount:     emitterLayout.count,
        lightCount:       opts.lightCount ?? 0,
        emitterDataWordOffset: emitterLayout.dataWordOffset,
        emitterAliasWordOffset: emitterLayout.aliasWordOffset,
        dims,
      });
      device.queue.writeBuffer(pass.cascadeParamsBuf, 0, pass.cascadeParamsRaw.buffer);
    }

    // Encode compute commands.
    const commandEncoder = device.createCommandEncoder({ label: 'RCDispatcher' });
    const encodedArenaCopies: SceneArenaCopy[] = [];
    for (const copy of handles.sceneArenaCopies) {
      if (!copy.dirty) continue;
      commandEncoder.copyBufferToBuffer(
        copy.source,
        0,
        handles.sceneArenaBuf,
        copy.destinationOffset,
        copy.size,
      );
      encodedArenaCopies.push(copy);
    }
    const passEncoder = commandEncoder.beginComputePass({ label: 'rc-cascade' });

    // Cast passes C0 → C(N-1).
    for (let k = 0; k < dims.length; k++) {
      const pass = handles.castPasses[k]!;
      passEncoder.setPipeline(pass.pipeline);
      passEncoder.setBindGroup(0, handles.castBindGroups[k]);
      passEncoder.dispatchWorkgroups(pass.dispatchX);
    }

    // Merge passes C3 → C0 (bottom-up).
    for (let m = 0; m < handles.mergePasses.length; m++) {
      const pass = handles.mergePasses[m]!;
      passEncoder.setPipeline(pass.pipeline);
      passEncoder.setBindGroup(0, handles.mergeBindGroups[m]);
      passEncoder.dispatchWorkgroups(pass.dispatchX);
    }

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    for (const copy of encodedArenaCopies) copy.dirty = false;
  }

  /** Drop cached bind groups so the next dispatch captures fresh caller buffers.
   *  Also nulls the cached shader modules: `invalidateBindings` fires on a device
   *  change, and a shader module is bound to the device that created it. Reusing an
   *  old-device module across a device swap raises a cross-device validation error,
   *  so the modules must be recompiled on the fresh device by the next
   *  `_buildHandlesRaw`. */
  invalidateBindings(): void {
    this._releaseHandles();
    this._lastError = null;
    this._castShaderModule  = null;
    this._mergeShaderModule = null;
    this._shaderDevice = null;
  }

  /** Release all GPU resources. Next `dispatchFrame()` will re-initialize. */
  dispose(): void {
    this._releaseHandles();
    this._lastError = null;
    this._castShaderModule  = null;
    this._mergeShaderModule = null;
    this._shaderDevice = null;
  }

  // ─── Private helpers ────────────────────────────────────────────────────────

  /** Cast pass: BVH+mat SSBOs, cascade out, env, uniforms, optional TLAS (C2), analytic lights (A7), material atlas (RC mapped emitters). */
  private _castBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: 'rc-cast-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rc_emitters
        // Runtime-sized analytic/directional lights plus alias table.
        { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rc_lights
        // The material atlas is a packed r32uint array. Metadata remains
        // rgba32float and carries the logical-layer address directory.
        { binding: 16, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d-array' } }, // rc_materialTextureAtlas
        { binding: 17, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },       // rc_materialMapMeta
        { binding: 18, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // rc_geom_normal
        { binding: 19, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },       // rc_geom_tangent
        { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },       // rc_geom_vertex_color
      ],
    });
  }

  private _dummyStorageBuffer(
    device: GPUDevice,
    label: string,
    size: number,
    own: OwnGpuResource,
  ): GPUBuffer {
    return own(device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
  }

  private _releaseHandleResources(handles: DispatchHandles | null): void {
    if (!handles) return;
    for (const resource of handles.ownedResources) destroyOwnedResource(resource);
  }

  private _releaseHandles(): void {
    const handles = this._handles;
    this._handles = null;
    this._bindingSignature = null;
    this._releaseHandleResources(handles);
  }

  /** Build bind group layout for a merge pass (3 entries: upper + lower cascades + uniforms). */
  private _mergeBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: 'rc-merge-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // upperCascade
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // lowerCascade (rw)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // MergeUniforms
      ],
    });
  }

  /**
   * Resolve the env binding for the raw (THREE-free) path. When the caller
   * supplies both `envTextureView` and `envSampler`, use them. Otherwise
   * create a 1×1 black placeholder.
   */
  private _resolveEnvBindingRaw(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    own: OwnGpuResource,
  ): {
    envTextureView: GPUTextureView;
    envSampler: GPUSampler;
    placeholderEnvTexture?: GPUTexture;
  } {
    if (opts.envTextureView && opts.envSampler) {
      return {
        envTextureView: opts.envTextureView,
        envSampler: opts.envSampler,
      };
    }
    const placeholderTex = own(device.createTexture({
      label:  'rc-env-placeholder',
      size:   [1, 1],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
    device.queue.writeTexture(
      { texture: placeholderTex },
      new Uint8Array([0, 0, 0, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    return {
      envTextureView: placeholderTex.createView({ label: 'rc-env-placeholder-view' }),
      envSampler: device.createSampler({ label: 'rc-env-placeholder-sampler' }),
      placeholderEnvTexture: placeholderTex,
    };
  }

  /**
   * Resolve optional material-atlas bindings for RC's material-backed emitter
   * NEE. The placeholder metadata advertises zero logical layers so shader
   * helpers fall back to scalar EmitterTri.Le when no atlas was supplied.
   */
  private _resolveMaterialAtlasBindingRaw(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    own: OwnGpuResource,
  ): {
    materialTextureAtlasView: GPUTextureView;
    materialMapMetaTextureView: GPUTextureView;
    placeholderMaterialAtlasTexture?: GPUTexture;
    placeholderMaterialMetaTexture?: GPUTexture;
  } {
    if (opts.materialTextureAtlasView && opts.materialMapMetaTextureView) {
      return {
        materialTextureAtlasView: opts.materialTextureAtlasView,
        materialMapMetaTextureView: opts.materialMapMetaTextureView,
      };
    }
    const atlasTexture = own(device.createTexture({
      label: 'rc-material-atlas-placeholder',
      size: { width: 1, height: 1, depthOrArrayLayers: 1 },
      format: 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
    device.queue.writeTexture(
      { texture: atlasTexture },
      new Uint32Array([0]),
      { bytesPerRow: 4, rowsPerImage: 1 },
      { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
    const metaTexture = own(device.createTexture({
      label: 'rc-material-meta-placeholder',
      size: { width: 4, height: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
    const meta = new Float32Array(4 * 4);
    meta.set([
      3, 0, 0, 157,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ]);
    device.queue.writeTexture(
      { texture: metaTexture },
      // ABI-v3 header with zero materials, triangles, and logical layers.
      meta,
      { bytesPerRow: 64 },
      { width: 4, height: 1 },
    );
    return {
      materialTextureAtlasView: atlasTexture.createView({ label: 'rc-material-atlas-placeholder-view', dimension: '2d-array' }),
      materialMapMetaTextureView: metaTexture.createView({ label: 'rc-material-meta-placeholder-view' }),
      placeholderMaterialAtlasTexture: atlasTexture,
      placeholderMaterialMetaTexture: metaTexture,
    };
  }

  private _resolveTangentTextureBindingRaw(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    own: OwnGpuResource,
  ): {
    bvhTangentTextureView: GPUTextureView;
    placeholderTangentTexture?: GPUTexture;
  } {
    if (opts.bvhTangentTextureView) {
      return { bvhTangentTextureView: opts.bvhTangentTextureView };
    }
    const tangentTexture = own(device.createTexture({
      label: 'rc-bvh-tangent-placeholder',
      size: { width: 1, height: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
    device.queue.writeTexture(
      { texture: tangentTexture },
      new Float32Array([0, 0, 0, 0]),
      { bytesPerRow: 16 },
      { width: 1, height: 1 },
    );
    return {
      bvhTangentTextureView: tangentTexture.createView({ label: 'rc-bvh-tangent-placeholder-view' }),
      placeholderTangentTexture: tangentTexture,
    };
  }

  private _resolveVertexColorTextureBindingRaw(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    own: OwnGpuResource,
  ): {
    bvhVertexColorTextureView: GPUTextureView;
    placeholderVertexColorTexture?: GPUTexture;
  } {
    if (opts.bvhVertexColorTextureView) {
      return { bvhVertexColorTextureView: opts.bvhVertexColorTextureView };
    }
    const vertexColorTexture = own(device.createTexture({
      label: 'rc-bvh-vertex-color-placeholder',
      size: { width: 1, height: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
    device.queue.writeTexture(
      { texture: vertexColorTexture },
      new Float32Array([1, 1, 1, 1]),
      { bytesPerRow: 16 },
      { width: 1, height: 1 },
    );
    return {
      bvhVertexColorTextureView: vertexColorTexture.createView({ label: 'rc-bvh-vertex-color-placeholder-view' }),
      placeholderVertexColorTexture: vertexColorTexture,
    };
  }

  /**
   * Build all pipelines and bind groups (one-time setup).
   * Called lazily on first `dispatchFrameRaw()`.
   *
   * Takes {@link RCDispatchOptsRaw} (raw GPU types). The W8 Phase 1B
   * refactor (2026-05-18) extracted this from a THREE-tied builder; the
   * legacy entry was dropped the same day.
   */
  private _buildHandlesRaw(device: GPUDevice, opts: RCDispatchOptsRaw): DispatchHandles {
    const ownedResources: OwnedGpuResource[] = [];
    const own: OwnGpuResource = resource => {
      ownedResources.push(resource);
      return resource;
    };

    try {
      const canReuseModules = this._shaderDevice === device;
      const castShaderModule = canReuseModules && this._castShaderModule
        ? this._castShaderModule
        : device.createShaderModule({ label: 'rc-probe-ray-cast', code: PROBE_RAY_CAST_WGSL });
      const mergeShaderModule = canReuseModules && this._mergeShaderModule
        ? this._mergeShaderModule
        : device.createShaderModule({ label: 'rc-cascade-merge', code: CASCADE_MERGE_WGSL });

      const castBGL  = this._castBindGroupLayout(device);
      const mergeBGL = this._mergeBindGroupLayout(device);
      const castPipelineLayout  = device.createPipelineLayout({ bindGroupLayouts: [castBGL] });
      const mergePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [mergeBGL] });

      const bvhBindings = this._resolveBvhBindings(device, opts, own);
      const { envTextureView, envSampler, placeholderEnvTexture } =
        this._resolveEnvBindingRaw(device, opts, own);
      const {
        materialTextureAtlasView,
        materialMapMetaTextureView,
        placeholderMaterialAtlasTexture,
        placeholderMaterialMetaTexture,
      } = this._resolveMaterialAtlasBindingRaw(device, opts, own);
      const {
        bvhTangentTextureView,
        placeholderTangentTexture,
      } = this._resolveTangentTextureBindingRaw(device, opts, own);
      const {
        bvhVertexColorTextureView,
        placeholderVertexColorTexture,
      } = this._resolveVertexColorTextureBindingRaw(device, opts, own);

      const { castPasses, castBindGroups } = this._buildCastPasses(
        device, opts, castBGL, castPipelineLayout, castShaderModule, bvhBindings,
        envTextureView, envSampler, materialTextureAtlasView, materialMapMetaTextureView,
        bvhTangentTextureView, bvhVertexColorTextureView, own,
      );
      const { mergePasses, mergeBindGroups } = this._buildMergePasses(
        device, opts, mergeBGL, mergePipelineLayout, mergeShaderModule, own,
      );

      const candidate: DispatchHandles = {
        castPasses,
        mergePasses,
        envTextureView,
        envSampler,
        castBindGroups,
        mergeBindGroups,
        sceneArenaBuf: bvhBindings.sceneArenaBuf,
        sceneArenaCopies: bvhBindings.sceneArenaCopies,
        materialArenaVersion: opts.materialArenaVersion ?? 0,
        tlasArenaVersion: opts.tlasArenaVersion ?? 0,
        ownedResources,
        ...(placeholderEnvTexture ? { placeholderEnvTexture } : {}),
        ...(placeholderMaterialAtlasTexture ? { placeholderMaterialAtlasTexture } : {}),
        ...(placeholderMaterialMetaTexture ? { placeholderMaterialMetaTexture } : {}),
        ...(placeholderTangentTexture ? { placeholderTangentTexture } : {}),
        ...(placeholderVertexColorTexture ? { placeholderVertexColorTexture } : {}),
      };

      // Publish device-bound shader cache state only after every fallible build step succeeds.
      this._castShaderModule = castShaderModule;
      this._mergeShaderModule = mergeShaderModule;
      this._shaderDevice = device;
      return candidate;
    } catch (error) {
      for (const resource of ownedResources) destroyOwnedResource(resource);
      throw error;
    }
  }

  /**
   * Pack material metadata and the five TLAS arrays behind one storage binding.
   * The 64-byte header stores `(wordOffset, elementCount)` pairs in this order:
   * materials, triangle-material ids, TLAS nodes, instance ids, BLAS roots,
   * world-to-local columns, local-to-world columns. The last pair is reserved.
   *
   * Only dirty subranges are recopied: replacement identities are detected
   * automatically and in-place writers opt in through the two version fields.
   */
  private _buildSceneArena(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    own: OwnGpuResource,
  ): { sceneArenaBuf: GPUBuffer; sceneArenaCopies: SceneArenaCopy[] } {
    const header = new Uint32Array(RC_SCENE_ARENA_HEADER_WORDS);
    const copies: SceneArenaCopy[] = [];
    let cursor = RC_SCENE_ARENA_HEADER_BYTES;

    const append = (
      pairIndex: number,
      label: string,
      source: GPUBuffer,
      elementStride: number,
    ): void => {
      const size = source.size;
      if (!Number.isSafeInteger(size) || size < elementStride || size % 4 !== 0) {
        throw new Error(
          `[RCDispatcher] ${label} has invalid byte size ${String(size)}; ` +
          `expected a positive multiple of 4 and at least ${elementStride} bytes.`,
        );
      }
      if (
        typeof source.usage === 'number' &&
        (source.usage & GPUBufferUsage.COPY_SRC) === 0
      ) {
        throw new Error(
          `[RCDispatcher] ${label} must include GPUBufferUsage.COPY_SRC ` +
          'so RC can pack its portable eight-binding scene arena.',
        );
      }
      header[pairIndex * 2] = cursor / 4;
      header[pairIndex * 2 + 1] = Math.floor(size / elementStride);
      copies.push({
        source,
        destinationOffset: cursor,
        size,
        category: pairIndex <= 1 ? 'material' : 'tlas',
        dirty: true,
      });
      cursor += size;
    };

    const markEmpty = (pairIndex: number): void => {
      header[pairIndex * 2] = cursor / 4;
      header[pairIndex * 2 + 1] = 0;
    };

    append(0, 'materialsBuf', opts.materialsBuf, 64);
    append(1, 'triMaterialIdBuf', opts.triMaterialIdBuf, 4);

    const tlasBuffers = [
      opts.tlasNodesBuf,
      opts.tlasInstanceIndicesBuf,
      opts.tlasBlasRootsBuf,
      opts.tlasInstanceWorldToLocalBuf,
      opts.tlasInstanceLocalToWorldBuf,
    ] as const;
    if (opts.bvhMode === 'tlas') {
      if (tlasBuffers.some((buffer) => buffer == null)) {
        throw new Error(
          '[RCDispatcher] bvhMode="tlas" requires all five TLAS buffers.',
        );
      }
      append(2, 'tlasNodesBuf', tlasBuffers[0]!, 32);
      append(3, 'tlasInstanceIndicesBuf', tlasBuffers[1]!, 4);
      append(4, 'tlasBlasRootsBuf', tlasBuffers[2]!, 4);
      append(5, 'tlasInstanceWorldToLocalBuf', tlasBuffers[3]!, 16);
      append(6, 'tlasInstanceLocalToWorldBuf', tlasBuffers[4]!, 16);
    } else {
      for (let pairIndex = 2; pairIndex <= 6; pairIndex += 1) {
        markEmpty(pairIndex);
      }
    }

    const sceneArenaBuf = own(device.createBuffer({
      label: 'rc-scene-arena',
      size: cursor,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    device.queue.writeBuffer(
      sceneArenaBuf,
      0,
      header.buffer,
      header.byteOffset,
      header.byteLength,
    );
    return { sceneArenaBuf, sceneArenaCopies: copies };
  }

  /**
   * Refresh arena source identities without rebuilding bind groups. Returns
   * false only when a replacement changes the packed shape and needs a new
   * arena allocation. Static frames leave every copy clean.
   */
  private _refreshSceneArenaSources(
    handles: DispatchHandles,
    opts: RCDispatchOptsRaw,
  ): boolean {
    const desired: GPUBuffer[] = [opts.materialsBuf, opts.triMaterialIdBuf];
    if (opts.bvhMode === 'tlas') {
      const tlas = [
        opts.tlasNodesBuf,
        opts.tlasInstanceIndicesBuf,
        opts.tlasBlasRootsBuf,
        opts.tlasInstanceWorldToLocalBuf,
        opts.tlasInstanceLocalToWorldBuf,
      ] as const;
      if (tlas.some((buffer) => buffer == null)) return false;
      desired.push(tlas[0]!, tlas[1]!, tlas[2]!, tlas[3]!, tlas[4]!);
    }
    if (desired.length !== handles.sceneArenaCopies.length) return false;

    // Validate the complete replacement shape before mutating any retained copy.
    for (let index = 0; index < desired.length; index += 1) {
      if (desired[index]!.size !== handles.sceneArenaCopies[index]!.size) return false;
    }
    for (let index = 0; index < desired.length; index += 1) {
      const source = desired[index]!;
      const copy = handles.sceneArenaCopies[index]!;
      if (source !== copy.source) {
        copy.source = source;
        copy.dirty = true;
      }
    }

    const materialVersion = opts.materialArenaVersion ?? 0;
    if (materialVersion !== handles.materialArenaVersion) {
      for (const copy of handles.sceneArenaCopies) {
        if (copy.category === 'material') copy.dirty = true;
      }
      handles.materialArenaVersion = materialVersion;
    }
    const tlasVersion = opts.tlasArenaVersion ?? 0;
    if (tlasVersion !== handles.tlasArenaVersion) {
      for (const copy of handles.sceneArenaCopies) {
        if (copy.category === 'tlas') copy.dirty = true;
      }
      handles.tlasArenaVersion = tlasVersion;
    }
    return true;
  }

  /**
   * Resolve all BVH + TLAS + emitter + lights GPUBuffers, substituting
   * sized dummy placeholders for any that the caller omitted. Extracted from
   * `_buildHandlesRaw` (D13.1).
   */
  private _resolveBvhBindings(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    own: OwnGpuResource,
  ): {
    bvhBinding: GPUBufferBinding;
    indexBinding: GPUBufferBinding;
    positionBinding: GPUBufferBinding;
    normalBinding: GPUBufferBinding;
    sceneArenaBuf: GPUBuffer; sceneArenaCopies: SceneArenaCopy[];
    emittersBinding: GPUBufferBinding; lightsBinding: GPUBufferBinding;
  } {
    const bvhBinding = validateStorageBinding(
      device, opts.bvhNodesBuf, 'bvhNodesBuf', 32, GPUBufferUsage.STORAGE,
      opts.bvhNodesOffset, opts.bvhNodesSize,
    ).resource;
    const indexBinding = validateStorageBinding(
      device, opts.bvhIndicesBuf, 'bvhIndicesBuf', 16, GPUBufferUsage.STORAGE,
      opts.bvhIndicesOffset, opts.bvhIndicesSize,
    ).resource;
    const positionBinding = validateStorageBinding(
      device, opts.bvhPositionsBuf, 'bvhPositionsBuf', 16, GPUBufferUsage.STORAGE,
      opts.bvhPositionsOffset, opts.bvhPositionsSize,
    ).resource;
    const normalBinding = validateStorageBinding(
      device, opts.bvhNormalsBuf, 'bvhNormalsBuf', 16, GPUBufferUsage.STORAGE,
      opts.bvhNormalsOffset, opts.bvhNormalsSize,
    ).resource;
    const { sceneArenaBuf, sceneArenaCopies } = this._buildSceneArena(device, opts, own);
    // Raw emitter+alias window. count=0 never reads beyond the 16-byte dummy.
    const emittersBinding: GPUBufferBinding = opts.emittersBuf == null
      ? { buffer: this._dummyStorageBuffer(device, 'rc-emitters-dummy', 16, own), offset: 0, size: 16 }
      : {
          buffer: opts.emittersBuf,
          offset: opts.emittersOffset ?? 0,
          ...(opts.emittersSize == null ? {} : { size: opts.emittersSize }),
        };
    // Runtime-sized RCLight+alias window; absent scenes use a 16-byte header.
    const lightsBinding: GPUBufferBinding = opts.lightsBuf == null
      ? { buffer: this._dummyStorageBuffer(device, 'rc-lights-dummy', RC_LIGHTS_BUFFER_BYTES, own), offset: 0, size: RC_LIGHTS_BUFFER_BYTES }
      : {
          buffer: opts.lightsBuf,
          offset: opts.lightsOffset ?? 0,
          size: opts.lightsSize ?? opts.lightsBuf.size - (opts.lightsOffset ?? 0),
        };
    return {
      bvhBinding,
      indexBinding,
      positionBinding,
      normalBinding,
      sceneArenaBuf,
      sceneArenaCopies,
      emittersBinding,
      lightsBinding,
    };
  }

  /**
   * Build one cast pipeline + bind group + uniform buffer per cascade level.
   * Extracted from `_buildHandlesRaw` (D13.1).
   */
  private _buildCastPasses(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    castBGL: GPUBindGroupLayout,
    castPipelineLayout: GPUPipelineLayout,
    castShaderModule: GPUShaderModule,
    bvhBindings: ReturnType<typeof RCDispatcher.prototype._resolveBvhBindings>,
    envTextureView: GPUTextureView,
    envSampler: GPUSampler,
    materialTextureAtlasView: GPUTextureView,
    materialMapMetaTextureView: GPUTextureView,
    bvhTangentTextureView: GPUTextureView,
    bvhVertexColorTextureView: GPUTextureView,
    own: OwnGpuResource,
  ): { castPasses: CastPassHandles[]; castBindGroups: GPUBindGroup[] } {
    const {
      bvhBinding, indexBinding, positionBinding, normalBinding, sceneArenaBuf,
      emittersBinding, lightsBinding,
    } = bvhBindings;

    const castPasses: CastPassHandles[] = [];
    const castBindGroups: GPUBindGroup[] = [];
    const cascadeDims = this._cascadeDims;
    const emitterLayout = resolveEmitterLayout(opts);

    for (let k = 0; k < cascadeDims.length; k++) {
      const dim = cascadeDims[k]!;
      const totalRays = dim.probes[0] * dim.probes[1] * dim.probes[2] * dim.rays;

      const pipeline = device.createComputePipeline({
        label:  `rc-cast-C${k}`,
        layout: castPipelineLayout,
        compute: {
          module:     castShaderModule,
          entryPoint: 'probeRayCastKernel',
        },
      });

      // Per-pass uniform buffer for CascadeUniforms (40 words = 160 bytes).
      // WebGPU guarantees at least 16 KiB for a uniform binding, so this is
      // portable and avoids consuming one of the eight guaranteed storage slots.
      const cascadeParamsRaw = new Float32Array(40);
      buildCascadeUniformDataInto(cascadeParamsRaw, k, {
        probeOriginWorld: opts.probeOriginWorld,
        roomSize:         opts.roomSize,
        sunDir:           opts.sunDirection,
        sunColor:         opts.sunColor,
        sunCastShadowDisabled: opts.sunCastShadowDisabled === true,
        sunAngularRadius: opts.sunAngularRadius ?? 0,
        envIntensity:     opts.envIntensity ?? 1,
        envRotationY:     opts.envRotationY ?? 0,
        scalarSkyRadiance: opts.scalarSkyRadiance ?? [0, 0, 0],
        hasDirectionalEnvironment:
          opts.hasDirectionalEnvironment ??
          (opts.envTextureView != null && opts.envSampler != null),
        frameSeed:        opts.frameSeed,
        triIntersectEpsilon: opts.triIntersectEpsilon ?? 1e-5,
        transmittedInterfaceBudget: opts.transmittedInterfaceBudget
          ?? RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
        bvhMode:          opts.bvhMode === 'tlas' ? 1 : 0,
        tlasNodeCount:    opts.tlasNodeCount ?? 0,
        emitterCount:     emitterLayout.count,
        lightCount:       opts.lightCount ?? 0,
        emitterDataWordOffset: emitterLayout.dataWordOffset,
        emitterAliasWordOffset: emitterLayout.aliasWordOffset,
        dims:             cascadeDims,
      });
      const cascadeParamsBuf = own(device.createBuffer({
        label:  `rc-cast-C${k}-uniforms`,
        size:   cascadeParamsRaw.byteLength,
        usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      }));
      new Float32Array(cascadeParamsBuf.getMappedRange()).set(cascadeParamsRaw);
      cascadeParamsBuf.unmap();

      const cascadeBuf = opts.cascadeBufs[k]!;
      const bindGroup = device.createBindGroup({
        label:  `rc-cast-C${k}-bg`,
        layout: castBGL,
        entries: [
          { binding: 0, resource: bvhBinding },
          { binding: 1, resource: indexBinding },
          { binding: 2, resource: positionBinding },
          { binding: 3, resource: { buffer: sceneArenaBuf } },
          { binding: 5, resource: { buffer: cascadeBuf } },
          { binding: 6, resource: envTextureView },
          { binding: 7, resource: envSampler },
          { binding: 8, resource: { buffer: cascadeParamsBuf } },
          { binding: 14, resource: emittersBinding },
          { binding: 15, resource: lightsBinding },  // runtime RCLight + alias window
          { binding: 16, resource: materialTextureAtlasView },
          { binding: 17, resource: materialMapMetaTextureView },
          { binding: 18, resource: normalBinding },
          { binding: 19, resource: bvhTangentTextureView },
          { binding: 20, resource: bvhVertexColorTextureView },
        ],
      });

      castPasses.push({ pipeline, cascadeParamsBuf, cascadeParamsRaw, dispatchX: Math.ceil(totalRays / 64) });
      castBindGroups.push(bindGroup);
    }

    return { castPasses, castBindGroups };
  }

  /**
   * Build one merge pipeline + bind group + uniform buffer per cascade boundary
   * (bottom-up: C(N-2) → C0). Extracted from `_buildHandlesRaw` (D13.1).
   */
  private _buildMergePasses(
    device: GPUDevice,
    opts: RCDispatchOptsRaw,
    mergeBGL: GPUBindGroupLayout,
    mergePipelineLayout: GPUPipelineLayout,
    mergeShaderModule: GPUShaderModule,
    own: OwnGpuResource,
  ): { mergePasses: MergePassHandles[]; mergeBindGroups: GPUBindGroup[] } {
    const mergePasses: MergePassHandles[] = [];
    const mergeBindGroups: GPUBindGroup[] = [];
    const cascadeDims = this._cascadeDims;

    for (let lower = cascadeDims.length - 2; lower >= 0; lower--) {
      const lowerDim = cascadeDims[lower]!;
      const upperDim = cascadeDims[lower + 1]!;
      const totalLower = lowerDim.probes[0] * lowerDim.probes[1] * lowerDim.probes[2] * lowerDim.rays;

      const pipeline = device.createComputePipeline({
        label:  `rc-merge-${lower}→${lower + 1}`,
        layout: mergePipelineLayout,
        compute: {
          module:     mergeShaderModule,
          entryPoint: 'cascadeMergeKernel',
        },
      });

      // MergeUniforms buffer (20 floats = 80 bytes).
      const mergeRaw = buildMergeUniformData(lowerDim, upperDim, opts.probeOriginWorld, opts.roomSize);
      const cascadeParamsBuf = own(device.createBuffer({
        label:  `rc-merge-${lower}-uniforms`,
        size:   mergeRaw.byteLength,
        usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      }));
      new Float32Array(cascadeParamsBuf.getMappedRange()).set(mergeRaw);
      cascadeParamsBuf.unmap();

      const lowerBuf = opts.cascadeBufs[lower]!;
      const upperBuf = opts.cascadeBufs[lower + 1]!;
      const bindGroup = device.createBindGroup({
        label:  `rc-merge-${lower}-bg`,
        layout: mergeBGL,
        entries: [
          { binding: 0, resource: { buffer: upperBuf } },
          { binding: 1, resource: { buffer: lowerBuf } },
          { binding: 2, resource: { buffer: cascadeParamsBuf } },
        ],
      });

      mergePasses.push({ pipeline, cascadeParamsBuf, dispatchX: Math.ceil(totalLower / 64) });
      mergeBindGroups.push(bindGroup);
    }

    return { mergePasses, mergeBindGroups };
  }
}

// W8 follow-up cleanup (2026-05-18) — the `dispatchCascadePasses` /
// `disposeSharedDispatcher` module-level singleton wrappers were removed
// after grep verified zero production consumers (host code instantiates
// `RCDispatcher` directly now via HybridEngineRC.ts). The singletons
// violated CLAUDE.md Design Principle 2 ("the host owns lifecycle") and
// only existed as a backward-compat surface for legacy callers that no
// longer exist in this monorepo.
