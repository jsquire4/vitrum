// pt-webgpu capabilities builder (T3-B god-file split, 2026-07-20).
//
// Pure builder for the `EngineCapabilities` object the `PTEngineWebGPU.capabilities`
// getter previously constructed inline (~165 lines). Extracting it keeps the large
// documented literal out of the engine class body while preserving BYTE-IDENTICAL
// output: the getter now delegates to `ptWebgpuCapabilities(flags)`, passing the
// engine's resolved private-field values as a plain flags record. No logic change.
//
// The `#postDenoiser instanceof OIDNFinalDispatcher` test stays in the engine (it
// inspects a live instance); the caller passes the resolved boolean `isOidnFinal`.

import type {
  AnalyticShape,
  EngineCapabilities,
  Scene,
  SceneEmitter,
  ScenePrimitive,
} from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { PT_WEBGPU_SUPPORT } from './scene/uploadSceneBuffers.js';
import { PT_WEBGPU_LITE_MATERIALS } from './supportDetails.js';
import type { PtWebgpuTraceTier } from './traceTier.js';
import type { PtWebgpuBvhTraversalMode } from './gpuResources.js';
import type { PtWebgpuSamplingMode } from './wgsl/common.wgsl.js';

/** Emitter kinds supported by the lite tier via texture packing (B12). */
export const PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS: ReadonlySet<SceneEmitter['kind']> =
  new Set(['directional', 'point', 'spot', 'rect-area', 'disc-area']);
/** Emitter kinds in the full-tier set but NOT in the lite tier (no NEE path in lite kernel). */
export const PT_WEBGPU_LITE_UNSUPPORTED_EMITTER_KINDS: ReadonlySet<SceneEmitter['kind']> =
  new Set(
    [...PT_WEBGPU_SUPPORT.supportedEmitterKinds].filter(
      (k) => !PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS.has(k),
    ),
  );

/** Resolved engine state the capabilities object derives from. */
export interface PtWebgpuCapabilitiesFlags {
  readonly traceTier: PtWebgpuTraceTier;
  readonly maxSamplesLimit: number;
  readonly maxBouncesLimit: number;
  readonly bdpt: boolean;
  readonly restirPtReuse: boolean;
  readonly sampling: PtWebgpuSamplingMode;
  readonly bvhTraversal: PtWebgpuBvhTraversalMode;
  readonly causticStrategy: EngineCapabilities['causticStrategy'];
  /** Resolved `#postDenoiser instanceof OIDNFinalDispatcher`. */
  readonly isOidnFinal: boolean;
}

/**
 * Build the `EngineCapabilities` object for a pt-webgpu engine. Byte-identical to
 * the former inline literal in `PTEngineWebGPU.capabilities`.
 */
export function ptWebgpuCapabilities(flags: PtWebgpuCapabilitiesFlags): EngineCapabilities {
  return {
    // Material / transform / positions primitive patches upload in place.
    // `topology: true` — every COUNT-changing patch updatePrimitive can legally
    // receive is absorbed without a full setScene:
    //   • instanced-mesh instance-count change → TLAS-only rebuild, BLAS reused
    //     verbatim (slice-1);
    //   • mesh/skinned-mesh vertex/index-count change → rebuild ONLY the
    //     changed primitive's BLAS, splice into the concat buffers, rebase
    //     downstream offsets + TLAS roots, realloc the 10 geometry buffers
    //     (slice-2).
    // `id`/`kind` morphs throw (contract violation); whole-primitive add/remove
    // is `setScene`, not a patch — both correctly outside the `topology` flag.
    supportsIncrementalScene: true,
    incrementalPatchSupport: flags.traceTier === 'lite'
      ? {
          transform: false,
          positions: false,
          material: true,
          emitter: true,
          topology: false,
        }
      : {
          transform: true,
          positions: true,
          material: true,
          emitter: true,
          topology: true,
        },
    // Explicit whole-primitive add/remove API (addPrimitive / removePrimitive)
    // is implemented via a full buildPackedScene repack of the mutated scene.
    supportsAddRemovePrimitive: true,
    supportsAuxBuffers: flags.traceTier === 'full',
    accumulates: true,
    // Progressive walkaround→PT handoff (P8): this engine can seed its accum
    // buffers with an initial image as a decaying prior (seedAccumulator).
    supportsAccumulatorSeed: true,
    maxSamplesPerPixel: flags.maxSamplesLimit,
    maxBounces: flags.maxBouncesLimit,
    // H12 — lite-tier capabilities reflect what the lite kernel ACTUALLY binds:
    //   • No analytic shapes (group-1 is not bound on the lite layout; the
    //     analytic geometry/params/localToWorld/worldToLocal buffers are absent).
    //   • Mesh/skinned/instanced primitives are statically supported by baking
    //     all mesh-like primitives into one world-space BLAS at setScene time.
    //   • Emitters: directional + point + spot + rect-area + disc-area
    //     (B12 — texture-packed). Mesh-area remains unsupported.
    //   • Environments: none + procedural-sky + hdri (B12 — texture-packed).
    //   • No pt-webgpu-bdpt in experimentalFeatures even when bdpt:true was
    //     passed at construction (BDPT requires the full-tier group-2 layout).
    //
    // B12 (Wave B) — lite-tier fidelity cliff, SHIPPED.
    // The lite tier targets adapters reporting maxStorageBuffersPerShaderStage
    // as low as 8 (PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE). The lite
    // group-0 layout already consumes 7 storage buffers (bindings 2,3,4,5,6,7,8
    // = accum, positions, indices, triMaterialIds, materials, bvhNodes,
    // normals), leaving exactly ONE free storage-buffer slot under the 8 cap.
    //
    // B12 resolution: light data and HDRI env packed as sampled texture_2d<f32>
    // (bindings 12-14 in group-0, type = 'texture' not 'buffer' — counted from
    // maxSampledTexturesPerShaderStage ≥ 16, NOT the storage-buffer budget).
    //   • liteLightTex (binding 14): 1×N RGBA32F, directional (2 vec4/light)
    //     + point (3 vec4/light) + spot (4 vec4/light) + rect/disc-area
    //     (4 vec4/light) packed contiguously.
    //   • liteEnvTex (binding 12): W×H RGBA32F, .rgb = HDR radiance, .a = pdf/sr.
    //   • liteEnvCdfTex (binding 13): W×H RGBA32F, .r = marginal/conditional CDF
    //     value at pixel i+1 (2D layout to avoid 8192-width limit).
    // Budget arithmetic post-B12: 7 storage buffers (unchanged) + 3 sampled
    // textures (new, drawn from a separate ≥16 budget). The budget arithmetic is
    // PINNED by the liteTierBindingBudget test in webgpuLimits.test.ts.
    //
    // For the full tier the capability is derived from PT_WEBGPU_SUPPORT so
    // the declared set and the ingestion/packer behavior stay in sync.
    supportedAnalyticShapes: flags.traceTier === 'lite'
      ? new Set<AnalyticShape>()
      : new Set(PT_WEBGPU_SUPPORT.supportedAnalyticShapes),
    supportedEmitterKinds: flags.traceTier === 'lite'
      // B12 — point/spot/rect/disc-area now supported via lite texture packing (liteLightTex).
      ? new Set(PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS)
      : new Set(PT_WEBGPU_SUPPORT.supportedEmitterKinds),
    supportedPrimitiveKinds: flags.traceTier === 'lite'
      ? new Set<ScenePrimitive['kind']>(['mesh', 'skinned-mesh', 'instanced-mesh'])
      : new Set(PT_WEBGPU_SUPPORT.supportedPrimitiveKinds),
    supportedEnvironmentKinds: flags.traceTier === 'lite'
      // B12 — HDRI env now supported via lite texture packing (liteEnvTex + liteEnvCdfTex).
      ? new Set<Scene['environment']['kind']>(['none', 'procedural-sky', 'hdri'])
      : new Set(PT_WEBGPU_SUPPORT.supportedEnvironmentKinds),
    presentationMode: 'offscreen-texture',
    // H12 — lite-tier supportDetails must reflect what group-0 ACTUALLY binds,
    // not the full-tier ledger. Group-0 lite omits group-1 (analytic, env, lights)
    // and group-2 (TLAS, BDPT). Mesh-area emitters, analytic primitives, and
    // analytic shapes remain unsupported (no NEE path for those in the lite kernel).
    // Static instanced meshes are native because the lite packer bakes each
    // instance into its single root-0 BLAS. Geometry, transform, and topology
    // patches are accepted through a fallback merged-BLAS repack rather than
    // the full-tier BLAS/TLAS-native fast paths.
    // B12 — point/spot/rect/disc-area upgraded to 'native' (texture-packed NEE).
    // B12 — hdri upgraded to 'native' (liteEnvTex + liteEnvCdfTex importance sampling).
    supportDetails:
      flags.traceTier === 'lite'
        ? {
            primitives: {
              mesh: 'native',
              'skinned-mesh': 'native',
              'instanced-mesh': 'native',
              analytic: 'unsupported',
            },
            emitters: {
              directional: 'native',
              point: 'native',
              spot: 'native',
              'rect-area': 'native',
              'disc-area': 'native',
              'mesh-area': 'unsupported',
            },
            environments: {
              none: 'native',
              'procedural-sky': 'approximate',
              hdri: 'native',
            },
            analyticShapes: {
              sphere: 'unsupported',
              box: 'unsupported',
              capsule: 'unsupported',
              cylinder: 'unsupported',
              'h-channel-came': 'unsupported',
            },
            materials: PT_WEBGPU_LITE_MATERIALS,
            // SHADOW-01 — same rows as the full tier: primitive castShadow is
            // enforced in the SHARED traceMeshBvh any-hit path; the lite NEE
            // loops gate directional/point/spot/rect emitter flags through
            // the lite light texture records.
            shadows: BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.shadows,
            denoisers: BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.denoisers,
            mutations: {
              ...BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.mutations,
              transform: 'fallback-rebuild',
              positions: 'fallback-rebuild',
              material: 'native',
              topology: 'fallback-rebuild',
            },
          }
        : BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails,
    experimentalFeatures: new Set([
      ...(flags.traceTier === 'lite' ? (['pt-webgpu-lite-tier'] as const) : []),
      ...(flags.isOidnFinal
        ? (['pt-webgpu-oidn-final'] as const)
        : []),
      // BDPT requires the full-tier group-2 layout; suppress from lite even
      // when bdpt:true was passed at construction (the engine silently ignores
      // the flag for lite — the shader does not have the bdptEnabled UBO slot
      // and the BDPT sub-path pipeline is not created on the lite layout).
      ...(flags.bdpt && flags.traceTier !== 'lite' ? (['pt-webgpu-bdpt'] as const) : []),
      ...(flags.restirPtReuse ? (['pt-webgpu-restir-pt-reuse'] as const) : []),
      ...(flags.sampling === 'sobol' ? (['pt-webgpu-sobol-sampling'] as const) : []),
      ...(flags.bvhTraversal === 'cwbvh-closest-experimental'
        ? (['pt-webgpu-cwbvh-closest-traversal'] as const)
        : []),
      ...(flags.traceTier !== 'lite' && flags.causticStrategy === 'photon-map'
        ? (['pt-webgpu-photon-map-sppm'] as const)
        : []),
    ]),
    causticStrategy: flags.traceTier === 'lite' ? 'none' : flags.causticStrategy,
    // W3-D8 — this engine exposes `debug.estimatedGpuMemoryBytes()`.
    debugSurface: true,
  };
}
