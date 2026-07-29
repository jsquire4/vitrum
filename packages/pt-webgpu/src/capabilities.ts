// pt-webgpu capabilities builder (T3-B god-file split, 2026-07-20).
//
// Pure builder for the `EngineCapabilities` object the `PTEngineWebGPU.capabilities`
// getter previously constructed inline (~165 lines). Extracting it keeps the large
// documented literal out of the engine class body while preserving BYTE-IDENTICAL
// output: the getter now delegates to `ptWebgpuCapabilities(flags)`, passing the
// engine's resolved private-field values as a plain flags record. No logic change.

import type {
  EngineCapabilities,
  EngineFeatureId,
  MaterialSpec,
  SceneEmitter,
} from '@vitrum/core';
import {
  PT_WEBGPU_LITE_SUPPORT,
  ptWebgpuSupportManifest,
  ptWebgpuSupportSets,
} from './supportManifest.js';
import type { PtWebgpuTraceTier } from './traceTier.js';
import type { PtWebgpuBvhTraversalMode } from './gpuResources.js';
import type { PtWebgpuSamplingMode } from './wgsl/common.wgsl.js';

/** Build a runtime-immutable ReadonlySet view. TypeScript's `ReadonlySet` alone
 *  does not stop a host from casting back to `Set` and mutating a capability
 *  snapshot; this view exposes no mutator and freezes its public surface. */
function runtimeReadonlySet<T>(items: Iterable<T>): ReadonlySet<T> {
  const values = new Set(items);
  const view: ReadonlySet<T> = {
    get size() { return values.size; },
    has: (value) => values.has(value),
    entries: () => values.entries(),
    keys: () => values.keys(),
    values: () => values.values(),
    forEach: (callback, thisArg) => {
      for (const value of values) callback.call(thisArg, value, value, view);
    },
    [Symbol.iterator]: () => values[Symbol.iterator](),
  };
  return Object.freeze(view);
}

/** Emitter kinds supported by the lite tier via texture packing (B12). */
export const PT_WEBGPU_LITE_SUPPORTED_EMITTER_KINDS:
ReadonlySet<SceneEmitter['kind']> = PT_WEBGPU_LITE_SUPPORT.supportedEmitterKinds;

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
  readonly spectral: boolean;
  readonly denoiser: 'none' | 'oidn-final';
}

const PT_WEBGPU_FULL_INVERSE_RENDERING: NonNullable<
  EngineCapabilities['inverseRendering']
> = Object.freeze({
  methods: Object.freeze({
    'finite-difference': 'native',
    'path-replay': 'native',
  }),
  pathReplay: Object.freeze({
    failurePolicy: 'error',
    materialFields: runtimeReadonlySet<keyof MaterialSpec>([
      'emissive',
    ]),
    emitterFields: runtimeReadonlySet<'color' | 'intensity'>([]),
    maxBounces: 1,
    supportsSpectral: false,
    supportsBdpt: false,
    supportsRestirPtReuse: false,
    supportsCausticStrategies: false,
  }),
});

const PT_WEBGPU_LITE_INVERSE_RENDERING: NonNullable<
  EngineCapabilities['inverseRendering']
> = Object.freeze({
  methods: Object.freeze({
    'finite-difference': 'native',
    'path-replay': 'unsupported',
  }),
});

/**
 * Build the `EngineCapabilities` object for a pt-webgpu engine. Byte-identical to
 * the former inline literal in `PTEngineWebGPU.capabilities`.
 */
export function ptWebgpuCapabilities(flags: PtWebgpuCapabilitiesFlags): EngineCapabilities {
  const supportManifest = ptWebgpuSupportManifest(flags.traceTier);
  const supportSets = ptWebgpuSupportSets(flags.traceTier);
  const mutationDetails = supportManifest.mutations;
  const mutationAccepted = (
    mode: (typeof mutationDetails)[keyof typeof mutationDetails],
  ): boolean => mode !== 'unsupported';
  const activeFeatures = new Set<EngineFeatureId>();
  if (flags.bdpt && supportManifest.bidirectionalPathTracing != null) {
    activeFeatures.add('pt-webgpu-bdpt');
  }
  if (flags.restirPtReuse) {
    activeFeatures.add('pt-webgpu-one-edge-gris-reconnection');
  }
  if (flags.sampling === 'sobol') activeFeatures.add('pt-webgpu-sobol-sampling');
  if (flags.bvhTraversal === 'cwbvh-closest') {
    activeFeatures.add('pt-webgpu-cwbvh-closest-traversal');
  }
  if (flags.traceTier !== 'lite' && flags.causticStrategy === 'photon-map') {
    activeFeatures.add('pt-webgpu-photon-map-sppm');
  }
  if (flags.spectral) activeFeatures.add('pt-webgpu-spectral');
  if (flags.denoiser === 'oidn-final') activeFeatures.add('pt-webgpu-oidn-final');

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
    supportsIncrementalScene:
      mutationAccepted(mutationDetails.transform) ||
      mutationAccepted(mutationDetails.positions) ||
      mutationAccepted(mutationDetails.material) ||
      mutationAccepted(mutationDetails.emitter) ||
      mutationAccepted(mutationDetails.topology),
    incrementalPatchSupport: {
      // These booleans promise API acceptance. The manifest is the exact
      // native-vs-fallback grade, so lite-tier rebuild routes remain true here.
      transform: mutationAccepted(mutationDetails.transform),
      positions: mutationAccepted(mutationDetails.positions),
      material: mutationAccepted(mutationDetails.material),
      emitter: mutationAccepted(mutationDetails.emitter),
      topology: mutationAccepted(mutationDetails.topology),
    },
    // Explicit whole-primitive add/remove API (addPrimitive / removePrimitive)
    // is implemented via a full buildPackedScene repack of the mutated scene.
    supportsAddRemovePrimitive:
      mutationAccepted(mutationDetails.addPrimitive) &&
      mutationAccepted(mutationDetails.removePrimitive),
    supportsAuxBuffers: supportManifest.motionVectors != null,
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
    //   • BDPT is absent from activeFeatures when the lite tier is selected
    //     because BDPT requires the full-tier group-2 layout.
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
    // Both tiers derive these sets from their executable support manifest so
    // capability reporting and ingestion/packing behavior stay in sync.
    supportedAnalyticShapes: new Set(supportSets.supportedAnalyticShapes),
    supportedEmitterKinds: new Set(supportSets.supportedEmitterKinds),
    supportedPrimitiveKinds: new Set(supportSets.supportedPrimitiveKinds),
    supportedEnvironmentKinds: new Set(supportSets.supportedEnvironmentKinds),
    presentationMode: 'offscreen-texture',
    // The selected backend-local manifest is the single runtime source for
    // host reporting, validation, and coarse supported-kind sets.
    supportDetails: supportManifest,
    // Inverse rendering is a stable finite-difference surface on both tiers.
    // Path replay is a separate, exact claim: only the full-tier material
    // `emissive` gradient has passed the end-to-end GPU fit gate. The public
    // engine rejects every out-of-domain path-replay request instead of
    // silently changing the selected method.
    inverseRendering:
      flags.traceTier === 'full'
        ? PT_WEBGPU_FULL_INVERSE_RENDERING
        : PT_WEBGPU_LITE_INVERSE_RENDERING,
    activeFeatures,
    causticStrategy: flags.traceTier === 'lite' ? 'none' : flags.causticStrategy,
    // W3-D8 — this engine exposes `debug.estimatedGpuMemoryBytes()`.
    debugSurface: true,
  };
}
