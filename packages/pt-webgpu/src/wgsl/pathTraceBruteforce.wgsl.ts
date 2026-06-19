import {
  PT_WEBGPU_COMMON_WGSL,
  composePtWebgpuCommonWgsl,
  type PtWebgpuSamplingMode,
} from './common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  HERO_WAVELENGTH_WGSL,
  LUMINANCE_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from './pathTrace/material.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL } from './pathTrace/intersection.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from './pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from './pathTrace/connect.wgsl.js';
import {
  MNEE_NEWTON_WGSL,
  MNEE_CHAIN_WGSL,
  MNEE_CONNECTION_WGSL,
} from './pathTrace/mneeNewton.wgsl.js';
import {
  SPPM_GROUP3_BINDINGS_WGSL,
  SPPM_PHOTON_PASS_WGSL,
} from './pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from './pathTrace/caustic.wgsl.js';
import {
  PT_WEBGPU_PATH_TRACE_KERNEL_WGSL,
  composePathTraceKernelWgsl,
} from './pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from './bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from './bdpt/bdptLightSubpath.wgsl.js';

/**
 * Brute-force path tracing kernel — orchestrator that concatenates the six
 * per-concern WGSL modules under `./pathTrace/` into a single shader source
 * for `device.createShaderModule`.
 *
 * The concatenation order is fixed by WGSL's struct-before-use requirement
 * (functions are order-independent at the spec level, but several modules
 * reference structs defined in earlier modules):
 *
 *   1. `PT_WEBGPU_COMMON_WGSL`  — PCG RNG + BVHNode/Ray + intersectTriangle.
 *   2. `HAMMERSLEY_WGSL` / `OCTAHEDRAL_CORE_WGSL` — shared-samplers helpers
 *      (uniformSphere, hammersley, octahedral encoders).
 *   3. `material`     — FrameParams + 24 group(0) bindings + material
 *                       constants + BsdfSample + Fresnel/microfacet/MIS
 *                       primitives + decodeMaterial.
 *   4. `intersection` — SceneHit + analytic-shape intersectors +
 *                       traceMeshBvh/traceAnalyticShapes/traceClosest/
 *                       traceAny + hitMaterialId.
 *   5. `bsdf`         — evaluateBrdf + brdfDirectionalPdf + cosine /
 *                       glossy VNDF samplers + BounceSample +
 *                       sampleNextBounceDirection.
 *   6. `connect`      — environment-map helpers (sky / equirect / importance)
 *                       + area-light directional intersectors +
 *                       BSDF→light/env MIS connection contributions.
 *   6b. `mneeNewton`  — the real MNEE half-vector Newton solve +
 *                       `mneeReflectionIrradiance` connection core. Placed BEFORE
 *                       `caustic` so the reflection-caustic strategy can call it.
 *   6c. `mneeChain`   — the 2-vertex specular chain (glass-slab enter+exit)
 *                       block-tridiagonal Newton + chain connection-PDF. Placed
 *                       AFTER `mneeNewton` (it reuses `mnee_safe_normalize`) and
 *                       BEFORE `caustic` so the glass-slab caustic can call it.
 *   6d. `sppm`        — A4 SPPM group-3 hash-grid bindings (SppmStats UBO +
 *                       sppmPhotonCells + sppmCellCounters) + sppmInsertPhoton +
 *                       sppmGatherProgressive.  Composed BEFORE `caustic` because
 *                       `photonMapContribution` calls `sppmGatherProgressive`
 *                       (WGSL requires callees to precede callers in source order).
 *   7. `caustic`      — REAL MNEE reflection caustic + transmissive cone-search
 *                       MNEE + SPPM gather shim (causticStrategy modes 1 / 2).
 *   8. `kernel`       — primary-ray generation, projectToNdc, causticMode,
 *                       RR helpers, accumulateFrame, and the @compute @main
 *                       entry point that walks each path.
 *
 * The compiled WGSL string is byte-equivalent (modulo whitespace) to the
 * pre-split monolith; behaviour and the public `PT_WEBGPU_TRACE_WGSL`
 * export name are preserved so the consumer in `src/index.ts` and the
 * existing `wgslContract.test.ts` / `wgslSmoke.gpu.test.ts` tests keep
 * working unchanged.
 */
/**
 * Compose the full-tier brute-force path-trace WGSL for a given integrator
 * configuration.
 *
 * WS4 — the volumetric subsurface-scattering random walk is compiled in ONLY
 * when BDPT is disabled. The BDPT light subpath has no participating-media
 * logic, so a medium that attenuates / scatters only the eye path would break
 * energy conservation; the gate is therefore structural (the SSS WGSL symbols
 * are simply absent from the BDPT-on shader) rather than a runtime UBO branch.
 * When `bdptEnabled` is true the kernel emits the legacy per-channel
 * Beer-Lambert absorption fallback instead.
 */
export interface PtWebgpuTraceComposeOptions {
  readonly sampling?: PtWebgpuSamplingMode;
}

export function composePtWebgpuTraceWgsl(
  bdptEnabled: boolean,
  opts: PtWebgpuTraceComposeOptions = {},
): string {
  const kernel = composePathTraceKernelWgsl({ volumetricSss: !bdptEnabled });
  const common = composePtWebgpuCommonWgsl(opts.sampling ?? 'pcg');
  return /* wgsl */ `
${common}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL}
${PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL}
${PT_WEBGPU_PATH_TRACE_BSDF_WGSL}
${PT_WEBGPU_PATH_TRACE_CONNECT_WGSL}
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}
${MNEE_CONNECTION_WGSL}
${SPPM_GROUP3_BINDINGS_WGSL}
${PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL}
${PT_WEBGPU_BDPT_CONNECTION_WGSL}
${PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL}
${kernel}
`;
}

/**
 * A1 — the ReSTIR-PT COMPOSITE full-tier megakernel. The kernel is composed in
 * composite mode (E0-direct-only + add the resolve indirect; see
 * composePathTraceKernelWgsl restirPtComposite). The kernel declares the resolve
 * output at `@group(4) @binding(3)`; this wrapper RELOCATES that decl onto
 * `@group(0) @binding(23)` — the SAME relocated slot the reuse passes' rpt_result
 * occupies (RPT_GROUP0_BINDING_BASE + 3 = 23) — so the megakernel reads exactly the
 * buffer the resolve pass wrote, via the reuse-extended group-0 bind group + layout.
 * Composed ONLY when restirPtReuse is active; the default `PT_WEBGPU_TRACE_WGSL`
 * and `composePtWebgpuTraceWgsl` are untouched (OFF-path byte-identical).
 *
 * The relocation is a string rewrite of the module-scope binding DECL only
 * (`@group(4) @binding(N)` → `@group(0) @binding(20+N)`). Only binding 3 is
 * present in the kernel (rpt_result_in at group(0) binding 23). SPPM bindings
 * are in group(3) (bindings 6/7/8) and are not relocated by this transform.
 */
export function composePtWebgpuCompositeTraceWgsl(
  bdptEnabled: boolean,
  opts: PtWebgpuTraceComposeOptions = {},
): string {
  const kernel = composePathTraceKernelWgsl({
    volumetricSss: !bdptEnabled,
    restirPtComposite: true,
  });
  const common = composePtWebgpuCommonWgsl(opts.sampling ?? 'pcg');
  const body = /* wgsl */ `
${common}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL}
${PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL}
${PT_WEBGPU_PATH_TRACE_BSDF_WGSL}
${PT_WEBGPU_PATH_TRACE_CONNECT_WGSL}
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}
${MNEE_CONNECTION_WGSL}
${SPPM_GROUP3_BINDINGS_WGSL}
${PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL}
${PT_WEBGPU_BDPT_CONNECTION_WGSL}
${PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL}
${kernel}
`;
  // Relocate @group(4)@binding(N) → @group(0)@binding(20+N).
  // The ONLY group(4) binding in the composite megakernel is the RPT result
  // at binding(3) → @group(0)@binding(23). SPPM bindings are in group(3)
  // (A4 decision: bindings 6/7/8) and do not need relocation here.
  return body.replace(
    /@group\(4\)\s+@binding\((\d+)\)/g,
    (_m, b: string) => `@group(0) @binding(${20 + Number(b)})`,
  );
}

/**
 * A4 — compose the SPPM photon-emission pass WGSL.  This is a SEPARATE
 * compute pipeline (entry point `sppmEmitPhotons`, workgroup_size(64,1,1)) that
 * runs BEFORE the megakernel each frame when `causticStrategy == 'photon-map'`.
 *
 * The pass needs the full module stack (PCG RNG, scene bindings, BVH
 * traceClosest, material decodeMaterial, etc.) plus the SPPM group-3 bindings
 * (bindings 6–9; read_write — it WRITES photons into the hash grid).  The
 * megakernel separately reads from the same group-3 bind group.
 *
 * The `sppmEmitPhotons` entry point + its helpers live in
 * `SPPM_GROUP3_BINDINGS_WGSL` + `SPPM_PHOTON_PASS_WGSL`.  BSDF helpers
 * (uniformSphere, buildOnb) and environment helpers come from the standard
 * module stack.
 *
 * Full-tier only; never composed on lite.
 */
export function composeSppmPhotonPassWgsl(
  opts: PtWebgpuTraceComposeOptions = {},
): string {
  const common = composePtWebgpuCommonWgsl(opts.sampling ?? 'pcg');
  return /* wgsl */ `
${common}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL}
${PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL}
${PT_WEBGPU_PATH_TRACE_BSDF_WGSL}
${PT_WEBGPU_PATH_TRACE_CONNECT_WGSL}
${SPPM_GROUP3_BINDINGS_WGSL}
${SPPM_PHOTON_PASS_WGSL}
`;
}

/**
 * Default full-tier composition — BDPT off ⇒ volumetric SSS walk present.
 * Preserved as a const for the many WGSL-contract tests + the non-BDPT
 * pipeline path. `composePtWebgpuTraceWgsl(true)` yields the BDPT-on variant.
 *
 * Re-pinned 2026-06-10: A4 real SPPM progressive photon map replaces the
 * per-pixel 32-photon approximation (removed: gatherRadius=0.35, ×1.25 fudge).
 * SPPM_GROUP3_BINDINGS_WGSL added to the composition (RENDER-CHANGING for
 * causticStrategy:'photon-map'; off-path byte-identical for other strategies).
 * The WGSL string changes (hence this re-pin); the OFF runtime path
 * (causticStrategy:'none'/'manifold-nee') does not change radiometrically.
 * A/B pending V28-B.
 */
export const PT_WEBGPU_TRACE_WGSL = /* wgsl */ `
${PT_WEBGPU_COMMON_WGSL}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL}
${PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL}
${PT_WEBGPU_PATH_TRACE_BSDF_WGSL}
${PT_WEBGPU_PATH_TRACE_CONNECT_WGSL}
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}
${MNEE_CONNECTION_WGSL}
${SPPM_GROUP3_BINDINGS_WGSL}
${PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL}
${PT_WEBGPU_BDPT_CONNECTION_WGSL}
${PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL}
${PT_WEBGPU_PATH_TRACE_KERNEL_WGSL}
`;
