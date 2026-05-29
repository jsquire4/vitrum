import { PT_WEBGPU_COMMON_WGSL } from './common.wgsl.js';
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
 *   7. `caustic`      — MNEE chain walker + manifold-NEE + photon-map
 *                       gather strategies (causticStrategy modes 1 / 2).
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
export function composePtWebgpuTraceWgsl(bdptEnabled: boolean): string {
  const kernel = composePathTraceKernelWgsl({ volumetricSss: !bdptEnabled });
  return /* wgsl */ `
${PT_WEBGPU_COMMON_WGSL}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL}
${PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL}
${PT_WEBGPU_PATH_TRACE_BSDF_WGSL}
${PT_WEBGPU_PATH_TRACE_CONNECT_WGSL}
${PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL}
${PT_WEBGPU_BDPT_CONNECTION_WGSL}
${PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL}
${kernel}
`;
}

/**
 * Default full-tier composition — BDPT off ⇒ volumetric SSS walk present.
 * Preserved as a const for the many WGSL-contract tests + the non-BDPT
 * pipeline path. \`composePtWebgpuTraceWgsl(true)\` yields the BDPT-on variant.
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
${PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL}
${PT_WEBGPU_BDPT_CONNECTION_WGSL}
${PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL}
${PT_WEBGPU_PATH_TRACE_KERNEL_WGSL}
`;
