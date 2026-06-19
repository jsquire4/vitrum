#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env
// @ts-nocheck — this is a .mjs file; deno runs it as JS with --sloppy-imports.
/**
 * tools/shader-gate/gate.mjs
 *
 * In-repo WGSL compile gate.  Discovers every composed WGSL string from the
 * three target packages (pt-webgpu, walkaround-hybrid, shared-denoisers),
 * submits each to the WebGPU driver via createShaderModule + getCompilationInfo,
 * and exits non-zero on the first error so CI hard-fails with a named shader +
 * the first error lines.
 *
 * Usage (from repo root):
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
 *   deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env \
 *     tools/shader-gate/gate.mjs
 *
 * Self-test mode (proves detection works):
 *   ... gate.mjs --self-test
 *
 * The deno.json in this directory provides the @vitrum/* import-map so all
 * transitive imports from the production TS files resolve correctly.
 *
 * NOTE: This file uses relative imports rooted at the tools/shader-gate/
 * directory.  Deno resolves them relative to import.meta.url.  Do NOT use
 * hard-coded absolute paths — the wsl-gpu oracles already demonstrated how
 * those break when the repo is cloned to a different location.
 */

import { applyNagaFix } from "./nagaFix.mjs";

// ── Parse flags ──────────────────────────────────────────────────────────────
const selfTest = Deno.args.includes("--self-test");
const noPipelineGate = Deno.args.includes("--no-pipeline-gate");

// ── Acquire the WebGPU device ─────────────────────────────────────────────────
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  console.error("[shader-gate] ERROR: No WebGPU adapter available.");
  console.error("  Set VK_ICD_FILENAMES to point at the lavapipe or dzn ICD.");
  Deno.exit(1);
}

// Request the device with maxBindGroups elevated to what the adapter supports.
// By default requestDevice() gives maxBindGroups=4, but the NRC shader uses @group(4)
// (index 4 = 5th group).  If the adapter supports more than 4 groups we request them
// so the NRC shader compiles; otherwise it is skipped with a warning.
const adapterMaxBG = (adapter.limits && adapter.limits.maxBindGroups != null)
  ? adapter.limits.maxBindGroups
  : 4;
const device = await adapter.requestDevice(
  adapterMaxBG > 4
    ? { requiredLimits: { maxBindGroups: adapterMaxBG } }
    : {},
);

// ── Shader inventory ──────────────────────────────────────────────────────────
// Each entry: { name, wgsl, entryPoint?, entryPoints? }
// entryPoint(s) are optional for shader-module compilation, but when present
// the gate also asks the adapter to create a compute pipeline for that entry.
const shaders = [];

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1: pt-webgpu
// Main composers: composePtWebgpuTraceWgsl (full + bdpt variant),
// composePtWebgpuCompositeTraceWgsl, PT_WEBGPU_TRACE_WGSL (const, SSS default),
// PT_WEBGPU_TRACE_LITE_WGSL, PT_WEBGPU_SEED_BLIT_WGSL.
// ReSTIR-PT per-pass composers: composeRestirPt{Producer,Temporal,Spatial,Resolve}Wgsl.
// ─────────────────────────────────────────────────────────────────────────────
{
  const {
    composePtWebgpuTraceWgsl,
    composePtWebgpuCompositeTraceWgsl,
    composeSppmPhotonPassWgsl,
    PT_WEBGPU_TRACE_WGSL,
  } = await import("../../packages/pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts");

  const { PT_WEBGPU_TRACE_LITE_WGSL, composePtWebgpuTraceLiteWgsl } = await import(
    "../../packages/pt-webgpu/src/wgsl/pathTraceBruteforceLite.wgsl.ts"
  );

  const { PT_WEBGPU_SEED_BLIT_WGSL } = await import(
    "../../packages/pt-webgpu/src/wgsl/seedBlit.wgsl.ts"
  );

  const { PT_WEBGPU_ADJOINT_PASS_WGSL } = await import(
    "../../packages/pt-webgpu/src/wgsl/pathTrace/adjointPass.wgsl.ts"
  );

  const {
    composeRestirPtProducerWgsl,
    composeRestirPtTemporalWgsl,
    composeRestirPtSpatialWgsl,
    composeRestirPtResolveWgsl,
  } = await import(
    "../../packages/pt-webgpu/src/wgsl/pathTrace/restirPtCompose.wgsl.ts"
  );

  // Full-tier path-trace kernel — SSS walk (default, bdpt=false)
  shaders.push({
    name: "pt-webgpu/trace-full-sss",
    wgsl: PT_WEBGPU_TRACE_WGSL,
    entryPoint: "main",
  });

  // Full-tier path-trace kernel — BDPT on (SSS walk off)
  shaders.push({
    name: "pt-webgpu/trace-full-bdpt",
    wgsl: composePtWebgpuTraceWgsl(true),
    entryPoint: "main",
  });

  // Opt-in low-discrepancy sampling variants. The default PCG variants above
  // stay byte-pinned; these compile the selected binding-free Sobol RNG module.
  shaders.push({
    name: "pt-webgpu/trace-full-sss-sobol",
    wgsl: composePtWebgpuTraceWgsl(false, { sampling: "sobol" }),
    entryPoint: "main",
  });
  shaders.push({
    name: "pt-webgpu/trace-full-bdpt-sobol",
    wgsl: composePtWebgpuTraceWgsl(true, { sampling: "sobol" }),
    entryPoint: "main",
  });

  // Opt-in CWBVH closest/any-hit traversal variant. The any-hit wrapper keeps
  // castShadow predicate parity with the binary BVH shadow path.
  shaders.push({
    name: "pt-webgpu/trace-full-cwbvh-closest",
    wgsl: composePtWebgpuTraceWgsl(false, { cwbvhClosest: true }),
    entryPoint: "main",
  });

  // Lite-tier path-trace kernel
  shaders.push({
    name: "pt-webgpu/trace-lite",
    wgsl: PT_WEBGPU_TRACE_LITE_WGSL,
    entryPoint: "main",
  });
  shaders.push({
    name: "pt-webgpu/trace-lite-sobol",
    wgsl: composePtWebgpuTraceLiteWgsl({ sampling: "sobol" }),
    entryPoint: "main",
  });

  // Composite megakernel (ReSTIR-PT A1, SSS variant)
  shaders.push({
    name: "pt-webgpu/composite-trace-sss",
    wgsl: composePtWebgpuCompositeTraceWgsl(false),
    entryPoint: "main",
  });

  // Composite megakernel — BDPT on
  shaders.push({
    name: "pt-webgpu/composite-trace-bdpt",
    wgsl: composePtWebgpuCompositeTraceWgsl(true),
    entryPoint: "main",
  });
  shaders.push({
    name: "pt-webgpu/composite-trace-sss-sobol",
    wgsl: composePtWebgpuCompositeTraceWgsl(false, { sampling: "sobol" }),
    entryPoint: "main",
  });

  // Seed-blit kernel (progressive walkaround→PT handoff)
  shaders.push({
    name: "pt-webgpu/seed-blit",
    wgsl: PT_WEBGPU_SEED_BLIT_WGSL,
    entryPoint: "main",
  });

  // Standalone inverse-rendering adjoint pass. Compile-only gate: the pass binds
  // a focused one-group read subset with many storage buffers, so pipeline
  // creation is adapter-limit-sensitive; module compilation still catches WGSL
  // syntax/type drift.
  shaders.push({
    name: "pt-webgpu/adjoint-pass",
    wgsl: PT_WEBGPU_ADJOINT_PASS_WGSL,
  });

  // A4 — SPPM photon-emission pass (full-tier only; @group(3) bindings 6/7/8).
  // The SPPM bindings live in group 3 (same group as the light-tree / material
  // textures) so maxBindGroups=4 is sufficient — works on lavapipe.
  shaders.push({
    name: "pt-webgpu/sppm-photon-pass",
    wgsl: composeSppmPhotonPassWgsl(),
    entryPoint: "sppmEmitPhotons",
  });
  shaders.push({
    name: "pt-webgpu/sppm-photon-pass-sobol",
    wgsl: composeSppmPhotonPassWgsl({ sampling: "sobol" }),
    entryPoint: "sppmEmitPhotons",
  });

  // ReSTIR-PT per-pass composers (per-pass because the combined unit has
  // conflicting binding decls that naga rejects — see restirPtCompose.wgsl.ts)
  shaders.push({
    name: "pt-webgpu/restirpt-producer",
    wgsl: composeRestirPtProducerWgsl(),
    entryPoint: "restirPtProduce",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-producer-sobol",
    wgsl: composeRestirPtProducerWgsl({ sampling: "sobol" }),
    entryPoint: "restirPtProduce",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-temporal",
    wgsl: composeRestirPtTemporalWgsl(),
    entryPoint: "restirPtTemporal",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-temporal-sobol",
    wgsl: composeRestirPtTemporalWgsl({ sampling: "sobol" }),
    entryPoint: "restirPtTemporal",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-spatial",
    wgsl: composeRestirPtSpatialWgsl(),
    entryPoint: "restirPtSpatial",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-spatial-sobol",
    wgsl: composeRestirPtSpatialWgsl({ sampling: "sobol" }),
    entryPoint: "restirPtSpatial",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-resolve",
    wgsl: composeRestirPtResolveWgsl(),
    entryPoint: "restirPtResolve",
  });
  shaders.push({
    name: "pt-webgpu/restirpt-resolve-sobol",
    wgsl: composeRestirPtResolveWgsl({ sampling: "sobol" }),
    entryPoint: "restirPtResolve",
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2: walkaround-hybrid
//
// walkaround-hybrid uses a declarative include-graph: composeWgsl(module, WGSL_MODULES)
// produces a standalone compiled WGSL string from any root module.  We enumerate
// every module that pipelineCompiler.ts actually compiles as a pipeline entry point.
//
// Two special cases require parametric builders:
//   - buildPpgUpdateWgsl(maxDTreeNodesPerCell)  — PPG update kernel
//   - buildRisGiNrcModule(cfg)                  — NRC variant of GI-RIS
//     (this is a WgslModule; compose with composeWgsl to get the string)
// ─────────────────────────────────────────────────────────────────────────────
{
  const { composeWgsl } = await import(
    "../../packages/walkaround-hybrid/src/pipeline/wgslComposer.ts"
  );

  const {
    WGSL_MODULES,
    RIS_MODULE,
    REGIR_BUILD_MODULE,
    TEMPORAL_MODULE,
    SPATIAL_MODULE,
    SHADE_MODULE,
    MOTION_VECTORS_MODULE,
    RIS_GI_MODULE,
    TEMPORAL_GI_MODULE,
    TEMPORAL_GI_GRIS_MODULE,
    SPATIAL_GI_MODULE,
    SPATIAL_GI_GRIS_MODULE,
    WELFORD_TEMPORAL_MODULE,
    SAMPLE_BUDGET_MODULE,
    RESOLVE_MODULE,
    GTAO_MODULE,
    GTAO_UPSAMPLE_MODULE,
    INDIRECT_COMBINE_MODULE,
    INDIRECT_TEMPORAL_ACCUM_MODULE,
    TRANSPARENT_OIT_MODULE,
    COMPOSITE_VERT_MODULE,
    COMPOSITE_FRAG_MODULE,
    PPG_UPDATE_MODULE: _PPG_UPDATE_MODULE,
    ATROUS_MODULE,
    TEMPORAL_ACCUM_MODULE,
    ATROUS_VARIANCE_MODULE,
    SVGF_REPROJECTION_MODULE,
    SVGF_VARIANCE_FROM_MOMENTS_MODULE,
    SVGF_7X7_SPATIAL_FALLBACK_MODULE,
    BMFR_MODULE,
    CB_PREFILL_MODULE,
  } = await import(
    "../../packages/walkaround-hybrid/src/pipeline/wgslModules.ts"
  );

  const { buildPpgUpdateWgsl } = await import(
    "../../packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts"
  );

  const { PPG_TREE_LAYOUT_WGSL } = await import(
    "../../packages/walkaround-hybrid/src/ppg/ppgTreeLayout.wgsl.ts"
  );

  const { LUMINANCE_WGSL: WH_LUMINANCE_WGSL } = await import(
    "../../packages/shared-samplers/src/wgsl/luminance.wgsl.ts"
  );

  const { buildRisGiNrcModule } = await import(
    "../../packages/walkaround-hybrid/src/shaders/risGiNrc.wgsl.ts"
  );

  // Helper: add a composed walkaround-hybrid shader.
  // The naga fix is applied to every composed walkaround-hybrid WGSL because
  // shared-bvh traversal helpers use ptr<storage> function parameters which
  // naga/lavapipe rejects.  The fix is purely mechanical (zero semantic change);
  // see nagaFix.mjs for the full rationale.
  const addWh = (name, rootModule) => {
    const raw = composeWgsl(rootModule, WGSL_MODULES);
    shaders.push({
      name: `walkaround-hybrid/${name}`,
      wgsl: applyNagaFix(raw),
    });
  };

  // Core ReSTIR-DI + walkaround passes
  addWh("ris", RIS_MODULE);
  addWh("temporal", TEMPORAL_MODULE);
  addWh("spatial", SPATIAL_MODULE);
  addWh("shade", SHADE_MODULE);
  addWh("motionVectors", MOTION_VECTORS_MODULE);
  addWh("sampleBudget", SAMPLE_BUDGET_MODULE);
  addWh("resolve", RESOLVE_MODULE);
  addWh("cbPrefill", CB_PREFILL_MODULE);
  addWh("regirBuild", REGIR_BUILD_MODULE);

  // GTAO passes
  addWh("gtao", GTAO_MODULE);
  addWh("gtaoUpsample", GTAO_UPSAMPLE_MODULE);

  // ReSTIR-GI passes (default + GRIS variants)
  addWh("risGi", RIS_GI_MODULE);
  addWh("temporalGi", TEMPORAL_GI_MODULE);
  addWh("temporalGiGris", TEMPORAL_GI_GRIS_MODULE);
  addWh("spatialGi", SPATIAL_GI_MODULE);
  addWh("spatialGiGris", SPATIAL_GI_GRIS_MODULE);

  // Indirect channel
  addWh("indirectCombine", INDIRECT_COMBINE_MODULE);
  addWh("indirectTemporalAccum", INDIRECT_TEMPORAL_ACCUM_MODULE);
  addWh("transparentOit", TRANSPARENT_OIT_MODULE);

  // Adaptive-sampling Welford temporal
  addWh("welfordTemporal", WELFORD_TEMPORAL_MODULE);

  // Composite render pipeline (vert + frag shaders — these are vertex/fragment
  // entry points, not compute; getCompilationInfo works on them identically)
  addWh("compositeVert", COMPOSITE_VERT_MODULE);
  addWh("compositeFrag", COMPOSITE_FRAG_MODULE);

  // Shared-denoiser modules (consumed by walkaround-hybrid's denoiser passes)
  addWh("atrous", ATROUS_MODULE);
  // temporalAccum has requires:[] so composeWgsl emits the raw source
  addWh("temporalAccum", TEMPORAL_ACCUM_MODULE);
  addWh("atrousVariance", ATROUS_VARIANCE_MODULE);
  addWh("svgfReprojection", SVGF_REPROJECTION_MODULE);
  addWh("svgfVarianceFromMoments", SVGF_VARIANCE_FROM_MOMENTS_MODULE);
  addWh("svgf7x7SpatialFallback", SVGF_7X7_SPATIAL_FALLBACK_MODULE);
  addWh("bmfr", BMFR_MODULE);

  // PPG update kernel — parametric builder (default maxDTreeNodesPerCell = 341).
  // buildPpgUpdateWgsl returns a raw WGSL string that references:
  //   - STREE_HEADER_F32 / STREE_NODE_STRIDE / DTREE_* (from ppgTreeLayout)
  //   - luminance() (from shared-samplers/luminance)
  // Prepend both — exactly what pipelineCompiler does via the composeWgsl require-graph
  // (PPG_UPDATE_MODULE.requires includes 'ppgTreeLayout', which transitively pulls
  // luminance via the common module chain, but the gate uses a direct prepend for clarity).
  shaders.push({
    name: "walkaround-hybrid/ppgUpdate",
    wgsl: applyNagaFix(`${WH_LUMINANCE_WGSL}\n${PPG_TREE_LAYOUT_WGSL}\n${buildPpgUpdateWgsl(341)}`),
    entryPoint: "ppgUpdateMain",
  });

  // NRC GI-RIS variant — compiled only when nrcEnabled.  We use a representative
  // config matching the defaults documented in risGiNrc.wgsl.ts / nrcQuery.wgsl.ts.
  // NrcQueryWgslOptions extends NrcEncodeWgslOptions:
  //   levels (hash-grid levels L), featuresPerEntry (F), oneBlobBins (k),
  //   width (hidden width W), outWidth (output neurons), hidden (hidden layers).
  const nrcCfg = {
    levels: 16,
    featuresPerEntry: 2,
    oneBlobBins: 4,
    width: 64,
    outWidth: 3,
    hidden: 5,
  };
  // NRC GI-RIS module uses @group(4) for its NRC bindings.  WebGPU guarantees only
  // maxBindGroups=4 (groups 0..3).  lavapipe also reports maxBindGroups=4, so this
  // shader fails with "group index 4 exceeds limit" on lavapipe.  Adapters with
  // maxBindGroups>=5 (e.g. dzn on D3D12, some Vulkan ICDs) can compile it.
  // We compile it only when the adapter reports sufficient bind groups.
  // adapterMaxBG is set near the top of the file from the adapter limit.
  if (adapterMaxBG >= 5) {
    const nrcModule = buildRisGiNrcModule(nrcCfg);
    shaders.push({
      name: "walkaround-hybrid/risGiNrc",
      wgsl: applyNagaFix(composeWgsl(nrcModule, WGSL_MODULES)),
      entryPoint: "risGiMain",
    });
  } else {
    console.log(`[shader-gate] SKIP  walkaround-hybrid/risGiNrc  (requires maxBindGroups>=5; adapter has ${adapterMaxBG})`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3: shared-denoisers standalone WGSL strings
//
// These are the raw WGSL string exports from @vitrum/shared-denoisers.
// We compile each one in isolation (no deps prepended) matching the standalone
// usage documented in their module headers.  The walkaround-hybrid versions
// (above) are compiled with the composeWgsl dep-graph; these cover the standalone
// path (e.g. runSVGFRealWebGPU, runBmfrWebGPU, runHdrLuminanceBilateralWebGPU).
//
// Note: WELFORD_VARIANCE_WGSL is a fragment, not a standalone shader — it has no
// entry point.  We compile it here anyway so any syntax error is caught, even
// though it would only surface as an "no entry point" validation message, not a
// compile error.
// ─────────────────────────────────────────────────────────────────────────────
{
  const {
    ATROUS_WGSL: _ATROUS_WGSL,
    TEMPORAL_ACCUM_WGSL,
    ATROUS_VARIANCE_WGSL,
    SVGF_REPROJECTION_WGSL,
    SVGF_VARIANCE_FROM_MOMENTS_WGSL,
    SVGF_7X7_SPATIAL_FALLBACK_WGSL,
    BMFR_WGSL,
    HDR_LUMINANCE_BILATERAL_WGSL,
    WELFORD_VARIANCE_WGSL,
  } = await import("../../packages/shared-denoisers/src/index.ts");

  // ATROUS_WGSL is NOT standalone: it requires COMMON_WGSL prepended (calls luminance,
  // etc.).  The walkaround-hybrid ATROUS_MODULE adds it via the include-graph
  // (see wgslModules.ts).  This entry is skipped here as a standalone; the
  // walkaround-hybrid/atrous entry above (compiled with its full include-graph) covers it.
  // shaders.push({ name: "shared-denoisers/atrous-raw", wgsl: ATROUS_WGSL });
  // (kept as a skipped comment so the inventory is explicit)
  shaders.push({ name: "shared-denoisers/temporalAccum", wgsl: TEMPORAL_ACCUM_WGSL, entryPoint: "temporalAccumMain" });
  shaders.push({
    name: "shared-denoisers/atrousVariance-standalone",
    wgsl: ATROUS_VARIANCE_WGSL,
    entryPoints: ["svgfVarianceMain", "svgfAtrousMain"],
  });
  shaders.push({ name: "shared-denoisers/svgfReprojection-standalone", wgsl: SVGF_REPROJECTION_WGSL, entryPoint: "svgfReprojMain" });
  shaders.push({ name: "shared-denoisers/svgfVarianceFromMoments-standalone", wgsl: SVGF_VARIANCE_FROM_MOMENTS_WGSL, entryPoint: "svgfVarianceFromMomentsMain" });
  shaders.push({ name: "shared-denoisers/svgf7x7SpatialFallback-standalone", wgsl: SVGF_7X7_SPATIAL_FALLBACK_WGSL, entryPoint: "svgf7x7FallbackMain" });
  shaders.push({ name: "shared-denoisers/bmfr-standalone", wgsl: BMFR_WGSL, entryPoint: "bmfrMain" });
  shaders.push({ name: "shared-denoisers/hdrLuminanceBilateral", wgsl: HDR_LUMINANCE_BILATERAL_WGSL, entryPoint: "hdrLuminanceBilateralMain" });
  shaders.push({ name: "shared-denoisers/welfordVariance-fragment", wgsl: WELFORD_VARIANCE_WGSL });
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4: shared-bvh standalone WGSL strings
// ─────────────────────────────────────────────────────────────────────────────
{
  const { CWBVH_INTERSECT_WGSL } = await import(
    "../../packages/shared-bvh/src/index.ts"
  );

  const cwbvhGateWgsl = `${CWBVH_INTERSECT_WGSL}

@group(0) @binding(0) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(0) @binding(1) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(0) @binding(2) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(0) @binding(3) var<storage, read> cwbvhChildCount: array<u32>;
@group(0) @binding(4) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(5) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> cwbvhGateOut: array<u32>;

@compute @workgroup_size(1)
fn cwbvhGateMain() {
  var ray: CwbvhRay;
  ray.origin = vec3f(0.0, 0.0, 1.0);
  ray.direction = vec3f(0.0, 0.0, -1.0);
  let anyHit = cwbvhIntersectAny(
    &cwbvhNodeBounds,
    &cwbvhChildBoundsPacked,
    &cwbvhChildMeta,
    &cwbvhChildCount,
    &bvh_index,
    &bvh_position,
    ray.origin,
    ray.direction,
    1.0e20,
    1.0e-5,
    0u,
    false,
  );
  let closest = cwbvhIntersectFirstHit(
    &cwbvhNodeBounds,
    &cwbvhChildBoundsPacked,
    &cwbvhChildMeta,
    &cwbvhChildCount,
    &bvh_index,
    &bvh_position,
    ray,
    1.0e-5,
    0u,
    false,
  );
  cwbvhGateOut[0] = select(0u, 1u, anyHit || closest.didHit);
}
`;

  shaders.push({
    name: "shared-bvh/cwbvhIntersect",
    wgsl: applyNagaFix(cwbvhGateWgsl),
    entryPoint: "cwbvhGateMain",
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5: walkaround-rc
//
// The RC probe-ray-cast shader uses ptr<storage> TLAS params from
// TLAS_TRAVERSAL_WGSL and RC-prefixed binding names (rc_tlas_nodes etc.).
// nagaFix.mjs must rename those to the canonical names before naga can compile.
// This section was added 2026-06-10 after the rcEnabled GPU validation error
// (shader parse: "no definition in scope for identifier: tlasNodes") was found
// to be a nagaFix gap — the RC TLAS binding renames were missing from
// RC_SCENE_GLOBAL_RENAMES.  Adding these two shaders to the gate ensures the
// same regression cannot go undetected again.
// ─────────────────────────────────────────────────────────────────────────────
{
  const { PROBE_RAY_CAST_WGSL, CASCADE_MERGE_WGSL } = await import(
    "@vitrum/walkaround-rc"
  );

  shaders.push({
    name: "walkaround-rc/probeRayCast",
    wgsl: applyNagaFix(PROBE_RAY_CAST_WGSL),
    entryPoint: "probeRayCastKernel",
  });

  shaders.push({
    name: "walkaround-rc/cascadeMerge",
    wgsl: CASCADE_MERGE_WGSL,
    entryPoint: "cascadeMergeKernel",
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-test mode: append an intentionally broken shader to the list and assert
// the gate REPORTS it.  The broken shader is a synthetic string never written to
// any production file.
// ─────────────────────────────────────────────────────────────────────────────
if (selfTest) {
  shaders.push({
    name: "__self-test/intentionally-broken",
    wgsl: `
// Intentional type error injected by --self-test mode.
// An i32 value is assigned where a u32 is required; naga/wgpu rejects this.
@compute @workgroup_size(1)
fn main() {
  var x: u32 = -1i;  // i32 literal where u32 expected — naga error
}
`,
  });
  console.log("[shader-gate] --self-test: added 1 intentionally-broken shader");
}

// ── Compile loop ──────────────────────────────────────────────────────────────
let passed = 0;
let failed = 0;
const errors = [];
const pipelineEntries = [];

for (const entry of shaders) {
  const { name, wgsl } = entry;
  if (typeof wgsl !== "string" || wgsl.trim().length === 0) {
    console.warn(`[shader-gate] SKIP  ${name}  (empty or non-string WGSL)`);
    continue;
  }
  const module = device.createShaderModule({ label: name, code: wgsl });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length > 0) {
    failed++;
    const firstLines = errs
      .slice(0, 3)
      .map((m) => `    ${m.lineNum}:${m.linePos}: ${m.message}`)
      .join("\n");
    const errMsg = `[shader-gate] FAIL  ${name}\n${firstLines}`;
    console.error(errMsg);
    errors.push({ name, messages: errs });
  } else {
    passed++;
    console.log(`[shader-gate] OK    ${name}`);
  }
  const entryPoints = entry.entryPoints ?? (entry.entryPoint ? [entry.entryPoint] : []);
  for (const entryPoint of entryPoints) {
    if (hasComputeEntry(wgsl, entryPoint)) {
      pipelineEntries.push({ name, wgsl, entryPoint });
    }
  }
}

const total = passed + failed;
console.log("");
console.log(`[shader-gate] ${total} shader(s) compiled — ${passed} OK, ${failed} FAILED`);

const brokenSelfTest = errors.find((e) => e.name === "__self-test/intentionally-broken");
const realCompileFailures = errors.filter((e) => e.name !== "__self-test/intentionally-broken");
if (selfTest) {
  // In self-test mode, exactly 1 failure is expected (the injected broken shader).
  if (!brokenSelfTest) {
    console.error("[shader-gate] --self-test FAILED: injected broken shader was NOT detected!");
    Deno.exit(1);
  }
  if (realCompileFailures.length > 0) {
    console.error("[shader-gate] --self-test ABORTED: production shaders also failed:");
    for (const e of realCompileFailures) console.error(`  ${e.name}`);
    Deno.exit(1);
  }
}

if (!selfTest && failed > 0) {
  Deno.exit(1);
}

if (!noPipelineGate) {
  await runPipelineCreationGates();
}

if (selfTest) {
  console.log("[shader-gate] --self-test PASSED: injected error was correctly detected.");
}
Deno.exit(0);

function hasComputeEntry(wgsl, entryPoint) {
  const escaped = entryPoint.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`@compute[\\s\\S]{0,240}?fn\\s+${escaped}\\b`);
  return re.test(wgsl);
}

async function runPipelineCreationGates() {
  let pipelinePassed = 0;
  let pipelineFailed = 0;
  const pipelineErrors = [];

  for (const entry of pipelineEntries) {
    try {
      const module = device.createShaderModule({
        label: `${entry.name}/${entry.entryPoint}/pipeline`,
        code: entry.wgsl,
      });
      await device.createComputePipelineAsync({
        label: `${entry.name}/${entry.entryPoint}`,
        layout: "auto",
        compute: { module, entryPoint: entry.entryPoint },
      });
      pipelinePassed++;
      console.log(`[shader-gate] PIPE  ${entry.name}::${entry.entryPoint}`);
    } catch (err) {
      pipelineFailed++;
      const message = String(err?.message ?? err);
      console.error(`[shader-gate] PFAIL ${entry.name}::${entry.entryPoint}\n    ${message}`);
      pipelineErrors.push({ name: entry.name, entryPoint: entry.entryPoint, message });
    }
  }

  const productionVariants = await runWalkaroundProductionPipelineGate();
  pipelinePassed += productionVariants.passed;
  pipelineFailed += productionVariants.failed;
  pipelineErrors.push(...productionVariants.errors);

  console.log("");
  console.log(`[shader-gate] ${pipelinePassed + pipelineFailed} pipeline gate(s) — ${pipelinePassed} OK, ${pipelineFailed} FAILED`);

  if (pipelineFailed > 0) {
    console.error("[shader-gate] Pipeline creation failures:");
    for (const e of pipelineErrors) {
      console.error(`  ${e.name}${e.entryPoint ? `::${e.entryPoint}` : ""}: ${e.message.split("\n")[0]}`);
    }
    Deno.exit(1);
  }
}

async function runWalkaroundProductionPipelineGate() {
  const { compilePipelines } = await import(
    "../../packages/walkaround-hybrid/src/pipeline/pipelineCompiler.ts"
  );

  const nrcCfg = {
    levels: 16,
    featuresPerEntry: 2,
    oneBlobBins: 4,
    width: 64,
    outWidth: 3,
    hidden: 5,
  };
  const variants = [
    { name: "walkaround-hybrid/production-default", opts: {} },
    { name: "walkaround-hybrid/production-gris", opts: { restirPtReuse: true } },
    { name: "walkaround-hybrid/production-ppg", opts: { ppgEnabled: true } },
    { name: "walkaround-hybrid/production-regir", opts: { regirEnabled: true } },
  ];
  if (adapterMaxBG >= 5) {
    variants.push({ name: "walkaround-hybrid/production-nrc", opts: { nrcConfig: nrcCfg } });
  } else {
    console.log(`[shader-gate] PSKIP walkaround-hybrid/production-nrc (requires maxBindGroups>=5; adapter has ${adapterMaxBG})`);
  }

  const pipelineDevice = makeNagaPatchedPipelineDevice(device);
  let passed = 0;
  let failed = 0;
  const errors = [];

  for (const variant of variants) {
    try {
      await compilePipelines(pipelineDevice, {}, "rgba8unorm", variant.opts);
      passed++;
      console.log(`[shader-gate] PIPE  ${variant.name}`);
    } catch (err) {
      failed++;
      const message = String(err?.message ?? err);
      console.error(`[shader-gate] PFAIL ${variant.name}\n    ${message}`);
      errors.push({ name: variant.name, message });
    }
  }

  return { passed, failed, errors };
}

function makeNagaPatchedPipelineDevice(realDevice) {
  return {
    createShaderModule(desc) {
      return realDevice.createShaderModule({
        ...desc,
        code: typeof desc?.code === "string" ? applyNagaFix(desc.code) : desc?.code,
      });
    },
    createBindGroupLayout(desc) {
      return realDevice.createBindGroupLayout(desc);
    },
    createPipelineLayout(desc) {
      return realDevice.createPipelineLayout(desc);
    },
    createComputePipelineAsync(desc) {
      return realDevice.createComputePipelineAsync(desc);
    },
    createRenderPipelineAsync(desc) {
      return realDevice.createRenderPipelineAsync(desc);
    },
  };
}
