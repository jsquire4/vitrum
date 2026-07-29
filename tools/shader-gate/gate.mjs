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
 * Backend-only mode (keeps pt-webgpu independently gateable during unrelated
 * package work):
 *   ... gate.mjs --pt-webgpu-only
 *
 * A substring can focus the expensive native pipeline-creation half while the
 * module-compile half still covers the full selected inventory:
 *   ... gate.mjs --pt-webgpu-only --pipeline-filter=restirpt-producer
 *
 * CPU/WGSL Sobol parity can be executed independently of unrelated shaders:
 *   ... gate.mjs --sobol-parity-only
 *
 * The shared finite-root BDPT strategy mask has the same independent gate:
 *   ... gate.mjs --bdpt-mask-parity-only
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

// ── Parse flags ──────────────────────────────────────────────────────────────
const selfTest = Deno.args.includes("--self-test");
const noPipelineGate = Deno.args.includes("--no-pipeline-gate");
const ptWebgpuOnly = Deno.args.includes("--pt-webgpu-only");
const sobolParityOnly = Deno.args.includes("--sobol-parity-only");
const bdptMaskParityOnly = Deno.args.includes("--bdpt-mask-parity-only");
const pipelineFilter = Deno.args
  .find((arg) => arg.startsWith("--pipeline-filter="))
  ?.slice("--pipeline-filter=".length);

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
const requiredLimits = {};
if (adapterMaxBG > 4) {
  requiredLimits.maxBindGroups = adapterMaxBG;
}
for (const limitName of [
  "maxStorageBuffersPerShaderStage",
  "maxStorageTexturesPerShaderStage",
]) {
  const value = adapter.limits?.[limitName];
  if (typeof value === "number") {
    requiredLimits[limitName] = value;
  }
}
const device = await adapter.requestDevice({ requiredLimits });

if (sobolParityOnly) {
  await runSobolParityGate();
  Deno.exit(0);
}
if (bdptMaskParityOnly) {
  await runBdptStrategyMaskParityGate();
  Deno.exit(0);
}

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
  // Regression kernel for Naga's GLSL robustness lowering around a dynamically
  // selected mip in a sampled 2D-array texture. Keeping the mip signed at the
  // builtin boundary must produce GLSL's textureSize(..., int) overload.
  shaders.push({
    name: "pt-webgpu/material-array-signed-mip",
    wgsl: `
@group(0) @binding(0) var mipTexture: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> mipResult: array<vec4f>;

@compute @workgroup_size(1)
fn materialArraySignedMipMain(@builtin(global_invocation_id) gid: vec3u) {
  let mip = gid.x & 1u;
  let dimensions = textureDimensions(mipTexture, i32(mip));
  mipResult[0] =
    textureLoad(mipTexture, vec2i(0), 0, i32(mip)) +
    vec4f(vec2f(dimensions), 0.0, 0.0);
}
`,
    entryPoint: "materialArraySignedMipMain",
  });

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
    MATERIAL_TEXTURE_GPU_SOURCE_BLIT_WGSL,
    MATERIAL_TEXTURE_MIPMAP_WGSL,
  } = await import(
    "../../packages/pt-webgpu/src/scene/materialTextureArray.ts"
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

  // Opt-in CWBVH closest-hit traversal variant. Visibility reuses successive
  // closest candidates so sidedness, castShadow, and alpha semantics stay on
  // one canonical walker.
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
  shaders.push({
    name: "pt-webgpu/composite-trace-bdpt-sobol",
    wgsl: composePtWebgpuCompositeTraceWgsl(true, { sampling: "sobol" }),
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

  // Runtime material-array render shaders have paired vertex/fragment entry
  // points, so the generic compute-pipeline half of this gate does not apply.
  // Module compilation still validates their complete shipped WGSL source.
  shaders.push({
    name: "pt-webgpu/material-texture-gpu-source-blit",
    wgsl: MATERIAL_TEXTURE_GPU_SOURCE_BLIT_WGSL,
  });
  shaders.push({
    name: "pt-webgpu/material-texture-mipmap",
    wgsl: MATERIAL_TEXTURE_MIPMAP_WGSL,
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
if (!ptWebgpuOnly) {
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
    SPATIAL_GI_MODULE,
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
    SVGF_REAL_ATROUS_MODULE,
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

  const {
    MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
    MATERIAL_ATLAS_GENERATE_MIP_WGSL,
  } = await import(
    "../../packages/walkaround-hybrid/src/pipeline/materialTextureAtlas.ts"
  );

  // Add the exact composed shader shipped at runtime. Shared traversal now uses
  // module-scope value-return loaders, so tool-only source rewriting is banned.
  const addWh = (name, rootModule) => {
    const raw = composeWgsl(rootModule, WGSL_MODULES);
    shaders.push({
      name: `walkaround-hybrid/${name}`,
      wgsl: raw,
      portableSource: true,
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

  // ReSTIR-GI passes (sole live generalized-reuse roots)
  addWh("risGi", RIS_GI_MODULE);
  addWh("temporalGi", TEMPORAL_GI_MODULE);
  addWh("spatialGi", SPATIAL_GI_MODULE);

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
  addWh("svgfRealAtrous", SVGF_REAL_ATROUS_MODULE);
  addWh("bmfr", BMFR_MODULE);

  // Atlas upload shaders are runtime compute pipelines rather than registry
  // modules, so gate their exported source strings directly.
  shaders.push({
    name: "walkaround-hybrid/materialAtlasGpuSourceConvert",
    wgsl: MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
    entryPoint: "main",
  });
  shaders.push({
    name: "walkaround-hybrid/materialAtlasGenerateMip",
    wgsl: MATERIAL_ATLAS_GENERATE_MIP_WGSL,
    entryPoint: "main",
  });

  // PPG update kernel — parametric builder (default maxDTreeNodesPerCell = 341).
  // buildPpgUpdateWgsl returns a raw WGSL string that references:
  //   - STREE_HEADER_F32 / STREE_NODE_STRIDE / DTREE_* (from ppgTreeLayout)
  //   - luminance() (from shared-samplers/luminance)
  // Prepend both — exactly what pipelineCompiler does via the composeWgsl require-graph
  // (PPG_UPDATE_MODULE.requires includes 'ppgTreeLayout', which transitively pulls
  // luminance via the common module chain, but the gate uses a direct prepend for clarity).
  shaders.push({
    name: "walkaround-hybrid/ppgUpdate",
    wgsl: `${WH_LUMINANCE_WGSL}\n${PPG_TREE_LAYOUT_WGSL}\n${buildPpgUpdateWgsl(341)}`,
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
      wgsl: composeWgsl(nrcModule, WGSL_MODULES),
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
    SVGF_REAL_ATROUS_WGSL,
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
  shaders.push({ name: "shared-denoisers/svgfRealAtrous-standalone", wgsl: SVGF_REAL_ATROUS_WGSL, entryPoint: "svgfRealAtrousMain" });
  shaders.push({ name: "shared-denoisers/bmfr-standalone", wgsl: BMFR_WGSL, entryPoint: "bmfrMain" });
  shaders.push({ name: "shared-denoisers/hdrLuminanceBilateral", wgsl: HDR_LUMINANCE_BILATERAL_WGSL, entryPoint: "hdrLuminanceBilateralMain" });
  shaders.push({ name: "shared-denoisers/welfordVariance-fragment", wgsl: WELFORD_VARIANCE_WGSL });
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4: shared-bvh standalone WGSL strings
// ─────────────────────────────────────────────────────────────────────────────
{
  const { SHARED_BVH_PORTABLE_COMPOSITIONS } = await import(
    "./sharedBvhPortableCompositions.mjs"
  );
  // These strings are intentionally compiled verbatim with no textual
  // substitutions. Naga and the Chromium Tint gate import this exact
  // shared inventory, preventing validator-specific composition drift.
  for (const composition of SHARED_BVH_PORTABLE_COMPOSITIONS) {
    shaders.push({
      name: composition.name,
      wgsl: composition.code,
      entryPoint: composition.entryPoint,
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5: walkaround-rc
//
// RC composes the same portable value-return BVH/TLAS loader seam as hybrid.
// Compile the exact exports so validator-specific dependencies cannot be hidden.
// ─────────────────────────────────────────────────────────────────────────────
{
  const { PROBE_RAY_CAST_WGSL, CASCADE_MERGE_WGSL } = await import(
    "@vitrum/walkaround-rc"
  );

  shaders.push({
    name: "walkaround-rc/probeRayCast",
    wgsl: PROBE_RAY_CAST_WGSL,
    entryPoint: "probeRayCastKernel",
    portableSource: true,
  });

  shaders.push({
    name: "walkaround-rc/cascadeMerge",
    wgsl: CASCADE_MERGE_WGSL,
    entryPoint: "cascadeMergeKernel",
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 6: DDGI probe update, relocation/classification, and atlas blend.
// These raw-device shaders are not part of the declarative walkaround pass
// registry, so inventory them explicitly rather than leaving their first
// compilation to a live renderer.
// ─────────────────────────────────────────────────────────────────────────────
{
  const { makeProbeUpdateRaysWGSL } = await import(
    "../../packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts"
  );
  const { PROBE_CLASSIFY_RELOCATE_WGSL } = await import(
    "../../packages/walkaround-hybrid/src/ddgi/wgsl/probeClassifyRelocate.wgsl.ts"
  );
  const {
    makeProbeUpdateBlendIrrWGSL,
    makeProbeUpdateBlendVisWGSL,
  } = await import(
    "../../packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBlend.wgsl.ts"
  );
  const { makeProbeUpdateBorderVisWGSL } = await import(
    "../../packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateBorder.wgsl.ts"
  );

  shaders.push({
    name: "walkaround-hybrid/ddgi-probe-rays",
    wgsl: makeProbeUpdateRaysWGSL(256),
    entryPoint: "probeUpdateRays",
  });
  shaders.push({
    name: "walkaround-hybrid/ddgi-classify-relocate",
    wgsl: PROBE_CLASSIFY_RELOCATE_WGSL,
    entryPoint: "probeClassifyRelocate",
  });
  shaders.push({
    name: "walkaround-hybrid/ddgi-blend-irradiance",
    wgsl: makeProbeUpdateBlendIrrWGSL(),
    entryPoint: "probeUpdateBlendIrradiance",
  });
  shaders.push({
    name: "walkaround-hybrid/ddgi-blend-visibility",
    wgsl: makeProbeUpdateBlendVisWGSL(),
    entryPoint: "probeUpdateBlendVisibility",
  });
  shaders.push({
    name: "walkaround-hybrid/ddgi-border-visibility",
    wgsl: makeProbeUpdateBorderVisWGSL(),
    entryPoint: "probeUpdateBorderVisibility",
  });
}
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
let portablePassed = 0;
let portableFailed = 0;
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
    if (entry.portableSource === true) portableFailed++;
    const firstLines = errs
      .slice(0, 3)
      .map((m) => `    ${m.lineNum}:${m.linePos}: ${m.message}`)
      .join("\n");
    const errMsg = `[shader-gate] FAIL  ${name}\n${firstLines}`;
    console.error(errMsg);
    errors.push({ name, messages: errs });
  } else {
    passed++;
    if (entry.portableSource === true) portablePassed++;
    console.log(`[shader-gate] OK    ${name}`);
  }
  const entryPoints = entry.entryPoints ?? (entry.entryPoint ? [entry.entryPoint] : []);
  for (const entryPoint of entryPoints) {
    if (hasComputeEntry(wgsl, entryPoint)) {
      pipelineEntries.push({ name, wgsl, entryPoint });
    }
  }
}

const portableTotal = portablePassed + portableFailed;
// 29 sole live walkaround/RC roots. The retired temporalGiGris and
// spatialGiGris aliases are intentionally not double-counted as pipelines.
const expectedPortableTotal = ptWebgpuOnly ? 0 : 29;
if (portableTotal !== expectedPortableTotal) {
  failed++;
  const message =
    `portable shipped shader inventory changed: expected ${expectedPortableTotal}, ` +
    `found ${portableTotal}`;
  errors.push({ name: "portable-shipped-inventory", messages: [{ message }] });
  console.error(`[shader-gate] FAIL  portable-shipped-inventory\n    ${message}`);
}

const total = passed + failed;
console.log("");
console.log(`[shader-gate] ${total} shader(s) compiled — ${passed} OK, ${failed} FAILED`);
console.log(
  `[shader-gate] portable shipped walkaround/RC: ${portablePassed}/${portableTotal} ` +
    `compile verbatim (fatal gate${portableFailed === 0 ? " passed" : " FAILED"})`,
);


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
  await runSobolParityGate();
  await runBdptStrategyMaskParityGate();
}

if (selfTest) {
  console.log("[shader-gate] --self-test PASSED: injected error was correctly detected.");
}
Deno.exit(0);

async function runSobolParityGate() {
  const { PT_WEBGPU_SOBOL_RNG_WGSL, SOBOL_FRAME_KEY_WGSL } = await import(
    "../../packages/pt-webgpu/src/wgsl/common.wgsl.ts"
  );
  const {
    SOBOL_DIMENSION_COUNT,
    initOwenScrambledSobolStream,
    nextOwenScrambledSobolU32,
    sobolFrameKey,
    sobolHashCombine,
  } = await import("../../packages/shared-samplers/src/sobol.ts");
  const auditedDrawBudget = 324;
  const boundaryStart = SOBOL_DIMENSION_COUNT - 2;
  const boundaryValueCount = 7;
  const code = `${PT_WEBGPU_SOBOL_RNG_WGSL}
${SOBOL_FRAME_KEY_WGSL}
@group(0) @binding(0) var<storage, read_write> parityOut: array<u32>;

@compute @workgroup_size(1)
fn sobolParityMain() {
  parityOut[0] = ptRngFrameKey(0x12345678u, 0u);
  parityOut[1] = ptRngFrameKey(0x12345678u, 0x0000ffffu);
  parityOut[2] = ptRngFrameKey(0x12345678u, 0x00010000u);
  parityOut[3] = ptRngFrameKey(0x12345678u, 0x00010001u);
  parityOut[4] = ptRngFrameKey(0x12345678u, 0xfffffffeu);
  parityOut[5] = ptRngFrameKey(0x12345678u, 0xffffffffu);

  var state = pcgInit(9u, 10u, ptRngFrameKey(123u, 0u));
  parityOut[6] = state.sampleIndex;
  parityOut[7] = state.dimension;
  parityOut[8] = state.pixelX;
  parityOut[9] = state.pixelY;
  parityOut[10] = state.sequenceKey;
  parityOut[11] = state.rotationTile;
  parityOut[12] = state.fallbackState;
  state.dimension = ${boundaryStart}u;
  for (var i = 0u; i < ${boundaryValueCount}u; i = i + 1u) {
    parityOut[13u + i] = pcgNext(&state);
  }

  var highState = pcgInit(9u, 10u, ptRngFrameKey(123u, 0u));
  var digest = 0u;
  for (var draw = 0u; draw < ${auditedDrawBudget}u; draw = draw + 1u) {
    let value = pcgNext(&highState);
    digest = ptSobolHashCombine(digest, value);
  }
  parityOut[20] = highState.dimension;
  parityOut[21] = digest;
  parityOut[22] = highState.fallbackState;

  // Both pixels select rank zero from the repeated 8x8 rotation tile. Their
  // full pixel identity must still produce distinct, CPU-identical streams.
  var rankZeroA = pcgInit(0u, 0u, ptRngFrameKey(0x12345678u, 99u));
  var rankZeroB = pcgInit(8u, 0u, ptRngFrameKey(0x12345678u, 99u));
  for (var i = 0u; i < 4u; i = i + 1u) {
    parityOut[23u + i] = pcgNext(&rankZeroA);
    parityOut[27u + i] = pcgNext(&rankZeroB);
  }
}
`;
  const frameIndices = [0, 0xffff, 0x10000, 0x10001, 0xfffffffe, 0xffffffff];
  const expected = frameIndices.map((frameIndex) =>
    sobolFrameKey(0x12345678, frameIndex)
  );
  const initialState = initOwenScrambledSobolStream(9, 10, sobolFrameKey(123, 0));
  expected.push(
    initialState.sampleIndex,
    initialState.dimension,
    initialState.pixelX,
    initialState.pixelY,
    initialState.sequenceKey,
    initialState.rotationTile,
    initialState.fallbackState,
  );

  const boundaryState = initOwenScrambledSobolStream(9, 10, sobolFrameKey(123, 0));
  boundaryState.dimension = boundaryStart;
  for (let i = 0; i < boundaryValueCount; i++) {
    expected.push(nextOwenScrambledSobolU32(boundaryState));
  }

  const highState = initOwenScrambledSobolStream(9, 10, sobolFrameKey(123, 0));
  let digest = 0;
  for (let draw = 0; draw < auditedDrawBudget; draw++) {
    digest = sobolHashCombine(digest, nextOwenScrambledSobolU32(highState));
  }
  expected.push(highState.dimension, digest, highState.fallbackState);

  for (const pixelX of [0, 8]) {
    const rankZeroState = initOwenScrambledSobolStream(
      pixelX,
      0,
      sobolFrameKey(0x12345678, 99),
    );
    for (let i = 0; i < 4; i++) {
      expected.push(nextOwenScrambledSobolU32(rankZeroState));
    }
  }
  const byteLength = expected.length * Uint32Array.BYTES_PER_ELEMENT;
  const output = device.createBuffer({
    label: "pt-webgpu/sobol-parity-output",
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "pt-webgpu/sobol-parity-readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const module = device.createShaderModule({
      label: "pt-webgpu/sobol-parity",
      code,
    });
    const info = await module.getCompilationInfo();
    const compileErrors = info.messages.filter((message) => message.type === "error");
    if (compileErrors.length > 0) {
      throw new Error(compileErrors.map((message) => message.message).join(" | "));
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "pt-webgpu/sobol-parity",
      layout: "auto",
      compute: { module, entryPoint: "sobolParityMain" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = Array.from(new Uint32Array(readback.getMappedRange()));
    readback.unmap();
    const mismatch = actual.findIndex((value, index) => value !== expected[index]);
    if (mismatch >= 0 || actual.length !== expected.length) {
      throw new Error(`index ${mismatch}: expected ${expected[mismatch]}, received ${actual[mismatch]}`);
    }
    console.log(
      `[shader-gate] EXEC  pt-webgpu/sobol-parity ` +
      `(${expected.length} u32 values, ${SOBOL_DIMENSION_COUNT} dimensions)`,
    );
  } finally {
    output.destroy();
    readback.destroy();
  }
}

async function runBdptStrategyMaskParityGate() {
  const {
    BDPT_EXPLICIT_STRATEGY_MASK_WGSL,
    bdptExplicitConnectionStrategyIsValid,
  } = await import("../../packages/shared-samplers/src/bdptMIS.ts");
  const pathVertexCountDomain = 20;
  const strategyDomain = 20;
  const lightLimitDomain = 9;
  const eyeLimitDomain = 9;
  const caseCount =
    pathVertexCountDomain * strategyDomain * lightLimitDomain * eyeLimitDomain;
  const code = `${BDPT_EXPLICIT_STRATEGY_MASK_WGSL}
@group(0) @binding(0) var<storage, read_write> parityOut: array<u32>;

@compute @workgroup_size(64)
fn bdptMaskParityMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${caseCount}u) {
    return;
  }
  var code = gid.x;
  let maxEyeVertices = code % ${eyeLimitDomain}u;
  code = code / ${eyeLimitDomain}u;
  let maxLightVertices = code % ${lightLimitDomain}u;
  code = code / ${lightLimitDomain}u;
  let strategyS = code % ${strategyDomain}u;
  let pathVertexCount = code / ${strategyDomain}u;
  parityOut[gid.x] = select(
    0u,
    1u,
    bdptExplicitConnectionStrategyIsValid(
      pathVertexCount,
      strategyS,
      maxLightVertices,
      maxEyeVertices,
    ),
  );
}
`;
  const expected = new Uint32Array(caseCount);
  let index = 0;
  for (let pathVertexCount = 0;
       pathVertexCount < pathVertexCountDomain;
       pathVertexCount++) {
    for (let strategyS = 0; strategyS < strategyDomain; strategyS++) {
      for (let maxLightVertices = 0;
           maxLightVertices < lightLimitDomain;
           maxLightVertices++) {
        for (let maxEyeVertices = 0;
             maxEyeVertices < eyeLimitDomain;
             maxEyeVertices++) {
          expected[index++] = bdptExplicitConnectionStrategyIsValid(
            pathVertexCount,
            strategyS,
            { maxLightVertices, maxEyeVertices },
          ) ? 1 : 0;
        }
      }
    }
  }

  const byteLength = expected.byteLength;
  const output = device.createBuffer({
    label: "pt-webgpu/bdpt-mask-parity-output",
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "pt-webgpu/bdpt-mask-parity-readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const module = device.createShaderModule({
      label: "pt-webgpu/bdpt-mask-parity",
      code,
    });
    const info = await module.getCompilationInfo();
    const compileErrors = info.messages.filter((message) => message.type === "error");
    if (compileErrors.length > 0) {
      throw new Error(compileErrors.map((message) => message.message).join(" | "));
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "pt-webgpu/bdpt-mask-parity",
      layout: "auto",
      compute: { module, entryPoint: "bdptMaskParityMain" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(caseCount / 64));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = Array.from(new Uint32Array(readback.getMappedRange()));
    const mismatch = actual.findIndex((value, caseIndex) =>
      value !== expected[caseIndex]
    );
    readback.unmap();
    if (mismatch >= 0 || actual.length !== expected.length) {
      throw new Error(
        `case ${mismatch}: expected ${expected[mismatch]}, received ${actual[mismatch]}`,
      );
    }
    console.log(
      `[shader-gate] EXEC  pt-webgpu/bdpt-mask-parity (${caseCount} cases)`,
    );
  } finally {
    output.destroy();
    readback.destroy();
  }
}

function hasComputeEntry(wgsl, entryPoint) {
  const escaped = entryPoint.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`@compute[\\s\\S]{0,240}?fn\\s+${escaped}\\b`);
  return re.test(wgsl);
}

async function runPipelineCreationGates() {
  let pipelinePassed = 0;
  let pipelineFailed = 0;
  const pipelineErrors = [];

  const selectedPipelineEntries = pipelineFilter
    ? pipelineEntries.filter((entry) => entry.name.includes(pipelineFilter))
    : pipelineEntries;
  if (pipelineFilter && selectedPipelineEntries.length === 0) {
    throw new Error(`--pipeline-filter=${pipelineFilter} matched no compute pipelines`);
  }

  for (const entry of selectedPipelineEntries) {
    try {
      const module = device.createShaderModule({
        label: `${entry.name}/${entry.entryPoint}/pipeline`,
        code: entry.wgsl,
      });
      device.pushErrorScope("validation");
      device.pushErrorScope("internal");
      let internalError;
      let validationError;
      try {
        await device.createComputePipelineAsync({
          label: `${entry.name}/${entry.entryPoint}`,
          layout: "auto",
          compute: { module, entryPoint: entry.entryPoint },
        });
      } finally {
        internalError = await device.popErrorScope();
        validationError = await device.popErrorScope();
      }
      if (internalError || validationError) {
        const scopedErrors = [internalError, validationError]
          .filter(Boolean)
          .map((error) => {
            const kind = error.constructor?.name ?? "GPUError";
            const message = error.message?.trim();
            return `${kind}: ${message || "<driver returned no diagnostic>"}`;
          })
          .join(" | ");
        throw new Error(scopedErrors);
      }
      pipelinePassed++;
      console.log(`[shader-gate] PIPE  ${entry.name}::${entry.entryPoint}`);
    } catch (err) {
      pipelineFailed++;
      const message = String(err?.message ?? err);
      console.error(`[shader-gate] PFAIL ${entry.name}::${entry.entryPoint}\n    ${message}`);
      pipelineErrors.push({ name: entry.name, entryPoint: entry.entryPoint, message });
    }
  }

  const productionVariants = ptWebgpuOnly
    ? { passed: 0, failed: 0, errors: [] }
    : await runWalkaroundProductionPipelineGate();
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
    { name: "walkaround-hybrid/production-canonical", opts: {} },
    { name: "walkaround-hybrid/production-ppg", opts: { ppgEnabled: true } },
    { name: "walkaround-hybrid/production-regir", opts: { regirEnabled: true } },
  ];
  if (adapterMaxBG >= 5) {
    variants.push({ name: "walkaround-hybrid/production-nrc", opts: { nrcConfig: nrcCfg } });
  } else {
    console.log(`[shader-gate] PSKIP walkaround-hybrid/production-nrc (requires maxBindGroups>=5; adapter has ${adapterMaxBG})`);
  }

  const pipelineDevice = makePipelineGateDevice(device);
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

function makePipelineGateDevice(realDevice) {
  return {
    createShaderModule(desc) {
      return realDevice.createShaderModule(desc);
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
