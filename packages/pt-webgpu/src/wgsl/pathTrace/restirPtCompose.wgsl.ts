/**
 * restirPtCompose.wgsl.ts — composes the ReSTIR-PT reuse unit as a SINGLE WGSL
 * module so it compiles standalone (one `device.createShaderModule` for all three
 * @compute entry points: restirPtProduce / restirPtTemporal / restirPtResolve).
 *
 * This MIRRORS the shared-module concatenation order of
 * `composePtWebgpuTraceWgsl` (pathTraceBruteforce.wgsl.ts) — which this file does
 * NOT modify and does NOT touch the default `PT_WEBGPU_TRACE_WGSL` export — so
 * every symbol the reuse passes reference (traceClosest / traceAny / evaluateBrdf
 * / brdfDirectionalPdf / cosineHemisphereSample / glossyReflectionSample /
 * decodeMaterial / hitMaterialId / sampleEnvironment* / powerHeuristic / pcgInit /
 * rand_f32 / luminance / INV_PI / PI / safe_normalize / generatePrimaryRay /
 * FrameParams `params` / the @group(0..3) scene bindings) resolves from the SAME
 * shared modules the megakernel uses. `kernelCore` is composed for
 * `generatePrimaryRay` (the producer's primary-ray generation); it also carries
 * projectToNdc / causticMode / RR helpers, none of which collide with the other
 * composed modules.
 *
 * The reuse-specific tail REPLACES the kernel:
 *   - RESERVOIR_PT_HERO_WITH_SHIFT_WGSL — RESTIR_PT_SHIFT_WGSL (the FD-validated
 *     reconnection-shift Jacobian) + RestirPtParams + the ReservoirPTHero ADT +
 *     target/finalize/pairwise-MIS helpers. (Includes the shift exactly ONCE; the
 *     shared trace modules above do NOT include restirPtShift, so no double-def.)
 *   - RESTIR_PT_PRODUCER_WGSL  — @compute restirPtProduce
 *   - RESTIR_PT_TEMPORAL_WGSL  — @compute restirPtTemporal
 *   - RESTIR_PT_RESOLVE_WGSL   — @compute restirPtResolve
 *
 * ── Bind-group note (the honest maxBindGroups constraint) ───────────────────
 * Composing the FULL-tier shared `PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL` declares
 * @group(0..3) (core/G-buffer, analytics+env+area-lights, TLAS+BDPT, light-tree+
 * material-textures). The reuse passes add their resources at @group(4)
 * (reservoirs + result + RestirPtParams). The composed unit therefore needs an
 * adapter with maxBindGroups ≥ 5. WebGPU only GUARANTEES 4 (indices 0..3), so the
 * WIRING step must either (a) require maxBindGroups ≥ 5 for the ReSTIR-PT path, or
 * (b) relocate the @group(4) resources into a free slot of an existing group
 * (e.g. high bindings of @group(3), which the producer/temporal/resolve do not
 * otherwise use). Each pass statically uses only a SUBSET of @group(0..3) (the
 * producer/temporal trace ⇒ groups 0/1/2; resolve ⇒ group 0 + its BRDF), so a
 * per-pass pipeline layout need only bind the groups that pass touches plus
 * @group(4). This is flagged as wiring/uncertainty in the agent report.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE WIRING RESOLUTION (chosen + implemented — 2026-06-04)
 * ════════════════════════════════════════════════════════════════════════════
 * TWO blockers force the wiring shape, both VERIFIED by reading the modules:
 *
 *  (1) The single combined `composePtWebgpuReuseWgsl()` below is NOT compilable:
 *      the three pass bodies declare DIFFERENT vars at the SAME @group(4)/@binding
 *      slot — @binding(1) is `rpt_resCurrent`(rw) in temporal AND `rpt_resResolved`
 *      (read) in resolve; @binding(4) is `rptParams` thrice. A WGSL module may not
 *      declare two module-scope vars at the same group+binding → naga rejects it.
 *      (The combined unit exists ONLY as a string-contract scaffold; the contract
 *      test never compiles it on a GPU, which is why this slipped past the goldens
 *      — exactly the gap the naga gate closes.) So the WIRING composes ONE module
 *      PER ENTRY POINT (`composeRestirPt{Producer,Temporal,Resolve}Wgsl`) — each
 *      declares only its own pass's bindings, so there is no slot collision.
 *
 *  (2) Option (a) (require maxBindGroups ≥ 5) is rejected: it is non-portable
 *      (WebGPU guarantees only 4). We take option (b), RELOCATING the @group(4)
 *      resources into FREE high bindings of @group(0) (bindings 20..24 — verified
 *      unused; the megakernel's group-0 tops out at binding 13). @group(0) is the
 *      ONE group EVERY reuse entry point already uses (FrameParams `params` is
 *      @group(0)@binding(1)), so adding the reservoir/result/params there keeps
 *      every pass at ≤ 4 groups (0,1,2,3) on a guaranteed-maxBindGroups=4 adapter.
 *      The relocation is a COMPOSE-LEVEL string rewrite of the `@group(4) @binding(b)`
 *      decl lines → `@group(0) @binding(20+b)`; the pass SHADER BODIES are NOT
 *      edited (the rewrite runs on the composed string). Binding map:
 *        rpt_reservoirOut  g4 b0 → g0 b20
 *        rpt_resCurrent    g4 b1 → g0 b21   (resolve's rpt_resResolved is the SAME
 *        rpt_resPrev       g4 b2 → g0 b22    buffer at the same b21 — both are the
 *        rpt_result        g4 b3 → g0 b23    "current" reservoir slot, by binding #)
 *        rptParams         g4 b4 → g0 b24
 *
 * Per-pass static group usage (verified from the call graph — see the wiring
 * report): producer & temporal trace (traceClosest/traceAny → groups 0,1,2);
 * resolve only evaluates the BRDF (pure math) + reads `params` (group 0). The
 * per-pass explicit pipeline layouts in `gpuResources.ts` declare exactly those
 * groups, so each pipeline validates with ZERO unused-group entries.
 */

/**
 * Relocate the reuse passes' `@group(4) @binding(b)` declaration lines onto
 * `@group(0) @binding(RPT_GROUP0_BINDING_BASE + b)`. A COMPOSE-LEVEL rewrite (it
 * runs on the composed WGSL string, never on the pass source files) so each
 * per-pass module needs only groups 0..3 — portable on a guaranteed
 * maxBindGroups = 4 adapter (no maxBindGroups ≥ 5 requirement). The regex matches
 * only the module-scope binding DECLARATIONS (`@group(4) @binding(N)`); it cannot
 * touch comments (the pass headers say "@group(4)" in prose, but those lack the
 * adjacent `@binding(...)`, so they are left as documentation).
 */
export const RPT_GROUP0_BINDING_BASE = 20;
function relocateReuseGroup4ToGroup0(wgsl: string): string {
  return wgsl.replace(
    /@group\(4\)\s+@binding\((\d+)\)/g,
    (_m, b: string) => `@group(0) @binding(${RPT_GROUP0_BINDING_BASE + Number(b)})`,
  );
}

/**
 * naga-compat transform (the analogue of wsl-gpu's `applyNagaGapFix`, but applied
 * AT COMPOSE for the runtime engine — not only at the gate).
 *
 * THE NAGA GAP (verified on dzn/lavapipe via the compile gate): the verified
 * reservoir helpers `loadReservoirPTHero_ro` / `loadReservoirPTHero_rw` /
 * `storeReservoirPTHero_rw` take their buffer as a `ptr<storage, array<u32>, …>`
 * parameter. **The wgpu/naga build behind WebGPU on dzn/lavapipe rejects ANY
 * storage-address-space pointer as a function parameter** — read OR read_write —
 * because the NON-CORE `unrestricted_pointer_parameters` WGSL feature is not even
 * an exposed enum there (the engine is host-owns-lifecycle, so it cannot require
 * a device feature regardless). naga's message: "pointer of space Storage … which
 * can't be passed into functions." The SHIPPING walkaround-hybrid reservoirGi has
 * the identical `loadReservoirGI_{ro,rw}(ptr<storage,…>)` shape; its NRC siblings
 * explicitly call this out as "WGSL-illegal without unrestricted_pointer_parameters"
 * (risGiNrc.wgsl.ts / nrcEncodeBackward.wgsl.ts) and AVOID it by reading the
 * module-scope global directly.
 *
 * THE FIX (touches ZERO math — a mechanical monomorphisation): for each reservoir
 * GLOBAL a pass uses, emit a SPECIALIZED copy of the load/store helper whose body
 * indexes that module-scope global DIRECTLY (the `buf` pointer parameter removed),
 * then rewrite the call sites `loadReservoirPTHero_ro(&G, i)` →
 * `loadReservoirPTHero__G(i)` and DELETE the original pointer-parameter helpers so
 * naga never sees them. The specialized body is the verified body with the single
 * token `buf` renamed to the global's name — the bitcast/stride math is
 * byte-identical (direct module-global indexing is exactly what walkaround's NRC
 * comment prescribes as the legal form). This runs on the COMPOSED module string;
 * the pass SHADER BODIES + reservoirPtHero.wgsl.ts are NOT edited on disk.
 *
 * `globals` is the SET of reservoir globals THIS per-pass module references (e.g.
 * the producer uses only `rpt_reservoirOut`; the temporal uses `rpt_resCurrent` +
 * `rpt_resPrev`; the resolve uses `rpt_resResolved`). Specializing only the used
 * globals keeps each module minimal.
 */
function monomorphiseReservoirHelpers(
  wgsl: string,
  globals: { ro?: readonly string[]; rw?: readonly string[]; store?: readonly string[] },
): string {
  let out = wgsl;

  // Extract a top-level `fn NAME(...) {...}` body (brace-balanced from the `fn`
  // line to its column-0 closing brace). The reservoir helpers are emitted at
  // column 0 by reservoirPtHero.wgsl.ts, so the closing `}` is the first line that
  // is exactly `}`.
  const extractFn = (src: string, name: string): { full: string; start: number } | null => {
    const sig = `\nfn ${name}(`;
    const i = src.indexOf(sig);
    if (i < 0) return null;
    const close = src.indexOf('\n}', i);
    if (close < 0) return null;
    return { full: src.slice(i + 1, close + 2), start: i + 1 };
  };

  const roBody = extractFn(out, 'loadReservoirPTHero_ro');
  const rwBody = extractFn(out, 'loadReservoirPTHero_rw');
  const stBody = extractFn(out, 'storeReservoirPTHero_rw');
  if (roBody == null || rwBody == null || stBody == null) {
    // Helpers not present (composition changed) — leave the string untouched; the
    // naga gate will catch any resulting error.
    return out;
  }

  // Build the specialized variants for each requested global. A specialization
  // renames the function and substitutes the `buf` parameter token with the
  // global name throughout the body (and drops the `buf: ptr<…>, ` parameter).
  const specialize = (
    body: string,
    origName: string,
    global: string,
    extraParamsTail: string,
  ): string => {
    // Replace the signature: `fn ORIG(buf: ptr<…>, REST) -> …` →
    //                         `fn ORIG__GLOBAL(REST) -> …`.
    // The signature line is the first line of `body`.
    const nl = body.indexOf('\n');
    const sigLine = body.slice(0, nl);
    const rest = body.slice(nl);
    // Strip the `buf: ptr<storage, array<u32>, ...>, ` parameter from the sig.
    const newSig = sigLine
      .replace(`fn ${origName}(`, `fn ${origName}__${global}(`)
      .replace(/buf:\s*ptr<storage,\s*array<u32>,\s*(?:read|read_write)>,\s*/, '');
    // In the body, every `buf[` indexing becomes `GLOBAL[`. (`buf` appears only as
    // the indexed array in these helpers — verified — so a token replace is safe.)
    const newBody = rest.replace(/\bbuf\[/g, `${global}[`);
    void extraParamsTail;
    return newSig + newBody;
  };

  const specializations: string[] = [];
  for (const g of globals.ro ?? []) {
    specializations.push(specialize(roBody.full, 'loadReservoirPTHero_ro', g, ''));
  }
  for (const g of globals.rw ?? []) {
    specializations.push(specialize(rwBody.full, 'loadReservoirPTHero_rw', g, ''));
  }
  for (const g of globals.store ?? []) {
    specializations.push(specialize(stBody.full, 'storeReservoirPTHero_rw', g, ''));
  }

  // Remove the three original pointer-parameter helper definitions (naga rejects
  // them even though no specialized call references them anymore).
  for (const b of [stBody, rwBody, roBody]) {
    out = out.replace(b.full, '');
  }

  // Inject the specializations where the reservoir helpers used to live (right
  // before the now-removed `restirPtTargetAt`, a stable anchor that survives).
  const anchor = '\nfn restirPtTargetAt(';
  const ai = out.indexOf(anchor);
  const block = `\n// — naga-compat: monomorphised reservoir helpers (no storage-ptr params) —\n${specializations.join('\n')}\n`;
  if (ai >= 0) {
    out = out.slice(0, ai) + block + out.slice(ai);
  } else {
    out = block + out;
  }

  // Rewrite the call sites: `loadReservoirPTHero_ro(&G, ` → `loadReservoirPTHero_ro__G(`
  // etc. Covers all three helpers and every global.
  out = out.replace(
    /\b(loadReservoirPTHero_ro|loadReservoirPTHero_rw|storeReservoirPTHero_rw)\(&(\w+),\s*/g,
    (_m, fn: string, global: string) => `${fn}__${global}(`,
  );

  return out;
}

/**
 * The shared scene/trace/BSDF modules every reuse entry point composes (the SAME
 * set + order the combined unit uses, minus the per-pass tail). Factored so the
 * three per-pass composers stay byte-consistent with each other and with the
 * combined contract unit. `kernelCore` is included for `generatePrimaryRay`
 * (producer) / `projectToNdc`; the resolve pass does not call them but including
 * the module is harmless (unused helper fns don't affect compilation) and keeps
 * the three modules' shared prefix identical.
 */
function reuseSharedPrefix(): string {
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
${PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL}
${RESERVOIR_PT_HERO_WITH_SHIFT_WGSL}
`;
}

/**
 * Compose the ReSTIR-PT PRODUCER as a standalone, naga-compilable module: the
 * shared prefix + ONLY the producer body, with the producer's @group(4) decls
 * relocated to @group(0) high bindings. Statically uses groups 0,1,2 (trace+NEE)
 * + group 0's reuse bindings. The wiring builds its pipeline layout to match.
 */
export function composeRestirPtProducerWgsl(): string {
  // Producer only STORES into rpt_reservoirOut.
  return monomorphiseReservoirHelpers(
    relocateReuseGroup4ToGroup0(`${reuseSharedPrefix()}${RESTIR_PT_PRODUCER_WGSL}`),
    { store: ['rpt_reservoirOut'] },
  );
}

/**
 * Compose the ReSTIR-PT TEMPORAL pass standalone. Shared prefix + ONLY the
 * temporal body (relocated). Statically uses groups 0,1,2 (the reconnection-
 * visibility traceAny) + group 0's reuse bindings.
 */
export function composeRestirPtTemporalWgsl(): string {
  // Temporal LOADS+STORES rpt_resCurrent (rw) and LOADS rpt_resPrev (ro).
  return monomorphiseReservoirHelpers(
    relocateReuseGroup4ToGroup0(`${reuseSharedPrefix()}${RESTIR_PT_TEMPORAL_WGSL}`),
    { ro: ['rpt_resPrev'], rw: ['rpt_resCurrent'], store: ['rpt_resCurrent'] },
  );
}

/**
 * Compose the ReSTIR-PT SPATIAL pass standalone. Shared prefix + ONLY the spatial
 * body (relocated). Statically uses groups 0,1,2 (the reconnection-visibility
 * traceAny) + group 0's reuse bindings. Reads the temporal output (b21, the
 * "current" slot) and writes the spatial output (b25).
 */
export function composeRestirPtSpatialWgsl(): string {
  // Spatial READS rpt_resSpatialIn (b21 = the temporal output / "current" slot,
  // declared read_write in the shared layout → promote the `read` decl to match)
  // and WRITES rpt_resSpatialOut (b25). It never writes the slot it samples, so
  // neighbour reads are hazard-free.
  let wgsl = relocateReuseGroup4ToGroup0(`${reuseSharedPrefix()}${RESTIR_PT_SPATIAL_WGSL}`);
  wgsl = wgsl.replace(
    /(var<storage,)\s*read(>\s+rpt_resSpatialIn)/,
    (_m, pre: string, post: string) => `${pre} read_write${post}`,
  );
  // Spatial source calls loadReservoirPTHero_ro(&rpt_resSpatialIn, …) and stores
  // via storeReservoirPTHero_rw(&rpt_resSpatialOut, …).
  return monomorphiseReservoirHelpers(wgsl, {
    ro: ['rpt_resSpatialIn'],
    store: ['rpt_resSpatialOut'],
  });
}

/**
 * Compose the ReSTIR-PT RESOLVE pass standalone. Shared prefix + ONLY the resolve
 * body (relocated). Statically uses group 0 only (evaluateBrdf is pure math; the
 * pass reads `params` + the reservoir/result reuse bindings, all in group 0).
 */
export function composeRestirPtResolveWgsl(): string {
  // Resolve LOADS rpt_resResolved, which sits at the SPATIAL OUTPUT slot
  // (relocated binding 25 — the slot the spatial pass WRITES). The single shared
  // group-0 layout declares that slot `storage` (read_write), so resolve's binding
  // access mode must MATCH it (a `read` shader binding against a read_write layout
  // entry is a validation mismatch). Promote the binding decl to `read_write`
  // (reading a read_write storage global directly is legal); the monomorphised load
  // indexes it directly, so this only changes the access qualifier — never the math.
  let wgsl = relocateReuseGroup4ToGroup0(`${reuseSharedPrefix()}${RESTIR_PT_RESOLVE_WGSL}`);
  wgsl = wgsl.replace(
    /(var<storage,)\s*read(>\s+rpt_resResolved)/,
    (_m, pre: string, post: string) => `${pre} read_write${post}`,
  );
  // The resolve source calls `loadReservoirPTHero_ro(&rpt_resResolved, …)`, so the
  // call-site rewrite produces `loadReservoirPTHero_ro__rpt_resResolved` — specialize
  // under the `ro` key so the GENERATED name matches. The body indexes the global
  // directly; the global being `read_write` (promoted above to match the shared
  // layout slot) is irrelevant to a read, so the `ro`/`rw` template choice only
  // affects the (now-removed) parameter access, never the math.
  return monomorphiseReservoirHelpers(wgsl, { ro: ['rpt_resResolved'] });
}

import { PT_WEBGPU_COMMON_WGSL } from '../common.wgsl.js';
import {
  HAMMERSLEY_WGSL,
  HERO_WAVELENGTH_WGSL,
  LUMINANCE_WGSL,
  OCTAHEDRAL_CORE_WGSL,
} from '@vitrum/shared-samplers';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from './material.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL } from './intersection.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from './bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from './connect.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL } from './kernelCore.wgsl.js';
import { RESERVOIR_PT_HERO_WITH_SHIFT_WGSL } from './reservoirPtHero.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from './restirPtProducer.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from './restirPtTemporal.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from './restirPtSpatial.wgsl.js';
import { RESTIR_PT_RESOLVE_WGSL } from './restirPtResolve.wgsl.js';

/**
 * Compose the full ReSTIR-PT reuse unit (producer + temporal + resolve) as one
 * WGSL string. The kernel module is intentionally ABSENT — the megakernel's
 * `@compute fn main` is the default trace's entry, not this unit's.
 *
 * Note: the BDPT modules are NOT composed here (the reuse passes do not call any
 * BDPT helper, and composing them would pull in @group(2) bdpt buffers the reuse
 * passes never use). The MNEE / caustic modules are likewise omitted — the reuse
 * passes reference none of their symbols. Only the modules whose symbols the
 * three passes actually reference are concatenated, which keeps the compiled unit
 * minimal while still resolving every identifier.
 */
export function composePtWebgpuReuseWgsl(): string {
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
${PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL}
${RESERVOIR_PT_HERO_WITH_SHIFT_WGSL}
${RESTIR_PT_PRODUCER_WGSL}
${RESTIR_PT_TEMPORAL_WGSL}
${RESTIR_PT_SPATIAL_WGSL}
${RESTIR_PT_RESOLVE_WGSL}
`;
}
