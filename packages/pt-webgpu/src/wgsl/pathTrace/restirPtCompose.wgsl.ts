/**
 * Compose the four ReSTIR-PT passes as separate WGSL modules. Each pass receives
 * the same shared tracing prefix, only its own entry point, and a portable
 * group-0 relocation for its reuse resources. A combined module is deliberately
 * not exposed: the pass sources reuse binding numbers for different resources,
 * so concatenating them would produce invalid WGSL.
 *
 * Binding map after relocation:
 *   rpt_reservoirOut  g4 b0 → g0 b20
 *   rpt_resCurrent    g4 b1 → g0 b21
 *   rpt_resPrev       g4 b2 → g0 b22
 *   rpt_result        g4 b3 → g0 b23
 *   rptParams         g4 b4 → g0 b24
 *   rpt_resSpatial    g4 b5 → g0 b25
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
  // Fail loudly so a composition change never silently produces broken WGSL
  // (the old silent return-untouched left naga-illegal ptr<storage> bodies in
  // place and the error would only surface at shader compile time, far from the
  // root cause). If any body is missing, the caller changed the reservoir-helper
  // names or the composition order — fix the caller.
  if (roBody == null) {
    throw new Error(
      'monomorphiseReservoirHelpers: could not extract fn loadReservoirPTHero_ro — ' +
      'did the reservoir-helper composition change in reservoirPtHero.wgsl.ts?',
    );
  }
  if (rwBody == null) {
    throw new Error(
      'monomorphiseReservoirHelpers: could not extract fn loadReservoirPTHero_rw — ' +
      'did the reservoir-helper composition change in reservoirPtHero.wgsl.ts?',
    );
  }
  if (stBody == null) {
    throw new Error(
      'monomorphiseReservoirHelpers: could not extract fn storeReservoirPTHero_rw — ' +
      'did the reservoir-helper composition change in reservoirPtHero.wgsl.ts?',
    );
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
 * The shared scene/trace/BSDF modules every reuse entry point composes. Factored
 * so the four per-pass composers stay byte-consistent with each other.
 * `kernelCore` is included for `generatePrimaryRay`
 * (producer) / `projectToNdc`; the resolve pass does not call them but including
 * the module is harmless (unused helper fns don't affect compilation) and keeps
 * the three modules' shared prefix identical.
 */
export interface RestirPtComposeOptions {
  readonly sampling?: PtWebgpuSamplingMode;
}

function reuseSharedPrefix(opts: RestirPtComposeOptions = {}): string {
  const common = composePtWebgpuCommonWgsl(opts.sampling ?? 'pcg');
  return /* wgsl */ `
${common}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_CORE_WGSL}
${LUMINANCE_WGSL}
${HERO_WAVELENGTH_WGSL}
${OPTICAL_WATERTIGHT_TRIANGLE_WGSL}
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
export function composeRestirPtProducerWgsl(opts: RestirPtComposeOptions = {}): string {
  // Producer only STORES into rpt_reservoirOut.
  return monomorphiseReservoirHelpers(
    relocateReuseGroup4ToGroup0(`${reuseSharedPrefix(opts)}${RESTIR_PT_PRODUCER_WGSL}`),
    { store: ['rpt_reservoirOut'] },
  );
}

/**
 * Compose the ReSTIR-PT TEMPORAL pass standalone. Shared prefix + ONLY the
 * temporal body (relocated). Statically uses groups 0,1,2 (the reconnection-
 * visibility traceAny) + group 0's reuse bindings.
 */
export function composeRestirPtTemporalWgsl(opts: RestirPtComposeOptions = {}): string {
  // Temporal LOADS+STORES rpt_resCurrent (rw) and LOADS rpt_resPrev (ro).
  return monomorphiseReservoirHelpers(
    relocateReuseGroup4ToGroup0(`${reuseSharedPrefix(opts)}${RESTIR_PT_TEMPORAL_WGSL}`),
    { ro: ['rpt_resPrev'], rw: ['rpt_resCurrent'], store: ['rpt_resCurrent'] },
  );
}

/**
 * Compose the ReSTIR-PT SPATIAL pass standalone. Shared prefix + ONLY the spatial
 * body (relocated). Statically uses groups 0,1,2 (the reconnection-visibility
 * traceAny) + group 0's reuse bindings. Reads the temporal output (b21, the
 * "current" slot) and writes the spatial output (b25).
 */
export function composeRestirPtSpatialWgsl(opts: RestirPtComposeOptions = {}): string {
  // Spatial READS rpt_resSpatialIn (b21 = the temporal output / "current" slot,
  // declared read_write in the shared layout → promote the `read` decl to match)
  // and WRITES rpt_resSpatialOut (b25). It never writes the slot it samples, so
  // neighbour reads are hazard-free.
  let wgsl = relocateReuseGroup4ToGroup0(`${reuseSharedPrefix(opts)}${RESTIR_PT_SPATIAL_WGSL}`);
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
export function composeRestirPtResolveWgsl(opts: RestirPtComposeOptions = {}): string {
  // Resolve LOADS rpt_resResolved, which sits at the SPATIAL OUTPUT slot
  // (relocated binding 25 — the slot the spatial pass WRITES). The single shared
  // group-0 layout declares that slot `storage` (read_write), so resolve's binding
  // access mode must MATCH it (a `read` shader binding against a read_write layout
  // entry is a validation mismatch). Promote the binding decl to `read_write`
  // (reading a read_write storage global directly is legal); the monomorphised load
  // indexes it directly, so this only changes the access qualifier — never the math.
  let wgsl = relocateReuseGroup4ToGroup0(`${reuseSharedPrefix(opts)}${RESTIR_PT_RESOLVE_WGSL}`);
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

import {
  composePtWebgpuCommonWgsl,
  type PtWebgpuSamplingMode,
} from '../common.wgsl.js';
import {
  OPTICAL_WATERTIGHT_TRIANGLE_WGSL,
} from '@vitrum/shared-bvh';
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
