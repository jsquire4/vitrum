/**
 * Shared compute-dispatch helpers for walkaround passes + denoisers.
 *
 * Theme E (Task 2.1) — these collapse the copy-paste dispatch boilerplate
 * duplicated across the passes/denoisers under this directory. Each helper
 * reproduces the prior hand-written body byte-for-byte (same pipeline, same
 * bind-group slots, same workgroup dims, same compute-pass label, same
 * order) — they are pure mechanical de-duplication.
 *
 * Why these live here (passes/) rather than in `../Pass.ts`: the
 * single-bind-group helper is consumed only by the concrete passes in this
 * directory and the à-trous-chain helper is consumed by both passes AND
 * denoisers; co-locating them with their callers keeps the
 * already-large shared `Pass.ts` module focused on the Pass abstraction
 * + the frame/scene/ubo `dispatchSharedBindGroupPass` body.
 */

import type { PassDispatchContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

/**
 * Dispatch a single-bind-group compute pass: the identical
 * `beginComputePass(computeDesc(label)) → setPipeline → setBindGroup(0, bg)
 * → dispatchWorkgroups → end` body that 7 walkaround passes re-implement
 * (MotionVectors, GTAOUpsample, IndirectCombine, IndirectTemporalAccum,
 * TemporalAccum, Resolve, SampleBudget).
 *
 * The only per-pass variation is the prebuilt `bg` and which workgroup-count
 * pair to use: full-res 8×8 (`wgX/wgY`, default), the 16×16-sized counts
 * (`wgX16/wgY16`, opt in with `wg16: true`), or the half-res counts
 * (`halfWgX/halfWgY`, opt in with `half: true`). The optional `extraGroups`
 * binds additional groups at explicit slots AFTER slot 0 (e.g. the GRIS
 * scene group at @group(1)). Behavior is byte-identical to the prior inline
 * bodies.
 */
export function dispatchSingleBindGroup(
  ctx: PassDispatchContext,
  pipeline: GPUComputePipeline,
  bg: GPUBindGroup,
  label: PassLabel,
  opts: {
    readonly wg16?: boolean;
    readonly half?: boolean;
    readonly extraGroups?: ReadonlyArray<{ readonly slot: number; readonly group: GPUBindGroup }>;
    /**
     * Override the workgroup counts directly, bypassing the standard
     * `wgX/wgY` / `wg16` / `half` logic. Used by passes that compute their
     * own per-frame workgroup dimensions (e.g. GTAOPass, which divides by the
     * runtime downscale factor).
     */
    readonly dispatchOverride?: { readonly wgX: number; readonly wgY: number };
  } = {},
): void {
  const { encoder, computeDesc, wgX, wgY, wgX16, wgY16, halfWgX, halfWgY } = ctx;
  const dx = opts.dispatchOverride
    ? opts.dispatchOverride.wgX
    : opts.half ? halfWgX : opts.wg16 ? wgX16 : wgX;
  const dy = opts.dispatchOverride
    ? opts.dispatchOverride.wgY
    : opts.half ? halfWgY : opts.wg16 ? wgY16 : wgY;
  const pass = encoder.beginComputePass(computeDesc(label));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  if (opts.extraGroups) {
    for (const { slot, group } of opts.extraGroups) pass.setBindGroup(slot, group);
  }
  pass.dispatchWorkgroups(dx, dy, 1);
  pass.end();
}

/**
 * Run an à-trous ping-pong chain: the loop duplicated ×4 across
 * `denoisers/atrous.ts`, `denoisers/atrousVariance.ts`,
 * `denoisers/svgfReal.ts`, and `passes/AtrousIndirectPass.ts`.
 *
 * The shared shape each site reproduces is:
 *   1. `inputTex` starts at `startTex`.
 *   2. For `iter` in `0..iterations`:
 *      a. `outputTex` = even iter → `pingTex`, odd → `pongTex`.
 *      b. Build the per-iter bind group via `bindGroupFor(iter, inView,
 *         outView)` — each site keeps its EXACT builder + UBO policy
 *         (eager shared `UboRef` vs per-iter transient `createBuffer`),
 *         so the bind-group binding count + UBO lifetime are unchanged.
 *      c. `beginComputePass(computeDesc(labelFor(iter)))` →
 *         `setPipeline(pipeline)` → `setBindGroup(0, bg)` →
 *         `dispatchWorkgroups(wgX, wgY, 1)` → `end()`.
 *      d. `inputTex` = `outputTex` (ping-pong advance).
 *   3. Return the final `inputTex`.
 *
 * `stepWidth = 1 << iter` is computed by the caller inside `bindGroupFor`
 * when its builder needs it (atrous.ts / AtrousIndirectPass), and ignored by
 * the variance/svgf sites that pack iteration into their own UBO. Keeping
 * bind-group construction in the per-site callback is what makes this
 * behavior-identical across the 4 divergent builders.
 *
 * @returns the final output texture (the last `outputTex` written).
 */
export function runAtrousChain(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  opts: {
    readonly iterations: number;
    readonly startTex: GPUTexture;
    readonly pingTex: GPUTexture;
    readonly pongTex: GPUTexture;
    readonly wgX: number;
    readonly wgY: number;
    readonly computeDesc: (label: PassLabel) => GPUComputePassDescriptor;
    readonly textureViewFor?: (texture: GPUTexture) => GPUTextureView;
    /** Build the per-iteration bind group bound at slot 0. Receives the
     *  current input/output texture views so the site's builder can pack
     *  its UBO + assemble its (5- or 6-binding) layout exactly as before. */
    readonly bindGroupFor: (
      iter: number,
      inputView: GPUTextureView,
      outputView: GPUTextureView,
      inputTex: GPUTexture,
      outputTex: GPUTexture,
    ) => GPUBindGroup;
    readonly labelFor: (iter: number) => PassLabel;
  },
): GPUTexture {
  let inputTex: GPUTexture = opts.startTex;
  for (let iter = 0; iter < opts.iterations; iter++) {
    const outputTex = iter % 2 === 0 ? opts.pingTex : opts.pongTex;
    const inputView = opts.textureViewFor?.(inputTex) ?? inputTex.createView();
    const outputView = opts.textureViewFor?.(outputTex) ?? outputTex.createView();
    const bg = opts.bindGroupFor(iter, inputView, outputView, inputTex, outputTex);
    const pass = encoder.beginComputePass(opts.computeDesc(opts.labelFor(iter)));
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(opts.wgX, opts.wgY, 1);
    pass.end();
    inputTex = outputTex;
  }
  return inputTex;
}
