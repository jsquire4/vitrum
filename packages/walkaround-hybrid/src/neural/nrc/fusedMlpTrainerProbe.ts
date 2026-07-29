// fusedMlpTrainerProbe.ts — DEBUG / VALIDATION probe surface for the FUSED NRC
// MLP trainer (Task 4.5 Theme I).
//
// The production trainer surface (build / setWeights / setBatch / trainStep /
// dispose) is what the path tracer drives every frame. The finite-difference
// gradient check, the loss probe, and the grad/input-grad readbacks are
// VALIDATION-ONLY helpers used by the lavapipe harnesses (fusedMlpHarness.ts,
// nrcEncodeBackwardHarness.ts) — NOT the per-frame hot path. They were
// previously methods on {@link FusedMlpTrainer}; splitting them here keeps the
// production class lean and makes the "this is debug instrumentation" boundary
// explicit.
//
// The probe wraps a trainer instance and issues the same GPU command sequence
// as the production trainer — same encoders, dispatch counts, and readback
// discipline. Its loss readback evaluates NRC's prediction-luminance-relative
// L2 objective and is pinned by `__tests__/fusedMlpTrainerProbe.test.ts`.

import { f16BitsToF32 } from "./fusedMlpTrainer.js";
import type { FusedMlpTrainer } from "./fusedMlpTrainer.js";
import {
  NRC_RELATIVE_L2_EPSILON,
  NRC_RELATIVE_L2_MAX_LUMINANCE,
  nrcRelativeL2Luminance,
} from "./wgsl/fusedMlp.wgsl.js";

export class FusedMlpTrainerProbe {
  constructor(private readonly t: FusedMlpTrainer) {}

  /** Compute grads only (forward+backward+finalize), no Adam — for the FD check.
   *  Finalizes weight, bias AND dL/dX (input) grads. */
  computeGradsStep(): void {
    const t = this.t;
    const d = t.device;
    // clear fixed-point grad buffers (incl. dL/dX). The `!`s are safe: the
    // probe is only ever constructed around a built, undisposed trainer (the
    // trainer's own public methods #assertUsable-guard the disposed case).
    const enc0 = d.createCommandEncoder();
    enc0.clearBuffer(t.gradWfx!); enc0.clearBuffer(t.gradBfx!);
    enc0.clearBuffer(t.gradInputFx!);
    d.queue.submit([enc0.finish()]);
    const enc = d.createCommandEncoder();
    t.recordForward(enc);
    t.recordBackward(enc);
    t.recordGradFinalize(enc, t.gradWfx!, t.gradWf!, t.plan.totalW, t._gradFinUboW!);
    t.recordGradFinalize(enc, t.gradBfx!, t.gradBf!, t.plan.totalB, t._gradFinUboB!);
    t.recordGradFinalize(enc, t.gradInputFx!, t.gradInputF!, t.numSamples * t.spec.inW, t._gradFinUboX!);
    d.queue.submit([enc.finish()]);
  }

  async readGrads(): Promise<{ gw: Float32Array; gb: Float32Array }> {
    const t = this.t;
    return {
      gw: await this.readF32(t.gradWf!, t.plan.totalW),
      gb: await this.readF32(t.gradBf!, t.plan.totalB),
    };
  }

  /** Read back the finalized dL/dX (input gradient), [numSamples × inW]. Used by
   *  the FD gradient check and (debug) inspection of the hash-grid upstream grad. */
  async readInputGrads(): Promise<Float32Array> {
    const t = this.t;
    return this.readF32(t.gradInputF!, t.numSamples * t.spec.inW);
  }

  /** Capture the per-sample prediction luminance used as the detached
   * relative-L2 normalizer. Holding this baseline across +/- parameter probes
   * makes finite differences match the production stop-gradient objective. */
  async captureLossNormalization(): Promise<Float32Array> {
    const t = this.t;
    const d = t.device;
    const enc = d.createCommandEncoder();
    t.recordForward(enc);
    d.queue.submit([enc.finish()]);
    const W = t.spec.W, node = t.node, outW = t.spec.outW;
    const acts = await this.readScalar(t.actsGlob!, t.numSamples * node * W);
    if (outW !== 3) throw new Error(`NRC relative-L2 luminance requires RGB outW=3; got ${outW}`);
    const luminance = new Float32Array(t.numSamples);
    for (let S = 0; S < t.numSamples; S++) {
      const outputBase = S * node * W + (node - 1) * W;
      luminance[S] = nrcRelativeL2Luminance([
        acts[outputBase + 0]!,
        acts[outputBase + 1]!,
        acts[outputBase + 2]!,
      ]);
    }
    return luminance;
  }

  /** Forward-only + CPU relative-L2 luminance loss from prediction readback.
   * `normalizationLuminance` is optional for ordinary monitoring; the FD harness
   * supplies a captured baseline so the denominator remains stop-gradient. */
  async computeLoss(normalizationLuminance?: Float32Array): Promise<number> {
    const t = this.t;
    const d = t.device;
    const enc = d.createCommandEncoder();
    t.recordForward(enc);
    d.queue.submit([enc.finish()]);
    const W = t.spec.W, node = t.node, outW = t.spec.outW;
    if (outW !== 3) throw new Error(`NRC relative-L2 luminance requires RGB outW=3; got ${outW}`);
    if (
      normalizationLuminance != null &&
      normalizationLuminance.length !== t.numSamples
    ) {
      throw new RangeError(
        `NRC loss normalization has ${normalizationLuminance.length} samples; expected ${t.numSamples}`,
      );
    }
    const acts = await this.readScalar(t.actsGlob!, t.numSamples * node * W);
    const tgt = await this.readF32(t.targets!, t.numSamples * outW);
    let loss = 0;
    for (let S = 0; S < t.numSamples; S++) {
      const outputBase = S * node * W + (node - 1) * W;
      const luminance = normalizationLuminance?.[S] ?? nrcRelativeL2Luminance([
        acts[outputBase + 0]!,
        acts[outputBase + 1]!,
        acts[outputBase + 2]!,
      ]);
      const boundedLuminance = Math.min(
        Math.abs(luminance),
        NRC_RELATIVE_L2_MAX_LUMINANCE,
      );
      const denominator =
        boundedLuminance * boundedLuminance + NRC_RELATIVE_L2_EPSILON;
      for (let o = 0; o < outW; o++) {
        const pred = acts[outputBase + o]!;
        const tv = tgt[S * outW + o]!;
        loss += (pred - tv) * (pred - tv) / denominator;
      }
    }
    return loss / (t.numSamples * outW);
  }

  private async readF32(buf: GPUBuffer, count: number): Promise<Float32Array> {
    const d = this.t.device;
    const bytes = Math.max(16, count * 4);
    const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)).subarray(0, count);
    const copy = new Float32Array(out);
    rb.unmap(); rb.destroy();
    return copy;
  }

  // Read a scalar buffer (f16 or f32) back as f32.
  private async readScalar(buf: GPUBuffer, count: number): Promise<Float32Array> {
    const t = this.t;
    if (!t.cfg.useF16) return this.readF32(buf, count);
    const d = t.device;
    const bytes = Math.max(16, (count * 2 + 3) & ~3); // 4-byte-aligned copy size
    const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const bits = new Uint16Array(rb.getMappedRange().slice(0)).subarray(0, count);
    const out = f16BitsToF32(bits);
    rb.unmap(); rb.destroy();
    return out;
  }
}
