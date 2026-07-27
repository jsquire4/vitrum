/**
 * atrousChain.ts — shared à-trous ping-pong scaffolding for the one-shot WebGPU
 * denoiser hosts.
 *
 * Both `runAtrousVarianceWebGPU` (atrousVarianceWebGPU.ts — the à-trous +
 * variance lookup denoiser) and `runSVGFRealWebGPU` (svgfRealWebGPU.ts — the
 * real Schied 2017 SVGF) grew a byte-identical copy of the same two skeletons:
 *
 *   1. A per-call **resource tracker** — push-on-create `GPUTexture[]` /
 *      `GPUBuffer[]` arrays with `trackTexture` / `trackBuffer` closures and a
 *      `finally` teardown that destroys buffers, then textures, then the
 *      ephemeral device. Extracted here as {@link makeResourceTracker}.
 *
 *   2. The **à-trous chain** itself — allocate one tiny uniform buffer per
 *      iteration (per-iter UBOs avoid the shared-UBO write-vs-dispatch reorder
 *      hazard, see the inline note), pack `{ iteration, sigma* }` into each,
 *      build the alternating A→B / B→A ping-pong bind groups, encode one compute
 *      pass per iteration into the caller's encoder, and compute the final
 *      `readTex` from the iteration parity. Extracted here as
 *      {@link buildAtrousChain}.
 *
 * The two callers differ ONLY in: the ping/pong texture pair, the three shared
 * g-buffer bind-group views (normal / depth / variance), and the UBO debug-label
 * prefix. Those are the parameters below. The bind-group ENTRY layout (binding
 * 0 = read ping, 1 = write pong, 2 = normal, 3 = depth, 4 = variance, 5 = UBO),
 * the dispatch loop, and the parity-based `readTex` selection are identical and
 * live here as the single source of truth.
 *
 * Behavior preservation: this extraction is byte-identical to the two prior
 * inlined copies — same UBO size/pack call, same bind-group entries, same
 * dispatch counts, same `readTex` parity. The atrous-variance-vs-svgf albedo
 * demodulation divergence is INTENTIONAL and lives in the callers (see the
 * cross-reference comments there); it is not part of this shared chain.
 */

import {
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  packAtrousVarianceAtrousUniforms,
} from './atrousVarianceBindings.js';
import { ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE } from './wgsl/atrousVariance.wgsl.js';

/**
 * Per-call GPU resource tracker. Every texture/buffer created for a one-shot
 * denoiser pass is registered via the `track*` closures; {@link ResourceTracker.dispose}
 * destroys buffers first, then textures, then runs the ephemeral-device cleanup.
 * Call `dispose()` from the host's `finally` block so a mid-pipeline throw does
 * not leak GPU resources.
 */
export interface ResourceTracker {
  readonly trackTexture: (texture: GPUTexture) => GPUTexture;
  readonly trackBuffer: (buffer: GPUBuffer) => GPUBuffer;
  /** Destroy all tracked buffers, then all tracked textures, then the device. */
  readonly dispose: () => void;
}

/**
 * Build a {@link ResourceTracker}. `destroyEphemeral` is the device-teardown
 * callback returned by `acquireDenoiseDevice` (a no-op when the caller supplied
 * an explicit, externally owned device).
 */
export function makeResourceTracker(destroyEphemeral: () => void): ResourceTracker {
  const textures: GPUTexture[] = [];
  const buffers: GPUBuffer[] = [];
  return {
    trackTexture: (texture: GPUTexture): GPUTexture => {
      textures.push(texture);
      return texture;
    },
    trackBuffer: (buffer: GPUBuffer): GPUBuffer => {
      buffers.push(buffer);
      return buffer;
    },
    dispose: (): void => {
      for (const buffer of buffers) {
        try {
          buffer.destroy();
        } catch {
          // Continue releasing every independently owned resource.
        }
      }
      for (const texture of textures) {
        try {
          texture.destroy();
        } catch {
          // Continue releasing every independently owned resource.
        }
      }
      try {
        destroyEphemeral();
      } catch {
        // Teardown must not mask the dispatch result or its primary failure.
      }
    },
  };
}

/** Arguments for {@link buildAtrousChain}. */
export interface AtrousChainArgs {
  readonly device: GPUDevice;
  /** Compute pipeline with the shared six-binding à-trous ABI. */
  readonly atrousPipeline: GPUComputePipeline;
  /** Command encoder the atrous passes are recorded into (caller owns submit). */
  readonly encoder: GPUCommandEncoder;
  /** Number of à-trous iterations (already clamped by the caller). */
  readonly atrousIterations: number;
  readonly width: number;
  readonly height: number;
  readonly sigmaColor: number;
  readonly sigmaNormal: number;
  readonly sigmaDepth: number;
  /** Ping texture (iter 0 read / even-iter read). */
  readonly pingTex: GPUTexture;
  /** Pong texture (iter 0 write / even-iter write). */
  readonly pongTex: GPUTexture;
  /** Shared g-buffer normal view (bind-group binding 2). */
  readonly normalView: GPUTextureView;
  /** Shared g-buffer depth view (bind-group binding 3). */
  readonly depthView: GPUTextureView;
  /**
   * Initial variance-map view (binding 4). Standalone à-trous reads it on
   * every iteration; real SVGF reads it on iteration zero and then consumes
   * the variance propagated through the color ping-pong alpha channel.
   */
  readonly varianceView: GPUTextureView;
  /** Debug-label prefix for the per-iteration UBOs. */
  readonly uboLabelPrefix: string;
  /** Register each per-iteration UBO for teardown. */
  readonly trackBuffer: (buffer: GPUBuffer) => GPUBuffer;
  /**
   * Optional command-encoding hook invoked immediately after an iteration's
   * compute pass. It is ordered before the next iteration, allowing a caller
   * to preserve an intermediate wavelet level with a GPU copy.
   */
  readonly afterIteration?: (
    iteration: number,
    inputTexture: GPUTexture,
    outputTexture: GPUTexture,
  ) => void;
}

/**
 * Encode the à-trous ping-pong chain into `args.encoder` and return the texture
 * holding the final filtered result (parity of `atrousIterations`).
 *
 * The caller is responsible for: uploading the initial atrous input into
 * `pingTex`, any preceding passes (variance / reprojection), the `queue.submit`,
 * and the readback of the returned texture.
 */
export function buildAtrousChain(args: AtrousChainArgs): GPUTexture {
  const {
    device,
    atrousPipeline,
    encoder,
    atrousIterations,
    width: w,
    height: h,
    sigmaColor,
    sigmaNormal,
    sigmaDepth,
    pingTex,
    pongTex,
    normalView,
    depthView,
    varianceView,
    uboLabelPrefix,
    trackBuffer,
    afterIteration,
  } = args;

  // Pre-allocate one UBO per à-trous iteration so each pass reads its own
  // uniforms. With a shared UBO + per-iter writeBuffer, the driver can re-
  // order writes vs dispatches and produce wrong stepWidth per pass; per-iter
  // UBOs are tiny (32 B × N) and remove that hazard. Write all UBOs once,
  // then the caller batches every pass into a single encoder / single
  // queue.submit.
  const atrousUbos: GPUBuffer[] = [];
  const atrousScratch = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
  for (let iter = 0; iter < atrousIterations; iter += 1) {
    const ubo = trackBuffer(
      device.createBuffer({
        label: `${uboLabelPrefix}${iter}`,
        size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    );
    packAtrousVarianceAtrousUniforms(
      { iteration: iter, sigmaColor, sigmaNormal, sigmaDepth },
      atrousScratch,
    );
    device.queue.writeBuffer(ubo, 0, atrousScratch);
    atrousUbos.push(ubo);
  }

  // Build the alternating bind groups up front: A→B for even iterations,
  // B→A for odd. Each pair is paired with its iteration's UBO.
  const atrousBindGroups: GPUBindGroup[] = atrousUbos.map((ubo, iter) => {
    const isEven = iter % 2 === 0;
    return device.createBindGroup({
      layout: atrousPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: (isEven ? pingTex : pongTex).createView() },
        { binding: 1, resource: (isEven ? pongTex : pingTex).createView() },
        { binding: 2, resource: normalView },
        { binding: 3, resource: depthView },
        { binding: 4, resource: varianceView },
        { binding: 5, resource: { buffer: ubo } },
      ],
    });
  });

  const wg = ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE;
  for (let iter = 0; iter < atrousIterations; iter += 1) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(atrousPipeline);
    pass.setBindGroup(0, atrousBindGroups[iter]);
    pass.dispatchWorkgroups(Math.ceil(w / wg), Math.ceil(h / wg));
    pass.end();
    afterIteration?.(
      iter,
      iter % 2 === 0 ? pingTex : pongTex,
      iter % 2 === 0 ? pongTex : pingTex,
    );
  }

  // After N iterations, the last write went into pongTex when N is odd,
  // pingTex when N is even (the loop's ping/pong swap convention).
  return atrousIterations % 2 === 0 ? pingTex : pongTex;
}
