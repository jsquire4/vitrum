/**
 * ResolvePass — Sprint 9 sparse-shade gap-fill pass.
 *
 * Currently runs in passthrough mode (checkerboardOn=0) — every pixel
 * copies through from `writeAccum` to `common.resolvedTexture`. When
 * `shade.wgsl` is upgraded to write sparsely, flip `checkerboardOn=1`
 * in the resolve UBO and the gap-fill branch becomes active. Until
 * then this pass costs one extra texture copy per frame but produces
 * identical output.
 *
 * Sprint 9 layout note: resolve uses `@workgroup_size(8, 8, 1)` —
 * dispatch with `wgX/wgY` (`ceil(W/8)`), NOT the 16×16-sized counts.
 */

import { buildResolveBindGroup, type UboRef } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class ResolvePass implements Pass {
  readonly id = 'resolve' as const;
  readonly dependencies: readonly string[] = ['temporalAccum'];
  readonly passLabels: readonly PassLabel[] = ['resolve'];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _uboRef: UboRef;

  constructor(
    pipeline: GPUComputePipeline,
    uboRef: UboRef,
  ) {
    this._pipeline = pipeline;
    this._uboRef = uboRef;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, wgX, wgY, frameState, frameCount, width, height } = ctx;

    // ResolveUniforms: u32 W, u32 H, u32 frameParity, u32 checkerboardOn (16 bytes).
    device.queue.writeBuffer(
      this._uboRef.buf!,
      0,
      new Uint32Array([width, height, frameCount & 1, 0]),
    );
    const bg = buildResolveBindGroup(
      device, bglCache,
      this._uboRef.buf!,
      frameState.writeAccum.createView(),           // current radiance (post-accum)
      frameState.readAccum.createView(),            // prev radiance (other ping-pong slot)
      resources.common.motionVectorTexture.createView(),
      resources.common.resolvedTexture.createView(),
    );
    const pass = encoder.beginComputePass(computeDesc('resolve'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}
