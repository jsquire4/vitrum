/**
 * FrameCaptureHelper — lazy-compiled offscreen capture pipeline for
 * {@link WalkaroundGPUPipeline.captureOutputFrame}.
 *
 * Extracted from WalkaroundGPUPipeline (D3.3, 2026-06-11) to isolate the
 * three lazily-allocated GPU resources the capture path needs:
 *   - `_capturePipeline`  — rgba8unorm render pipeline (compiled once if
 *     the swap-chain format differs from rgba8unorm)
 *   - `_offscreenTex`     — rgba8unorm RENDER_ATTACHMENT + COPY_SRC texture
 *     (recreated on dimension change)
 *
 * Disposal order and conditions are identical to the original inline code.
 * The pipeline delegates all calls to this helper; the helper owns nothing
 * else.
 */

import { alignedTextureCopyBytesPerRow } from '@vitrum/shared-denoisers';
import type { BGLCache } from './bindGroupLayouts.js';
import { getCompositeBindGroupLayout } from './bindGroupLayouts.js';
import { buildCompositeBindGroup } from './bindGroupBuilders.js';
import { composeWgsl } from './wgslComposer.js';
import {
  COMPOSITE_FRAG_MODULE,
  COMPOSITE_VERT_MODULE,
  WGSL_MODULES,
} from './wgslModules.js';
import type { CompositePass } from './passes/index.js';
import type { FrameResources } from './resourceManager.js';
import {
  assertHybridSwapChainFormat,
  hybridCompositeFragmentConstants,
} from '../presentationTarget.js';

const CAPTURE_FORMAT: GPUTextureFormat = 'rgba8unorm';

export class FrameCaptureHelper {
  /** Lazily-compiled render pipeline targeting rgba8unorm.  Only allocated when
   *  `captureFrame` is first called AND the swap-chain format is not already
   *  rgba8unorm (in which case `compositePass.pipeline` is reused directly).
   *  Destroyed on `dispose()`. */
  private _capturePipeline: GPURenderPipeline | null = null;

  /** Lazily-created offscreen rgba8unorm texture.  Recreated whenever the
   *  render dimensions change (resize-aware).  Destroyed on `dispose()`. */
  private _offscreenTex: { tex: GPUTexture; w: number; h: number } | null = null;

  /**
   * Run the composite pass into an engine-owned offscreen `rgba8unorm` texture
   * and read it back to a Float32Array (RGBA, unorm → [0,1], row-major,
   * top-left origin).
   *
   * This is the 'output' capture path: the same tonemap / OETF / exposure
   * settings last written by `renderFrame` (or `presentLastFrame`) are reused
   * from the live `compositeUbo` buffer — the output is display-encoded,
   * post-OETF, unlike the 'linear' path that reads the pre-tonemap
   * `resolvedTexture`.
   *
   * Pipeline decision — the composite render pipeline is format-specialised
   * (compiled for `swapChainFormat`).  For the offscreen capture target we fix
   * the format at `rgba8unorm` (universally supports RENDER_ATTACHMENT + COPY_SRC):
   *   • If `swapChainFormat === 'rgba8unorm'`, the existing pipeline is reused
   *     (zero extra compile cost).
   *   • Otherwise a second render pipeline targeting `rgba8unorm` is compiled
   *     lazily on the first call and cached for the lifetime of this helper.
   *     It is destroyed in `dispose()`.
   *
   * Pipeline stall: submits copyTextureToBuffer + mapAsync.  Use for
   * debugging / screenshot export, not per-frame readback.
   */
  async captureFrame(
    device: GPUDevice,
    width: number,
    height: number,
    swapChainFormat: GPUTextureFormat,
    compositePass: CompositePass,
    compositeUbo: GPUBuffer,
    bglCache: BGLCache,
    res: FrameResources,
  ): Promise<Float32Array | null> {
    assertHybridSwapChainFormat(
      swapChainFormat,
      'FrameCaptureHelper.captureFrame.swapChainFormat',
    );
    if (width <= 0 || height <= 0) return null;

    // ── Resolve / lazily compile the capture render pipeline ──────────────
    let capturePipeline: GPURenderPipeline;
    if (swapChainFormat === CAPTURE_FORMAT) {
      // Format matches — reuse the existing compiled pipeline.
      capturePipeline = compositePass.pipeline;
    } else {
      // Compile a capture-format pipeline lazily (once per helper lifetime).
      if (this._capturePipeline === null) {
        const compVertSM = device.createShaderModule({
          label: 'comp-vert-capture',
          code: composeWgsl(COMPOSITE_VERT_MODULE, WGSL_MODULES),
        });
        const compFragSM = device.createShaderModule({
          label: 'comp-frag-capture',
          code: composeWgsl(COMPOSITE_FRAG_MODULE, WGSL_MODULES),
        });
        const captureLayout = device.createPipelineLayout({
          bindGroupLayouts: [getCompositeBindGroupLayout(device, bglCache)],
        });
        this._capturePipeline = await device.createRenderPipelineAsync({
          label: 'composite-capture',
          layout: captureLayout,
          vertex:   { module: compVertSM, entryPoint: 'vertMain' },
          fragment: {
            module: compFragSM,
            entryPoint: 'fragMain',
            targets: [{ format: CAPTURE_FORMAT }],
            // The capture target is non-sRGB. Preserve the requested output
            // transfer by running the OETF in the fragment shader.
            constants: hybridCompositeFragmentConstants(CAPTURE_FORMAT),
          },
          primitive: { topology: 'triangle-list' },
        });
      }
      capturePipeline = this._capturePipeline;
    }

    // ── Lazy-create / resize the offscreen capture texture ────────────────
    if (
      this._offscreenTex === null ||
      this._offscreenTex.w !== width ||
      this._offscreenTex.h !== height
    ) {
      this._offscreenTex?.tex.destroy();
      this._offscreenTex = {
        tex: device.createTexture({
          label: 'composite-capture-offscreen',
          size: { width, height },
          format: CAPTURE_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        }),
        w: width,
        h: height,
      };
    }
    const offscreenTex = this._offscreenTex.tex;

    // ── Build composite bind group (reuse the same inputs as renderFrame) ─
    const bg = buildCompositeBindGroup(
      device,
      bglCache,
      res.common.resolvedTexture.createView(),
      compositeUbo,
    );

    // ── Record the render pass into the offscreen texture ─────────────────
    const bytesPerRow = alignedTextureCopyBytesPerRow(width, 4); // 4 bytes per rgba8unorm pixel
    const readSize = bytesPerRow * height;
    const staging = device.createBuffer({
      label: 'composite-capture-staging',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    try {
      const encoder = device.createCommandEncoder({ label: 'composite-capture' });
      const pass = encoder.beginRenderPass({
        label: 'composite-capture',
        colorAttachments: [{
          view: offscreenTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(capturePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3, 1, 0, 0); // fullscreen triangle
      pass.end();
      encoder.copyTextureToBuffer(
        { texture: offscreenTex },
        { buffer: staging, bytesPerRow },
        { width, height, depthOrArrayLayers: 1 },
      );
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(staging.getMappedRange().slice(0));
      staging.unmap();

      // Decode rgba8unorm → float32 RGBA, 4 channels per pixel, [0, 1] range.
      const out = new Float32Array(width * height * 4);
      for (let y = 0; y < height; y++) {
        const rowOff = y * bytesPerRow;
        for (let x = 0; x < width; x++) {
          const srcOff = rowOff + x * 4;
          const dstOff = (y * width + x) * 4;
          out[dstOff]     = (src[srcOff]!     & 0xff) / 255;
          out[dstOff + 1] = (src[srcOff + 1]! & 0xff) / 255;
          out[dstOff + 2] = (src[srcOff + 2]! & 0xff) / 255;
          out[dstOff + 3] = (src[srcOff + 3]! & 0xff) / 255;
        }
      }
      return out;
    } finally {
      staging.destroy();
    }
  }

  /**
   * Release the lazily-compiled capture pipeline and offscreen texture.
   * Must be called from `WalkaroundGPUPipeline.dispose()` in the same
   * position the original inline destroy calls occupied.
   */
  dispose(): void {
    this._capturePipeline = null;
    this._offscreenTex?.tex.destroy();
    this._offscreenTex = null;
  }
}
