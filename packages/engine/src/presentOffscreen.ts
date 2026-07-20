// presentOffscreen.ts — blit an offscreen backend's FrameOutput texture to the
// host canvas (V1-1 / Wave-R2).
//
// Offscreen-texture backends (e.g. @vitrum/pt-webgpu) render to an INTERNAL
// GPUTexture and return it via `FrameOutput.primaryRadiance`; they never present
// to the host canvas. A canvas-owning lifecycle helper (`attachVitrum`, and the
// React `<VitrumCanvas>` progressive path) must therefore blit that texture onto
// the canvas's WebGPU context every frame — otherwise the canvas is black (an
// auto-selected pt-webgpu engine) or freezes on the last realtime frame (after a
// progressive walkaround→PT handoff).
//
// This module owns a tiny, device-scoped textured-quad render pipeline: a
// fullscreen-triangle vertex shader + a fragment shader that samples the source
// texture through a linear sampler and writes it to the current canvas texture.
// It configures the passed GPUCanvasContext once on construction and re-builds
// the per-frame bind group (the source texture changes across a progressive
// handoff, and is recycled per-frame by the backend), so `present(source)` is a
// single render pass per frame with no persistent GPU allocation beyond the
// pipeline + sampler.
//
// HOST-OWNS-LIFECYCLE: the presenter does NOT own the device (the backend /
// progressive facade does) and does NOT destroy it. `dispose()` only drops the
// presenter's own references; the pipeline/sampler are GC'd with the module.

/** WGSL for the fullscreen-triangle blit. The vertex shader emits a single
 *  oversized triangle covering the viewport (no vertex buffer); the fragment
 *  shader samples the source through a linear sampler. The source
 *  (`presentTexture`) is already tonemapped + OETF-encoded by the backend's
 *  present pass, so the fragment writes it straight through — the presenter does
 *  NOT re-tonemap or re-encode. */
const PRESENT_OFFSCREEN_WGSL = /* wgsl */ `
struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0)       uv:  vec2f,
}

@vertex
fn vsMain(@builtin(vertex_index) vid: u32) -> VsOut {
  // Fullscreen triangle: clip-space corners (-1,-1)-(3,-1)-(-1,3), UVs 0..1
  // with V flipped so the source texture's top-left maps to the canvas top-left.
  var out: VsOut;
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  out.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  out.uv = vec2f(x, y);
  return out;
}

@group(0) @binding(0) var srcTex:     texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4f {
  return textureSampleLevel(srcTex, srcSampler, in.uv, 0.0);
}
`;

/** A device-scoped presenter that blits an offscreen source texture to a canvas.
 *  Construct once per (device, canvas, context) and call `present(source)` each
 *  frame. Idempotent `dispose()`. */
export interface OffscreenPresenter {
  /** Blit `source` to the current canvas texture in a single render pass. */
  present(source: GPUTexture): void;
  /** Drop the presenter's references. Does NOT destroy the device. Idempotent. */
  dispose(): void;
}

export interface CreateOffscreenPresenterArgs {
  /** The GPUDevice that owns the source textures (the backend's device). The
   *  presenter allocates against it but does NOT own its lifecycle. */
  readonly device: GPUDevice;
  /** The host canvas (or OffscreenCanvas) the source is presented onto. */
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  /** The canvas's WebGPU context. The presenter configures it (format = the
   *  preferred canvas format unless `format` is given). */
  readonly context: GPUCanvasContext;
  /** Override the canvas texture format. Defaults to
   *  `navigator.gpu.getPreferredCanvasFormat()`. */
  readonly format?: GPUTextureFormat;
}

function preferredCanvasFormat(): GPUTextureFormat {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    const gpu = navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat };
    const fmt = gpu.getPreferredCanvasFormat?.();
    if (fmt != null) return fmt;
  }
  return 'bgra8unorm';
}

/**
 * Build a device-scoped textured-quad presenter that blits an offscreen source
 * texture onto a canvas WebGPU context each frame.
 *
 * On construction: configures `context` (format = `format ?? preferred`,
 * `opaque` alpha) and builds the fullscreen-triangle pipeline + a linear sampler.
 * The bind group is rebuilt per `present` because the source texture identity
 * changes across a progressive handoff (walkaround→pt-webgpu) and is recycled
 * per-frame by the backend.
 */
export function createOffscreenPresenter(args: CreateOffscreenPresenterArgs): OffscreenPresenter {
  const { device, context } = args;
  const format = args.format ?? preferredCanvasFormat();

  context.configure({ device, format, alphaMode: 'opaque' });

  const module = device.createShaderModule({
    label: 'vitrum.engine.presentOffscreen.module',
    code: PRESENT_OFFSCREEN_WGSL,
  });

  const pipeline = device.createRenderPipeline({
    label: 'vitrum.engine.presentOffscreen.pipeline',
    layout: 'auto',
    vertex: { module, entryPoint: 'vsMain' },
    fragment: { module, entryPoint: 'fsMain', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const sampler = device.createSampler({
    label: 'vitrum.engine.presentOffscreen.sampler',
    magFilter: 'linear',
    minFilter: 'linear',
  });

  let disposed = false;

  return {
    present(source: GPUTexture): void {
      if (disposed) return;
      const bindGroup = device.createBindGroup({
        label: 'vitrum.engine.presentOffscreen.bindGroup',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: source.createView() },
          { binding: 1, resource: sampler },
        ],
      });
      const view = context.getCurrentTexture().createView();
      const encoder = device.createCommandEncoder({ label: 'vitrum.engine.presentOffscreen.encoder' });
      const pass = encoder.beginRenderPass({
        label: 'vitrum.engine.presentOffscreen.pass',
        colorAttachments: [
          {
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      // Fullscreen triangle — 3 vertices, no vertex buffer.
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
    dispose(): void {
      disposed = true;
    },
  };
}
