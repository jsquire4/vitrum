// @ts-nocheck
/**
 * tools/lib/readback.mjs
 *
 * GPU texture readback + pixel-stat helpers for the native tooling harnesses.
 * Extracted from tools/behavioral-gate/gate.mjs (D17-1) — pure functions with no
 * gate-specific state. `readbackBgra8` (walkaround swap chain) lives in
 * tools/lib/whHarness.mjs; this module carries the rgba16f/rgba32f blit readback
 * used by the pt-webgpu lane plus the luminance / NaN pixel stats.
 */

/**
 * Readback an rgba16float (or rgba32float) texture via a blit-to-rgba8unorm
 * render pass, then CPU readback.  Returns Uint8Array of RGBA pixels.
 */
export async function readbackAsRgba8(device, srcTex, texW, texH) {
  const dstTex = device.createTexture({
    size: { width: texW, height: texH },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    label: "bg-blit-dst",
  });
  const blitMod = device.createShaderModule({
    label: "bg-blit",
    code: `
      @group(0) @binding(0) var srcTex: texture_2d<f32>;
      @group(0) @binding(1) var srcSmp: sampler;
      struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
      @vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
        var p = array<vec2f,3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
        var o: VO;
        o.pos = vec4f(p[vi], 0.0, 1.0);
        o.uv  = (p[vi] + vec2f(1,1)) * 0.5;
        o.uv.y = 1.0 - o.uv.y;
        return o;
      }
      @fragment fn fs(i: VO) -> @location(0) vec4f {
        return vec4f(clamp(textureSample(srcTex, srcSmp, i.uv).rgb, vec3f(0), vec3f(1)), 1.0);
      }
    `,
  });
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
  ]});
  const pipeline = device.createRenderPipeline({
    label: "bg-blit-pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex:   { module: blitMod, entryPoint: "vs" },
    fragment: { module: blitMod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const bg = device.createBindGroup({ layout: bgl, entries: [
    { binding: 0, resource: srcTex.createView() },
    { binding: 1, resource: device.createSampler({ magFilter: "nearest", minFilter: "nearest" }) },
  ]});
  const enc  = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{
    view: dstTex.createView(), loadOp: "clear", storeOp: "store",
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }]});
  pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  device.queue.submit([enc.finish()]);

  const bpr  = Math.ceil(texW * 4 / 256) * 256;
  const buf  = device.createBuffer({ size: bpr * texH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc2 = device.createCommandEncoder();
  enc2.copyTextureToBuffer({ texture: dstTex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: texH }, { width: texW, height: texH, depthOrArrayLayers: 1 });
  device.queue.submit([enc2.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buf.getMappedRange());
  const pixels = new Uint8Array(texW * texH * 4);
  for (let row = 0; row < texH; row++) {
    pixels.set(mapped.subarray(row * bpr, row * bpr + texW * 4), row * texW * 4);
  }
  buf.unmap(); buf.destroy(); dstTex.destroy();
  return pixels;
}

export function meanLuminance(pixels) {
  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    sum += 0.2126 * (pixels[i]/255) + 0.7152 * (pixels[i+1]/255) + 0.0722 * (pixels[i+2]/255);
  }
  return sum / (pixels.length / 4);
}

export function hasNaN(pixels) {
  for (let i = 0; i < pixels.length; i++) {
    if (Number.isNaN(pixels[i])) return true;
  }
  return false;
}
