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

export class FloatTextureNonFiniteError extends Error {
  constructor(stats) {
    super(
      `float texture contains ${stats.nonFiniteComponentCount} non-finite component(s) ` +
      `(${stats.nanComponentCount} NaN, ${stats.infiniteComponentCount} Inf) ` +
      `across ${stats.nonFinitePixelCount} pixel(s)`,
    );
    this.name = "FloatTextureNonFiniteError";
    this.stats = stats;
  }
}

/**
 * Scan the source texture in its float domain. This must run before any
 * normalized-byte blit, because rgba8 conversion clamps or canonicalizes the
 * NaN/Inf evidence.
 */
export async function scanFloatTextureNonFinite(device, srcTex, texW, texH) {
  if (!Number.isSafeInteger(texW) || texW <= 0 ||
      !Number.isSafeInteger(texH) || texH <= 0) {
    throw new RangeError("float texture scan dimensions must be positive safe integers");
  }
  const module = device.createShaderModule({
    label: "tool-float-non-finite-scan",
    code: `
      struct Counts {
        nanComponents: atomic<u32>,
        infiniteComponents: atomic<u32>,
        nonFinitePixels: atomic<u32>,
        padding: u32,
      }

      @group(0) @binding(0) var srcTex: texture_2d<f32>;
      @group(0) @binding(1) var<storage, read_write> counts: Counts;

      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) id: vec3u) {
        let dimensions = textureDimensions(srcTex);
        if (id.x >= dimensions.x || id.y >= dimensions.y) {
          return;
        }
        let texel = textureLoad(srcTex, vec2i(id.xy), 0);
        var pixelIsNonFinite = false;
        for (var channel = 0u; channel < 4u; channel = channel + 1u) {
          let bits = bitcast<u32>(texel[channel]);
          let exponent = bits & 0x7f800000u;
          if (exponent == 0x7f800000u) {
            let mantissa = bits & 0x007fffffu;
            if (mantissa == 0u) {
              atomicAdd(&counts.infiniteComponents, 1u);
            } else {
              atomicAdd(&counts.nanComponents, 1u);
            }
            pixelIsNonFinite = true;
          }
        }
        if (pixelIsNonFinite) {
          atomicAdd(&counts.nonFinitePixels, 1u);
        }
      }
    `,
  });
  const layout = device.createBindGroupLayout({
    label: "tool-float-non-finite-scan-layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const pipeline = device.createComputePipeline({
    label: "tool-float-non-finite-scan-pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "main" },
  });
  const countsBuffer = device.createBuffer({
    label: "tool-float-non-finite-counts",
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const readbackBuffer = device.createBuffer({
    label: "tool-float-non-finite-readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    device.queue.writeBuffer(countsBuffer, 0, new Uint32Array(4));
    const bindGroup = device.createBindGroup({
      label: "tool-float-non-finite-scan-bind-group",
      layout,
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: { buffer: countsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({
      label: "tool-float-non-finite-scan-encoder",
    });
    const pass = encoder.beginComputePass({
      label: "tool-float-non-finite-scan-pass",
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(texW / 8), Math.ceil(texH / 8));
    pass.end();
    encoder.copyBufferToBuffer(countsBuffer, 0, readbackBuffer, 0, 16);
    device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(readbackBuffer.getMappedRange());
    const nanComponentCount = counts[0];
    const infiniteComponentCount = counts[1];
    const nonFinitePixelCount = counts[2];
    const nonFiniteComponentCount = nanComponentCount + infiniteComponentCount;
    return {
      hasNonFinite: nonFiniteComponentCount > 0,
      nanComponentCount,
      infiniteComponentCount,
      nonFiniteComponentCount,
      nonFinitePixelCount,
    };
  } finally {
    if (readbackBuffer.mapState === "mapped") readbackBuffer.unmap();
    readbackBuffer.destroy();
    countsBuffer.destroy();
  }
}

export async function assertFloatTextureFinite(device, srcTex, texW, texH) {
  const stats = await scanFloatTextureNonFinite(device, srcTex, texW, texH);
  if (stats.hasNonFinite) throw new FloatTextureNonFiniteError(stats);
  return stats;
}

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
