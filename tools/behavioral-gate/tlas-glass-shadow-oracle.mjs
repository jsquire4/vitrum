#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env
// @ts-nocheck
/**
 * H32 TLAS glass-shadow behavioral oracle.
 *
 * Builds a tiny production-WGSL scene:
 *   camera ray -> glass triangle at z=1 -> opaque triangle at z=2
 *
 * The shader imports the canonical shared-bvh BVH/TLAS traversal strings and
 * dispatches traceTlasAny three ways:
 *   1. tMax before the opaque triangle, skipGlass=false => occluded by glass
 *   2. tMax before the opaque triangle, skipGlass=true  => no occluder
 *   3. tMax past the opaque triangle, skipGlass=true    => opaque occluder
 *
 * This distinguishes the historical H32 failure mode: TLAS any-hit respected
 * skipGlass in an initial pre-test, then a follow-up closest-hit traversal
 * ignored skipGlass and let the glass pane shadow the ray anyway.
 */

import { BVH_INTERSECT_WGSL } from "../../packages/shared-bvh/src/wgsl/bvhIntersect.wgsl.ts";
import { TLAS_TRAVERSAL_WGSL } from "../../packages/shared-bvh/src/wgsl/tlasTraversal.wgsl.ts";

const LEAF_FLAG = 0xffff0000;

function bvhNode(min, max, offset, count) {
  const buf = new ArrayBuffer(32);
  const view = new DataView(buf);
  for (let i = 0; i < 3; i += 1) view.setFloat32(i * 4, min[i], true);
  for (let i = 0; i < 3; i += 1) view.setFloat32((3 + i) * 4, max[i], true);
  view.setUint32(24, offset, true);
  view.setUint32(28, LEAF_FLAG | count, true);
  return buf;
}

function concatBuffers(...buffers) {
  const bytes = buffers.reduce((sum, b) => sum + b.byteLength, 0);
  const out = new Uint8Array(bytes);
  let offset = 0;
  for (const b of buffers) {
    out.set(new Uint8Array(b), offset);
    offset += b.byteLength;
  }
  return out;
}

function identityTransforms() {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function shaderCode() {
  const raw = /* wgsl */ `
${BVH_INTERSECT_WGSL}

fn safe_normalize(v: vec3f) -> vec3f {
  let len2 = dot(v, v);
  return select(vec3f(0.0, 1.0, 0.0), v * inverseSqrt(len2), len2 > 1e-20);
}

${TLAS_TRAVERSAL_WGSL}

@group(0) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read> tlasInstanceIndices: array<u32>;
@group(0) @binding(2) var<storage, read> tlasBlasRoots: array<u32>;
@group(0) @binding(3) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(0) @binding(4) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(0) @binding(5) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(6) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(7) var<storage, read> bvh: array<BVHNode>;
@group(0) @binding(8) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let origin = vec3f(0.0, 0.0, 0.0);
  let dir = vec3f(0.0, 0.0, 1.0);
  let eps = 1e-6;

  // Glass is in range, opaque is not.
  result[0] = select(0u, 1u, traceTlasAny(
    1u,
    origin,
    dir,
    1.5,
    eps,
    false,
  ));

  // Same segment: glass is ignored, no opaque occluder yet.
  result[1] = select(0u, 1u, traceTlasAny(
    1u,
    origin,
    dir,
    1.5,
    eps,
    true,
  ));

  // Longer segment: glass is ignored, then opaque geometry is found.
  result[2] = select(0u, 1u, traceTlasAny(
    1u,
    origin,
    dir,
    3.0,
    eps,
    true,
  ));

  // Sanity guard: nothing should hit before the glass pane.
  result[3] = select(0u, 1u, traceTlasAny(
    1u,
    origin,
    dir,
    0.5,
    eps,
    false,
  ));

}
`;
  return raw;
}

function makeBuffer(device, label, data, usage) {
  const size = Math.max(4, (data.byteLength + 3) & ~3);
  const buffer = device.createBuffer({ label, size, usage });
  device.queue.writeBuffer(buffer, 0, data.buffer ?? data);
  return buffer;
}

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.error("[tlas-glass-shadow] ERROR: no WebGPU adapter available");
  Deno.exit(1);
}
const maxStorageBuffersPerShaderStage =
  adapter.limits?.maxStorageBuffersPerShaderStage ?? 8;
const device = await adapter.requestDevice(
  maxStorageBuffersPerShaderStage > 8
    ? { requiredLimits: { maxStorageBuffersPerShaderStage } }
    : {},
);

const module = device.createShaderModule({
  label: "h32-tlas-glass-shadow-oracle",
  code: shaderCode(),
});
const info = await module.getCompilationInfo();
const errors = info.messages.filter((m) => m.type === "error");
if (errors.length > 0) {
  console.error("[tlas-glass-shadow] WGSL compile errors:");
  for (const e of errors) console.error(`  line ${e.lineNum}: ${e.message}`);
  Deno.exit(1);
}

device.pushErrorScope("validation");
const pipeline = await device.createComputePipelineAsync({
  label: "h32-tlas-glass-shadow-oracle",
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const pipelineError = await device.popErrorScope();
if (pipelineError) {
  console.error(`[tlas-glass-shadow] pipeline error: ${pipelineError.message ?? pipelineError}`);
  Deno.exit(1);
}

const glassPacked = 5 << 4; // transmission nibble > 4 means glass to skipGlass.
const bvhIndex = new Uint32Array([
  0, 1, 2, glassPacked,
  3, 4, 5, 0,
]);
const bvhPosition = new Float32Array([
  -1, -1, 1, 0,
   1, -1, 1, 0,
   0,  1, 1, 0,
  -1, -1, 2, 0,
   1, -1, 2, 0,
   0,  1, 2, 0,
]);
const blasNode = concatBuffers(bvhNode([-1, -1, 1], [1, 1, 2], 0, 2));
const tlasNode = concatBuffers(bvhNode([-1, -1, 1], [1, 1, 2], 0, 1));
const oneU32 = new Uint32Array([0]);
const identity = identityTransforms();
const result = device.createBuffer({
  label: "h32-tlas-glass-shadow-result",
  size: 16,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const readback = device.createBuffer({
  label: "h32-tlas-glass-shadow-readback",
  size: 16,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const bindGroup = device.createBindGroup({
  label: "h32-tlas-glass-shadow-bindings",
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: makeBuffer(device, "tlasNodes", tlasNode, storage) } },
    { binding: 1, resource: { buffer: makeBuffer(device, "tlasInstanceIndices", oneU32, storage) } },
    { binding: 2, resource: { buffer: makeBuffer(device, "tlasBlasRoots", oneU32, storage) } },
    { binding: 3, resource: { buffer: makeBuffer(device, "tlasInstanceWorldToLocal", identity, storage) } },
    { binding: 4, resource: { buffer: makeBuffer(device, "tlasInstanceLocalToWorld", identity, storage) } },
    { binding: 5, resource: { buffer: makeBuffer(device, "bvh_index", bvhIndex, storage) } },
    { binding: 6, resource: { buffer: makeBuffer(device, "bvh_position", bvhPosition, storage) } },
    { binding: 7, resource: { buffer: makeBuffer(device, "bvh", blasNode, storage) } },
    { binding: 8, resource: { buffer: result } },
  ],
});

device.pushErrorScope("validation");
device.pushErrorScope("internal");
device.pushErrorScope("out-of-memory");
const encoder = device.createCommandEncoder({ label: "h32-tlas-glass-shadow" });
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(1, 1, 1);
pass.end();
encoder.copyBufferToBuffer(result, 0, readback, 0, 16);
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();
const oomError = await device.popErrorScope();
const internalError = await device.popErrorScope();
const validationError = await device.popErrorScope();
for (const err of [validationError, internalError, oomError]) {
  if (err) {
    console.error(`[tlas-glass-shadow] GPU error: ${err.message ?? err}`);
    Deno.exit(1);
  }
}
await readback.mapAsync(GPUMapMode.READ);
const out = Array.from(new Uint32Array(readback.getMappedRange()).slice(0, 4));
readback.unmap();

const expected = [1, 0, 1, 0];
if (out.some((v, i) => v !== expected[i])) {
  console.error("[tlas-glass-shadow] FAIL");
  console.error(`  got      [${out.join(", ")}]`);
  console.error(`  expected [${expected.join(", ")}]`);
  console.error("  slots: [short/no-skip, short/skip-glass, long/skip-glass, before-glass]");
  Deno.exit(1);
}

console.log("[tlas-glass-shadow] PASS");
console.log("  short/no-skip      : glass occludes");
console.log("  short/skip-glass   : glass ignored, no opaque hit yet");
console.log("  long/skip-glass    : opaque triangle behind glass occludes");
console.log("  before-glass guard : no false positive");
