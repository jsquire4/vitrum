#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write=tools/behavioral-gate
// @ts-nocheck
/**
 * CWBVH GPU parity oracle.
 *
 * Builds a mixed scene (opaque grid + one glass triangle), packs it through the
 * shared CPU CWBVH builder, dispatches `CWBVH_INTERSECT_WGSL` on a real WebGPU
 * adapter, and compares closest-hit + any-hit results against the CPU oracle.
 */

import {
  CWBVH_INTERSECT_WGSL,
  buildCompressedWideBvh,
  intersectCompressedWideBvhAnyHit,
  intersectCompressedWideBvhFirstHit,
  packCwbvhBuildBoundsForWgsl,
  reorderCwbvhTrianglePayloads,
} from "../../packages/shared-bvh/src/index.ts";
import { applyNagaFix } from "../shader-gate/nagaFix.mjs";

const WRITE_STATUS = Deno.args.includes("--write-status");
const STATUS_PATH = new URL("./cwbvh-parity-status.json", import.meta.url);
const GLASS_PAYLOAD = 5 << 4;
const DIST_TOL = 1e-4;

function makeScene() {
  const size = 5;
  const positions = [];
  for (let y = 0; y <= size; y += 1) {
    for (let x = 0; x <= size; x += 1) {
      positions.push(x, y, 0, 0);
    }
  }

  const indices = [];
  const materialIds = [];
  const payloads = [];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const v0 = y * (size + 1) + x;
      const v1 = v0 + 1;
      const v2 = v0 + size + 2;
      const v3 = v0 + size + 1;
      indices.push(v0, v1, v2, 0, v0, v2, v3, 0);
      materialIds.push(0, 0);
      payloads.push(0, 0);
    }
  }

  const glassBase = positions.length / 4;
  positions.push(
    0, 0, 1, 0,
    0.8, 0, 1, 0,
    0, 0.8, 1, 0,
  );
  indices.push(glassBase, glassBase + 1, glassBase + 2, GLASS_PAYLOAD);
  materialIds.push(1);
  payloads.push(GLASS_PAYLOAD);

  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
    materialIds: new Uint32Array(materialIds),
    payloads: new Uint32Array(payloads),
  };
}

function makeRays() {
  return [
    { label: "glass-short", origin: [0.2, 0.3, 2], direction: [0, 0, -1], tMax: 1.25 },
    { label: "glass-long", origin: [0.2, 0.3, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "opaque-grid", origin: [2.2, 3.3, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "miss-outside", origin: [6.0, 6.0, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "short-before-surface", origin: [1.2, 1.2, 2], direction: [0, 0, -1], tMax: 0.5 },
  ];
}

function rayBuffer(rays) {
  const packed = new Float32Array(rays.length * 8);
  for (let i = 0; i < rays.length; i += 1) {
    const ray = rays[i];
    packed[i * 8 + 0] = ray.origin[0];
    packed[i * 8 + 1] = ray.origin[1];
    packed[i * 8 + 2] = ray.origin[2];
    packed[i * 8 + 3] = ray.tMax;
    packed[i * 8 + 4] = ray.direction[0];
    packed[i * 8 + 5] = ray.direction[1];
    packed[i * 8 + 6] = ray.direction[2];
    packed[i * 8 + 7] = 0;
  }
  return packed;
}

function shaderCode(rayCount, nodeCount) {
  const raw = /* wgsl */ `
${CWBVH_INTERSECT_WGSL}

const CWBVH_GATE_RAY_COUNT: u32 = ${rayCount}u;
const CWBVH_GATE_NODE_COUNT: u32 = ${nodeCount}u;

struct CwbvhGateRay {
  originAndTMax: vec4f,
  direction: vec4f,
};

@group(0) @binding(0) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(0) @binding(1) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(0) @binding(2) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(0) @binding(3) var<storage, read> cwbvhChildCount: array<u32>;
@group(0) @binding(4) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(5) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(6) var<storage, read> cwbvhGateRays: array<CwbvhGateRay>;
@group(0) @binding(7) var<storage, read_write> cwbvhGateOut: array<vec4u>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let rayIndex = gid.x;
  if (rayIndex >= CWBVH_GATE_RAY_COUNT) {
    return;
  }

  let packedRay = cwbvhGateRays[rayIndex];
  var ray: CwbvhRay;
  ray.origin = packedRay.originAndTMax.xyz;
  ray.direction = packedRay.direction.xyz;
  let tMax = packedRay.originAndTMax.w;

  let closestNoSkip = cwbvhIntersectFirstHit(
    &cwbvhNodeBounds,
    &cwbvhChildBoundsPacked,
    &cwbvhChildMeta,
    &cwbvhChildCount,
    &bvh_index,
    &bvh_position,
    ray,
    1.0e-5,
    CWBVH_GATE_NODE_COUNT,
    false,
  );
  let anyNoSkip = cwbvhIntersectAny(
    &cwbvhNodeBounds,
    &cwbvhChildBoundsPacked,
    &cwbvhChildMeta,
    &cwbvhChildCount,
    &bvh_index,
    &bvh_position,
    ray.origin,
    ray.direction,
    tMax,
    1.0e-5,
    CWBVH_GATE_NODE_COUNT,
    false,
  );

  let closestSkip = cwbvhIntersectFirstHit(
    &cwbvhNodeBounds,
    &cwbvhChildBoundsPacked,
    &cwbvhChildMeta,
    &cwbvhChildCount,
    &bvh_index,
    &bvh_position,
    ray,
    1.0e-5,
    CWBVH_GATE_NODE_COUNT,
    true,
  );
  let anySkip = cwbvhIntersectAny(
    &cwbvhNodeBounds,
    &cwbvhChildBoundsPacked,
    &cwbvhChildMeta,
    &cwbvhChildCount,
    &bvh_index,
    &bvh_position,
    ray.origin,
    ray.direction,
    tMax,
    1.0e-5,
    CWBVH_GATE_NODE_COUNT,
    true,
  );

  cwbvhGateOut[rayIndex * 2u + 0u] = vec4u(
    select(0u, 1u, closestNoSkip.didHit),
    closestNoSkip.triIndex,
    bitcast<u32>(closestNoSkip.dist),
    select(0u, 1u, anyNoSkip),
  );
  cwbvhGateOut[rayIndex * 2u + 1u] = vec4u(
    select(0u, 1u, closestSkip.didHit),
    closestSkip.triIndex,
    bitcast<u32>(closestSkip.dist),
    select(0u, 1u, anySkip),
  );
}
`;
  return applyNagaFix(raw);
}

function makeBuffer(device, label, data, usage) {
  const size = Math.max(4, (data.byteLength + 3) & ~3);
  const buffer = device.createBuffer({ label, size, usage });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function f32BitsToNumber(bits) {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setUint32(0, bits >>> 0, true);
  return view.getFloat32(0, true);
}

function gpuTriToSigned(tri) {
  return tri === 0xffffffff ? -1 : tri;
}

function compareHit(label, mode, gpu, cpu, mismatches) {
  const gpuDidHit = gpu[0] === 1;
  const gpuTri = gpuTriToSigned(gpu[1]);
  const gpuDist = f32BitsToNumber(gpu[2]);
  if (gpuDidHit !== cpu.didHit) {
    mismatches.push(`${label}/${mode}: didHit GPU=${gpuDidHit} CPU=${cpu.didHit}`);
    return;
  }
  if (!cpu.didHit) return;
  if (gpuTri !== cpu.triangleIndex) {
    mismatches.push(`${label}/${mode}: tri GPU=${gpuTri} CPU=${cpu.triangleIndex}`);
  }
  if (Math.abs(gpuDist - cpu.dist) > DIST_TOL) {
    mismatches.push(`${label}/${mode}: dist GPU=${gpuDist} CPU=${cpu.dist}`);
  }
}

const scene = makeScene();
const builtBase = buildCompressedWideBvh(scene.positions, scene.indices, scene.materialIds, {
  maxLeafTriangles: 1,
});
const built = {
  ...builtBase,
  reorderedIndices: reorderCwbvhTrianglePayloads(builtBase, scene.payloads),
};
const childBoundsPacked = packCwbvhBuildBoundsForWgsl(built);
const rays = makeRays();
const rayData = rayBuffer(rays);
const outputWordsPerRay = 8;
const output = new Uint32Array(rays.length * outputWordsPerRay);

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.error("[cwbvh-parity] ERROR: no WebGPU adapter available");
  Deno.exit(1);
}
const maxStorageBuffersPerShaderStage = adapter.limits?.maxStorageBuffersPerShaderStage ?? 8;
const device = await adapter.requestDevice(
  maxStorageBuffersPerShaderStage > 8
    ? { requiredLimits: { maxStorageBuffersPerShaderStage } }
    : {},
);

const module = device.createShaderModule({
  label: "cwbvh-parity-oracle",
  code: shaderCode(rays.length, built.cwbvhNodeCount),
});
const info = await module.getCompilationInfo();
const errors = info.messages.filter((m) => m.type === "error");
if (errors.length > 0) {
  console.error("[cwbvh-parity] WGSL compile errors:");
  for (const e of errors) console.error(`  line ${e.lineNum}: ${e.message}`);
  Deno.exit(1);
}

device.pushErrorScope("validation");
const pipeline = await device.createComputePipelineAsync({
  label: "cwbvh-parity-oracle",
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const pipelineError = await device.popErrorScope();
if (pipelineError) {
  console.error(`[cwbvh-parity] pipeline error: ${pipelineError.message ?? pipelineError}`);
  Deno.exit(1);
}

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const result = device.createBuffer({
  label: "cwbvh-parity-result",
  size: output.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const readback = device.createBuffer({
  label: "cwbvh-parity-readback",
  size: output.byteLength,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const bindGroup = device.createBindGroup({
  label: "cwbvh-parity-bindings",
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: makeBuffer(device, "cwbvhNodeBounds", built.cwbvhNodeBounds, storage) } },
    { binding: 1, resource: { buffer: makeBuffer(device, "cwbvhChildBoundsPacked", childBoundsPacked, storage) } },
    { binding: 2, resource: { buffer: makeBuffer(device, "cwbvhChildMeta", built.cwbvhChildMeta, storage) } },
    { binding: 3, resource: { buffer: makeBuffer(device, "cwbvhChildCount", built.cwbvhChildCount, storage) } },
    { binding: 4, resource: { buffer: makeBuffer(device, "bvh_index", built.reorderedIndices, storage) } },
    { binding: 5, resource: { buffer: makeBuffer(device, "bvh_position", scene.positions, storage) } },
    { binding: 6, resource: { buffer: makeBuffer(device, "cwbvhGateRays", rayData, storage) } },
    { binding: 7, resource: { buffer: result } },
  ],
});

device.pushErrorScope("validation");
device.pushErrorScope("internal");
device.pushErrorScope("out-of-memory");
const encoder = device.createCommandEncoder({ label: "cwbvh-parity-oracle" });
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(rays.length / 64), 1, 1);
pass.end();
encoder.copyBufferToBuffer(result, 0, readback, 0, output.byteLength);
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();
const oomError = await device.popErrorScope();
const internalError = await device.popErrorScope();
const validationError = await device.popErrorScope();
for (const err of [validationError, internalError, oomError]) {
  if (err) {
    console.error(`[cwbvh-parity] GPU error: ${err.message ?? err}`);
    Deno.exit(1);
  }
}

await readback.mapAsync(GPUMapMode.READ);
output.set(new Uint32Array(readback.getMappedRange()).slice(0, output.length));
readback.unmap();

const mismatches = [];
for (let i = 0; i < rays.length; i += 1) {
  const ray = rays[i];
  const cpuNoSkip = intersectCompressedWideBvhFirstHit(built, scene.positions, ray);
  const cpuSkip = intersectCompressedWideBvhFirstHit(built, scene.positions, ray, { skipGlass: true });
  const cpuAnyNoSkip = intersectCompressedWideBvhAnyHit(built, scene.positions, ray, { tMax: ray.tMax });
  const cpuAnySkip = intersectCompressedWideBvhAnyHit(built, scene.positions, ray, { tMax: ray.tMax, skipGlass: true });

  const gpuNoSkip = output.slice(i * outputWordsPerRay, i * outputWordsPerRay + 4);
  const gpuSkip = output.slice(i * outputWordsPerRay + 4, i * outputWordsPerRay + 8);
  compareHit(ray.label, "closest", gpuNoSkip, cpuNoSkip, mismatches);
  compareHit(ray.label, "closest-skip-glass", gpuSkip, cpuSkip, mismatches);
  if ((gpuNoSkip[3] === 1) !== cpuAnyNoSkip) {
    mismatches.push(`${ray.label}/any: GPU=${gpuNoSkip[3] === 1} CPU=${cpuAnyNoSkip}`);
  }
  if ((gpuSkip[3] === 1) !== cpuAnySkip) {
    mismatches.push(`${ray.label}/any-skip-glass: GPU=${gpuSkip[3] === 1} CPU=${cpuAnySkip}`);
  }
}

const status = {
  generatedAt: new Date().toISOString(),
  harness: "cwbvh-parity-oracle",
  verdict: mismatches.length === 0 ? "PASS" : "FAIL",
  command: "npm run behavioral-gate:cwbvh -- --write-status",
  rayCount: rays.length,
  cwbvhNodeCount: built.cwbvhNodeCount,
  triangleCount: Math.floor(scene.indices.length / 4),
  checks: {
    closestNoSkip: true,
    closestSkipGlass: true,
    anyNoSkip: true,
    anySkipGlass: true,
  },
  mismatches,
};

if (WRITE_STATUS) {
  await Deno.writeTextFile(STATUS_PATH, `${JSON.stringify(status, null, 2)}\n`);
}

if (mismatches.length > 0) {
  console.error("[cwbvh-parity] FAIL");
  for (const mismatch of mismatches) console.error(`  ${mismatch}`);
  Deno.exit(1);
}

console.log("[cwbvh-parity] PASS");
console.log(`  rays          : ${rays.length}`);
console.log(`  CWBVH nodes   : ${built.cwbvhNodeCount}`);
console.log(`  triangles     : ${Math.floor(scene.indices.length / 4)}`);
if (WRITE_STATUS) {
  console.log(`  status        : ${STATUS_PATH.pathname}`);
}
