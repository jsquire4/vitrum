// nrcEncodeBackwardHarness.ts — lavapipe validation for WS3: the NRC hash-grid
// encode-backward (the half that makes the multiresolution feature tables LEARN).
//
// Müller 2021 (NRC) + Müller 2022 (Instant-NGP §4). Proves on a real WebGPU
// adapter (lavapipe = CPU correctness) that:
//   1. the encode-backward scatter (nrcEncodeBackward.wgsl.ts) reproduces the CPU
//      oracle (nrcEncoding.ts hashGridBackward) given the trainer's dL/dX, and
//   2. a TABLE Adam step MOVES the tables (LIVENESS — guards the silent no-write
//      / frozen-table failure mode that this workstream exists to fix).
//
// Run (lavapipe / CPU — correctness, not perf):
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
//     ~/.deno/bin/deno run --unstable-webgpu -A \
//     packages/walkaround-hybrid/src/neural/nrc/nrcEncodeBackwardHarness.ts
//
// Self-contained; NOT wired into the path tracer.

import { FusedMlpTrainer, type FusedNetSpec, ADAM_WGSL } from "./fusedMlpTrainer.ts";
import { gradFinalizeWgsl } from "./wgsl/fusedMlp.wgsl.ts";
import { nrcEncodeBackwardWgsl } from "./wgsl/nrcEncodeBackward.wgsl.ts";
import {
  hashGridForward, hashGridBackward, levelResolution,
  type HashGridConfig, type HashGridLevel,
} from "./nrcEncoding.ts";

declare const Deno: { exit: (c?: number) => never };

function makeGrid(seed = 3): HashGridConfig {
  const F = 2, nMin = 4, growth = 2;
  const levels: HashGridLevel[] = [];
  let s = seed >>> 0;
  const rng = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  for (let l = 0; l < 4; l++) {
    levels.push({
      resolution: levelResolution(nMin, growth, l), tableSize: 53,
      table: new Float32Array(53 * F).map(() => (rng() * 2 - 1) * 0.5),
    });
  }
  return { dim: 3, featuresPerEntry: F, levels, aabbMin: [-1, -1, -1], aabbMax: [1, 1, 1] };
}

async function readF32(device: GPUDevice, buf: GPUBuffer, count: number): Promise<Float32Array> {
  const bytes = Math.max(16, count * 4);
  const rb = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0)).subarray(0, count);
  const copy = new Float32Array(out);
  rb.unmap(); rb.destroy();
  return copy;
}

async function main() {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) { console.log("no navigator.gpu"); Deno.exit(1); }
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { console.log("no adapter"); Deno.exit(1); }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupStorageSize: Math.min(adapter.limits.maxComputeWorkgroupStorageSize ?? 32768, 32768),
      maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage ?? 10, 10),
    },
  });
  console.log("=== NRC ENCODE-BACKWARD (table learning) — VALIDATION ===");

  // ── Build a tiny trainer whose inW = the grid's L·F + a fixed tail (mirrors
  // the real encoded layout: hash-grid features at the FRONT). ──
  const grid = makeGrid(5);
  const F = grid.featuresPerEntry, L = grid.levels.length;
  const LF = L * F;
  const tail = [0.3, 0.2, 0.7, 0.4, 0.8, 0.1, 0.5];
  const inW = LF + tail.length;
  const B = 16;
  const spec: FusedNetSpec = { inW, W: 16, outW: 3, hidden: 3 };
  const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB: 8 });
  await trainer.build(B);

  // He-init
  const plan = trainer.layerPlan;
  const w = new Float32Array(plan.totalW), b = new Float32Array(plan.totalB);
  let sd = 4242 >>> 0; const rng = () => { sd = (Math.imul(sd, 1664525) + 1013904223) >>> 0; return sd / 0x100000000; };
  for (let l = 0; l < plan.wlayers; l++) { const sc = Math.sqrt(2 / plan.inW[l]); for (let k = 0; k < plan.inW[l] * plan.outW[l]; k++) w[plan.wOff[l] + k] = (rng() * 2 - 1) * sc; }
  b.fill(0.1);
  trainer.setWeights(w, b);

  // batch: B query positions; x = [hashGridForward(pos) | tail]; y = random target.
  const positions: [number, number, number][] = [];
  const x = new Float32Array(B * inW), y = new Float32Array(B * 3);
  for (let s = 0; s < B; s++) {
    const pos: [number, number, number] = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
    positions.push(pos);
    const hg = hashGridForward(grid, pos);
    for (let i = 0; i < LF; i++) x[s * inW + i] = hg[i]!;
    for (let i = 0; i < tail.length; i++) x[s * inW + LF + i] = tail[i]!;
    for (let o = 0; o < 3; o++) y[s * 3 + o] = rng();
  }
  trainer.setBatch(x, y);
  trainer.computeGradsStep(); // produces gradInputF (dL/dX)
  const gpuDX = await trainer.readInputGrads(); // [B × inW]

  // ── GPU encode-backward: build the table-grad scatter + finalize ──
  const tableScalars = grid.levels.reduce((a, l) => a + l.tableSize * F, 0);
  const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const gradTablesFx = device.createBuffer({ size: tableScalars * 4, usage: ST });
  const gradTablesF = device.createBuffer({ size: tableScalars * 4, usage: ST });
  const tablesBuf = device.createBuffer({ size: tableScalars * 4, usage: ST });
  const mT = device.createBuffer({ size: tableScalars * 4, usage: ST });
  const vT = device.createBuffer({ size: tableScalars * 4, usage: ST });
  const posBuf = device.createBuffer({ size: B * 3 * 4, usage: ST });
  const levelsBuf = device.createBuffer({ size: L * 16, usage: ST });

  // pack levels + tables identical to nrcSubsystem
  const levelDescs = new Uint32Array(L * 4); let off = 0;
  const tableData = new Float32Array(tableScalars);
  for (let l = 0; l < L; l++) {
    levelDescs[l * 4 + 0] = grid.levels[l]!.resolution;
    levelDescs[l * 4 + 1] = grid.levels[l]!.tableSize;
    levelDescs[l * 4 + 2] = off;
    tableData.set(grid.levels[l]!.table, off);
    off += grid.levels[l]!.tableSize * F;
  }
  device.queue.writeBuffer(levelsBuf, 0, levelDescs);
  device.queue.writeBuffer(tablesBuf, 0, tableData);
  const posFlat = new Float32Array(B * 3);
  for (let s = 0; s < B; s++) { posFlat[s * 3] = positions[s]![0]; posFlat[s * 3 + 1] = positions[s]![1]; posFlat[s * 3 + 2] = positions[s]![2]; }
  device.queue.writeBuffer(posBuf, 0, posFlat);

  const encParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ep = new ArrayBuffer(32); const epf = new Float32Array(ep); const epu = new Uint32Array(ep);
  epf[0] = grid.aabbMin[0]; epf[1] = grid.aabbMin[1]; epf[2] = grid.aabbMin[2]; epu[3] = B;
  epf[4] = grid.aabbMax[0]; epf[5] = grid.aabbMax[1]; epf[6] = grid.aabbMax[2];
  device.queue.writeBuffer(encParams, 0, ep);

  const mkPipe = async (code: string, entry: string) =>
    device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: entry } });
  const pEnc = await mkPipe(nrcEncodeBackwardWgsl({ levels: L, featuresPerEntry: F, inWidth: inW }), "nrcEncodeBackward");
  const pFin = await mkPipe(gradFinalizeWgsl(), "gradFinalize");
  const pAdam = await mkPipe(ADAM_WGSL, "adamMain");

  // clear + dispatch encode-backward
  { const e = device.createCommandEncoder(); e.clearBuffer(gradTablesFx); device.queue.submit([e.finish()]); }
  { const e = device.createCommandEncoder();
    const bg = device.createBindGroup({ layout: pEnc.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: posBuf } }, { binding: 1, resource: { buffer: trainer.gradInputF } },
      { binding: 2, resource: { buffer: levelsBuf } }, { binding: 3, resource: { buffer: gradTablesFx } },
      { binding: 4, resource: { buffer: encParams } } ] });
    const p = e.beginComputePass(); p.setPipeline(pEnc); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(B / 64)); p.end();
    // finalize
    const u = new Uint32Array(4); u[0] = tableScalars;
    const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(ub, 0, u);
    const bgF = device.createBindGroup({ layout: pFin.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: gradTablesFx } }, { binding: 1, resource: { buffer: gradTablesF } }, { binding: 2, resource: { buffer: ub } } ] });
    const p2 = e.beginComputePass(); p2.setPipeline(pFin); p2.setBindGroup(0, bgF); p2.dispatchWorkgroups(Math.ceil(tableScalars / 64)); p2.end();
    device.queue.submit([e.finish()]);
  }
  const gpuTableGrad = await readF32(device, gradTablesF, tableScalars);

  // CPU oracle: scatter dL/dfeature (first LF of dX per sample) and accumulate.
  const cpuGrad = new Float32Array(tableScalars);
  for (let s = 0; s < B; s++) {
    const dFeat = gpuDX.slice(s * inW, s * inW + LF);
    const grads = hashGridBackward(grid, positions[s]!, dFeat);
    let o2 = 0; for (let l = 0; l < L; l++) { for (let k = 0; k < grads[l]!.length; k++) cpuGrad[o2 + k] += grads[l]![k]!; o2 += grads[l]!.length; }
  }
  let maxAbs = 0, maxMag = 0;
  for (let i = 0; i < tableScalars; i++) { maxAbs = Math.max(maxAbs, Math.abs(gpuTableGrad[i]! - cpuGrad[i]!)); maxMag = Math.max(maxMag, Math.abs(cpuGrad[i]!)); }
  const scatterOK = maxAbs < 1e-4 * (maxMag + 1);
  console.log(`encode-backward scatter GPU-vs-CPU: maxAbsErr=${maxAbs.toExponential(3)} (peak |g|=${maxMag.toExponential(3)}) → ${scatterOK ? "PASS" : "FAIL"}`);

  // ── LIVENESS: run 8 table Adam steps; tables must move > 1e-6 ──
  const before = await readF32(device, tablesBuf, tableScalars);
  for (let step = 1; step <= 8; step++) {
    // recompute grads each step (positions/targets fixed → grad steady)
    trainer.setBatch(x, y); trainer.computeGradsStep();
    { const e = device.createCommandEncoder(); e.clearBuffer(gradTablesFx); device.queue.submit([e.finish()]); }
    const e = device.createCommandEncoder();
    const bg = device.createBindGroup({ layout: pEnc.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: posBuf } }, { binding: 1, resource: { buffer: trainer.gradInputF } },
      { binding: 2, resource: { buffer: levelsBuf } }, { binding: 3, resource: { buffer: gradTablesFx } },
      { binding: 4, resource: { buffer: encParams } } ] });
    const p = e.beginComputePass(); p.setPipeline(pEnc); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(B / 64)); p.end();
    const u = new Uint32Array(4); u[0] = tableScalars;
    const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(ub, 0, u);
    const bgF = device.createBindGroup({ layout: pFin.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: gradTablesFx } }, { binding: 1, resource: { buffer: gradTablesF } }, { binding: 2, resource: { buffer: ub } } ] });
    const p2 = e.beginComputePass(); p2.setPipeline(pFin); p2.setBindGroup(0, bgF); p2.dispatchWorkgroups(Math.ceil(tableScalars / 64)); p2.end();
    const ab = new ArrayBuffer(48); new Uint32Array(ab, 0, 1)[0] = tableScalars; const af = new Float32Array(ab);
    af[4] = 0.1; af[5] = 0.9; af[6] = 0.999; af[7] = 1e-8; af[8] = 1 - Math.pow(0.9, step); af[9] = 1 - Math.pow(0.999, step);
    const aub = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(aub, 0, ab);
    const bgA = device.createBindGroup({ layout: pAdam.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: tablesBuf } }, { binding: 1, resource: { buffer: gradTablesF } },
      { binding: 2, resource: { buffer: mT } }, { binding: 3, resource: { buffer: vT } }, { binding: 4, resource: { buffer: aub } } ] });
    const p3 = e.beginComputePass(); p3.setPipeline(pAdam); p3.setBindGroup(0, bgA); p3.dispatchWorkgroups(Math.ceil(tableScalars / 64)); p3.end();
    device.queue.submit([e.finish()]);
  }
  const after = await readF32(device, tablesBuf, tableScalars);
  let maxDelta = 0; for (let i = 0; i < tableScalars; i++) maxDelta = Math.max(maxDelta, Math.abs(after[i]! - before[i]!));
  const live = maxDelta > 1e-6;
  console.log(`LIVENESS (8 table Adam steps): maxTableDelta=${maxDelta.toExponential(3)} → ${live ? "PASS (tables LEARN)" : "FAIL (FROZEN)"}`);

  console.log(JSON.stringify({ scatterOK, live }));
  device.destroy?.();
  Deno.exit(scatterOK && live ? 0 : 1);
}

await main();
