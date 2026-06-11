// nrcQueryHarness.ts — lavapipe validation of the INLINE single-sample NRC MLP
// forward (nrcQuery.wgsl.ts `nrcMlpForward`) against a CPU oracle.
//
// This isolates the load-bearing NEW code added for the live NRC query — the
// per-thread dense MLP forward that runs inside the gi-ris shader — and proves
// it matches an exact CPU forward for the SAME concatenated weight/bias layout
// the FusedMlpTrainer writes (so the network the trainer learns is the network
// the query evaluates). It also exercises the inline hash-grid + one-blob encode
// path so the full nrcAssembleInput → nrcMlpForward chain compiles + runs.
//
// Run (lavapipe / CPU — correctness, not perf):
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
//     ~/.deno/bin/deno run --unstable-webgpu -A \
//     packages/walkaround-hybrid/src/neural/nrc/nrcQueryHarness.ts
//
// Self-contained: it composes ONLY the NRC encoding + query helpers (no scene
// traversal / shared-package WGSL), wrapped in a tiny test kernel that writes
// nrcMlpForward(input) for a batch of queries to an output buffer.

import { nrcEncodeHelpersWgsl } from './wgsl/nrcEncoding.wgsl.ts';
import { nrcQueryWgsl, nrcQueryLayerPlan, type NrcQueryWgslOptions } from './wgsl/nrcQuery.wgsl.ts';
import { nrcInputWidth, type NrcEncodingConfig, levelResolution } from './nrcEncoding.ts';

declare const Deno: { exit: (c?: number) => never };

const CFG: NrcQueryWgslOptions = {
  levels: 4, featuresPerEntry: 2, oneBlobBins: 6, width: 16, outWidth: 3, hidden: 2,
};

// Build the encoding config (matching the WGSL sizes) so we can compute inW + run
// the CPU encode for the oracle.
function encCfg(aabbMin: [number, number, number], aabbMax: [number, number, number]): NrcEncodingConfig {
  const levels = [];
  for (let l = 0; l < CFG.levels; l++) {
    levels.push({
      resolution: levelResolution(4, 2.0, l),
      tableSize: 64,
      table: new Float32Array(64 * CFG.featuresPerEntry),
    });
  }
  return { hashGrid: { dim: 3, featuresPerEntry: CFG.featuresPerEntry, levels, aabbMin, aabbMax }, oneBlob: { bins: CFG.oneBlobBins, sigma: 1 / CFG.oneBlobBins } };
}

// CPU dense MLP forward for the trainer's concatenated layout (ReLU hidden,
// linear out). Mirrors fusedMlpHarness.cpuGrads' forward sweep.
function cpuForward(w: Float32Array, b: Float32Array, x: Float32Array, plan: ReturnType<typeof nrcQueryLayerPlan>, W: number, outW: number): number[] {
  const wl = plan.wlayers;
  let a: number[] = new Array<number>(W).fill(0);
  for (let i = 0; i < W; i++) a[i] = (i < x.length) ? x[i]! : 0;
  for (let l = 0; l < wl; l++) {
    const iN = plan.inW[l]!, oN = plan.outW[l]!, isOut = l === wl - 1;
    const na: number[] = new Array<number>(W).fill(0);
    for (let o = 0; o < oN; o++) {
      let acc = b[plan.bOff[l]! + o]!;
      for (let i = 0; i < iN; i++) acc += w[plan.wOff[l]! + o * iN + i]! * a[i]!;
      na[o] = isOut ? acc : Math.max(0, acc);
    }
    a = na;
  }
  return [a[0]!, a[1]!, a[2]!].map((v) => Math.max(0, v));
}

async function main() {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) { console.log(JSON.stringify({ ok: false, reason: 'no navigator.gpu' })); Deno.exit(1); }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { console.log(JSON.stringify({ ok: false, reason: 'no adapter' })); Deno.exit(1); }
  // The test kernel binds 9 storage buffers (the query module's 6 incl. the H27
  // nrcSlotClaims + the 3 harness-only qins/qout/qfeat) — above the WebGPU
  // default maxStorageBuffersPerShaderStage of 8, so request headroom (same
  // clamp-to-adapter discipline as fusedMlpHarness).
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage:
        Math.min(adapter.limits.maxStorageBuffersPerShaderStage ?? 10, 10),
    },
  });
  device.addEventListener?.('uncapturederror', (ev: unknown) => {
    const e = ev as { error?: { message?: string } };
    console.log('WGPU UNCAPTURED ERROR:', e?.error?.message ?? ev);
  });

  const aabbMin: [number, number, number] = [-5, -5, -5];
  const aabbMax: [number, number, number] = [5, 5, 5];
  const enc = encCfg(aabbMin, aabbMax);
  const inW = nrcInputWidth(enc);
  const plan = nrcQueryLayerPlan(CFG);

  // Random weights/biases for the trainer's layout.
  let s = 4242 >>> 0;
  const rng = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };
  const w = new Float32Array(plan.totalW);
  const b = new Float32Array(plan.totalB);
  for (let l = 0; l < plan.wlayers; l++) {
    const scale = Math.sqrt(2 / plan.inW[l]!);
    for (let k = 0; k < plan.inW[l]! * plan.outW[l]!; k++) w[plan.wOff[l]! + k] = (rng() * 2 - 1) * scale;
  }
  for (let i = 0; i < b.length; i++) b[i] = (rng() * 2 - 1) * 0.2;

  // Hash tables (zero — so the hash-grid features are 0; the encode still runs,
  // and the oracle uses the SAME zero tables, so the forward match is exact and
  // independent of the table contents).
  let totalRows = 0;
  const levelDescs = new Uint32Array(CFG.levels * 4);
  for (let l = 0; l < CFG.levels; l++) {
    levelDescs[l * 4 + 0] = levelResolution(4, 2.0, l) >>> 0;
    levelDescs[l * 4 + 1] = 64;
    levelDescs[l * 4 + 2] = (totalRows * CFG.featuresPerEntry) >>> 0;
    totalRows += 64;
  }
  const tables = new Float32Array(totalRows * CFG.featuresPerEntry);

  // A small batch of query (pos, normal, dir, rough, albedo).
  const N = 8;
  const queries: { pos: [number, number, number]; n: [number, number, number]; d: [number, number, number]; rough: number; alb: [number, number, number] }[] = [];
  for (let q = 0; q < N; q++) {
    const norm = (v: [number, number, number]) => { const m = Math.hypot(...v) || 1; return [v[0] / m, v[1] / m, v[2] / m] as [number, number, number]; };
    queries.push({
      pos: [rng() * 8 - 4, rng() * 8 - 4, rng() * 8 - 4],
      n: norm([rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]),
      d: norm([rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1]),
      rough: rng(),
      alb: [rng(), rng(), rng()],
    });
  }

  // Test kernel: one invocation per query → nrcQueryRadiance into out[q].
  // The query module declares its NRC buffers on @group(4) by default (the real
  // gi-ris pipeline binds them as the 5th group with the full-tier
  // maxBindGroups). The isolated harness only validates the FORWARD MATH, so we
  // emit the bindings on @group(0) via the builder's `group` option to fit
  // lavapipe's default maxBindGroups=4 — the binding NUMBERS (0..5) are
  // preserved, so the bind group below maps 1:1.
  const queryWgsl = nrcEncodeHelpersWgsl() + nrcQueryWgsl({ ...CFG, group: 0 });
  // Stub the spread-termination + reservoir symbols the query module does NOT
  // need (it only needs the encode/forward). nrcQuery references nrcCfg fields
  // recordCap/recordStride/spreadC/aabb — declared in nrcQuery's own NrcCfgUBO.
  // Harness-only buffers live at bindings 7..9: the query module itself owns
  // bindings 0..6 on the remapped group — including nrcSlotClaims at binding 6
  // (H27), which nrcWriteRecord below exercises. (The harness previously placed
  // qins at binding 6 and never bound the claims buffer — a naga "bindings
  // conflict" compile error since H27 landed; fixed here.)
  const kernel = /* wgsl */`
${queryWgsl}
struct QIn { pos: vec3f, rough: f32, n: vec3f, _p0: f32, d: vec3f, _p1: f32, alb: vec3f, _p2: f32 }
@group(0) @binding(7) var<storage, read> qins : array<QIn>;
@group(0) @binding(8) var<storage, read_write> qout : array<vec4f>;
@group(0) @binding(9) var<storage, read_write> qfeat : array<f32>;
@compute @workgroup_size(1,1,1)
fn queryMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let q = qins[i];
  qout[i] = vec4f(nrcQueryRadiance(q.pos, q.n, q.d, q.rough, q.alb), 0.0);
  // Touch nrcRecords so the layout:'auto' bind group keeps binding 4 (otherwise
  // the unused record buffer is dropped from the derived layout → 8 vs 9 bind
  // mismatch). The real gi-ris pipeline uses an explicit 6-binding layout.
  nrcWriteRecord(i, q.pos, q.n, q.d, q.rough, q.alb, qout[i].xyz);
  // Debug: dump the assembled encode for query 0 (bisects encode vs forward).
  if (i == 0u) {
    var feat: array<f32, NRC_IN_W>;
    nrcAssembleInput(q.pos, q.n, q.d, q.rough, q.alb, &feat);
    for (var k: u32 = 0u; k < NRC_IN_W; k = k + 1u) { qfeat[k] = feat[k]; }
  }
}`;

  const mod = device.createShaderModule({ code: kernel });
  const ci = await mod.getCompilationInfo();
  const errs = ci.messages.filter((m) => m.type === 'error');
  if (errs.length) {
    console.log('SHADER COMPILE ERRORS:');
    for (const e of errs) console.log(`  ${e.lineNum}:${e.linePos} ${e.message}`);
    Deno.exit(1);
  }
  console.log('NRC query kernel COMPILED on lavapipe ✓');

  const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mk = (data: Float32Array | Uint32Array, usage = ST) => {
    const buf = device.createBuffer({ size: Math.max(16, (data.byteLength + 3) & ~3), usage });
    device.queue.writeBuffer(buf, 0, data as unknown as GPUAllowSharedBufferSource);
    return buf;
  };
  const wBuf = mk(w), bBuf = mk(b), tBuf = mk(tables), lBuf = mk(levelDescs);
  // Production record layout: [inW input | 3 target | 3 query world pos]
  // (nrcWriteRecord writes through base + inW + 5; the old `inW + 3` stride
  // silently clamped the last record's pos tail out of bounds).
  const recStride = inW + 3 + 3;
  const recBuf = device.createBuffer({ size: Math.max(16, N * recStride * 4), usage: ST });
  // H27 per-slot claim flags (atomic u32, one per record slot; zero = unclaimed).
  const claimsBuf = device.createBuffer({ size: Math.max(16, N * 4), usage: ST });
  device.queue.writeBuffer(claimsBuf, 0, new Uint32Array(N));
  // cfg UBO
  const cfgAb = new ArrayBuffer(48);
  const cf = new Float32Array(cfgAb), cu = new Uint32Array(cfgAb);
  cf[0] = aabbMin[0]; cf[1] = aabbMin[1]; cf[2] = aabbMin[2]; cf[3] = 0.01;
  cf[4] = aabbMax[0]; cf[5] = aabbMax[1]; cf[6] = aabbMax[2];
  cu[7] = N; cu[8] = recStride;
  const cfgBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(cfgBuf, 0, cfgAb);

  // query input buffer (std140-ish vec4-aligned, stride 16 floats = 64 B).
  const qStride = 16;
  const qData = new Float32Array(N * qStride);
  for (let i = 0; i < N; i++) {
    const o = i * qStride; const q = queries[i]!;
    qData[o + 0] = q.pos[0]; qData[o + 1] = q.pos[1]; qData[o + 2] = q.pos[2]; qData[o + 3] = q.rough;
    qData[o + 4] = q.n[0]; qData[o + 5] = q.n[1]; qData[o + 6] = q.n[2];
    qData[o + 8] = q.d[0]; qData[o + 9] = q.d[1]; qData[o + 10] = q.d[2];
    qData[o + 12] = q.alb[0]; qData[o + 13] = q.alb[1]; qData[o + 14] = q.alb[2];
  }
  const qinBuf = mk(qData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qoutBuf = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const qfeatBuf = device.createBuffer({ size: Math.max(16, inW * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'queryMain' } });
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: tBuf } },
      { binding: 3, resource: { buffer: lBuf } },
      { binding: 4, resource: { buffer: recBuf } },
      { binding: 5, resource: { buffer: cfgBuf } },
      { binding: 6, resource: { buffer: claimsBuf } },
      { binding: 7, resource: { buffer: qinBuf } },
      { binding: 8, resource: { buffer: qoutBuf } },
      { binding: 9, resource: { buffer: qfeatBuf } },
    ],
  });
  const e = device.createCommandEncoder();
  const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(N); p.end();
  const rb = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const fb = device.createBuffer({ size: Math.max(16, inW * 4), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  e.copyBufferToBuffer(qoutBuf, 0, rb, 0, N * 16);
  e.copyBufferToBuffer(qfeatBuf, 0, fb, 0, Math.max(16, inW * 4));
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const gpuOut = new Float32Array(rb.getMappedRange().slice(0));
  await fb.mapAsync(GPUMapMode.READ);
  const gpuFeat = new Float32Array(fb.getMappedRange().slice(0));
  console.log('qout[0..3]:', Array.from(gpuOut.slice(0, 4)).map((v) => v.toExponential(2)).join(' '));

  // CPU oracle: reproduce nrcAssembleInput exactly then the dense forward.
  const { assembleNrcInput } = await import('./nrcEncoding.ts');
  // Encode bisect: compare query-0's assembled feature vector.
  {
    const q = queries[0]!;
    const x = assembleNrcInput(enc, { position: q.pos, normal: q.n, direction: q.d, roughness: q.rough, albedo: q.alb });
    let fErr = 0, worst = -1;
    for (let k = 0; k < inW; k++) { const er = Math.abs(gpuFeat[k]! - x[k]!); if (er > fErr) { fErr = er; worst = k; } }
    console.log(`encode GPU vs CPU: maxAbsErr=${fErr.toExponential(3)} at idx ${worst} (gpu=${gpuFeat[worst]?.toFixed(5)} cpu=${x[worst]?.toFixed(5)}, inW=${inW})`);
  }
  let maxErr = 0;
  for (let i = 0; i < N; i++) {
    const q = queries[i]!;
    const x = assembleNrcInput(enc, { position: q.pos, normal: q.n, direction: q.d, roughness: q.rough, albedo: q.alb });
    const cpu = cpuForward(w, b, x, plan, CFG.width, CFG.outWidth);
    for (let c = 0; c < 3; c++) {
      const g = gpuOut[i * 4 + c]!;
      const err = Math.abs(g - cpu[c]!);
      if (err > maxErr) maxErr = err;
    }
  }
  console.log(`inline-MLP-forward GPU vs CPU oracle: maxAbsErr = ${maxErr.toExponential(3)}`);
  const ok = maxErr < 1e-4;
  console.log('FORWARD MATCH:', ok ? 'PASS (inline forward == CPU dense forward)' : 'FAIL');
  rb.unmap();
  device.destroy?.();
  Deno.exit(ok ? 0 : 1);
}

await main();
