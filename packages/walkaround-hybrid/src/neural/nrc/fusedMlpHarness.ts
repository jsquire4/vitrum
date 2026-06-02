// fusedMlpHarness.ts — lavapipe validation harness for the FUSED NRC MLP kernel.
//
// Mirrors the discipline of the spike harness (tools/nrc-spike/mlpTrain.ts):
//   PART 1: finite-difference gradient check on a tiny net (analytic ≈ FD).
//   PART 2: fit a known radiance-like function (proves the fused loop learns).
//   PART 3: global-memory-traffic model — bytes/step fused vs the spike's
//           per-layer dispatch (the perf-engineering win on real hardware).
//
// Run (lavapipe / CPU — correctness, not wall-clock perf):
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
//     ~/.deno/bin/deno run --unstable-webgpu -A \
//     packages/walkaround-hybrid/src/neural/nrc/fusedMlpHarness.ts [--f16]
//
// NOT wired into the path tracer. Self-contained validation only.

import { FusedMlpTrainer, heInit, type FusedNetSpec } from "./fusedMlpTrainer.ts";
import { FusedMlpTrainerProbe } from "./fusedMlpTrainerProbe.ts";

declare const Deno: { args: string[]; exit: (c?: number) => never };

const wantF16 = Deno.args.includes("--f16");

// ── target function the MLP must learn (a small "radiance"-like field) ──
function targetFn(x0: number, x1: number): number {
  return 0.5 + 0.5 * Math.sin(3.0 * x0) * Math.cos(2.5 * x1) * Math.exp(-0.3 * (x0 * x0 + x1 * x1));
}

function makeBatch(B: number, inW: number, seed: { s: number }, outW: number) {
  const x = new Float32Array(B * inW);
  const y = new Float32Array(B * outW);
  const rng = () => { seed.s = (Math.imul(seed.s, 1664525) + 1013904223) >>> 0; return seed.s / 0x100000000; };
  for (let b = 0; b < B; b++) {
    const a = rng() * 2 - 1, c = rng() * 2 - 1;
    x[b * inW + 0] = a; x[b * inW + 1] = c;
    for (let i = 2; i < inW; i++) x[b * inW + i] = Math.sin(i * a + c); // encoding-like pad
    const t = targetFn(a, c);
    for (let o = 0; o < outW; o++) y[b * outW + o] = t * (1 + 0.1 * o); // distinct RGB-ish channels
  }
  return { x, y };
}

// CPU reference: exact analytic forward+backward for the SAME net the GPU runs.
// This is the ground-truth oracle (the GPU kernel must match THIS); FD is a
// secondary sanity check that carries ReLU-kink + step-size error. Layout:
// node-layers padded to W (matching the GPU saveOff scheme).
function cpuGrads(
  w: Float32Array, b: Float32Array, x: Float32Array, y: Float32Array,
  spec: FusedNetSpec, plan: { wOff: number[]; bOff: number[]; inW: number[]; outW: number[]; wlayers: number; totalW: number; totalB: number },
  B: number,
): { gw: Float32Array; gb: Float32Array } {
  const { W, outW, inW: rawInW } = spec;
  const wl = plan.wlayers;
  const node = spec.hidden + 2;
  const gw = new Float32Array(plan.totalW);
  const gb = new Float32Array(plan.totalB);
  for (let S = 0; S < B; S++) {
    // forward; a[nl][n], z[nl][n] padded to W
    const a: number[][] = [], z: number[][] = [];
    for (let nl = 0; nl < node; nl++) { a.push(new Array(W).fill(0)); z.push(new Array(W).fill(0)); }
    for (let i = 0; i < W; i++) a[0][i] = (i < rawInW) ? x[S * rawInW + i] : 0;
    for (let l = 0; l < wl; l++) {
      const iN = plan.inW[l], oN = plan.outW[l], isOut = l === wl - 1;
      for (let o = 0; o < oN; o++) {
        let acc = b[plan.bOff[l] + o];
        for (let i = 0; i < iN; i++) acc += w[plan.wOff[l] + o * iN + i] * a[l][i];
        z[l + 1][o] = acc;
        a[l + 1][o] = isOut ? acc : Math.max(0, acc);
      }
    }
    // backward
    const delta: number[][] = [];
    for (let nl = 0; nl < node; nl++) delta.push(new Array(W).fill(0));
    for (let o = 0; o < outW; o++) delta[node - 1][o] = (a[node - 1][o] - y[S * outW + o]) / B;
    for (let l = wl - 1; l >= 0; l--) {
      const iN = plan.inW[l], oN = plan.outW[l];
      for (let o = 0; o < oN; o++) {
        gb[plan.bOff[l] + o] += delta[l + 1][o];
        for (let i = 0; i < iN; i++) gw[plan.wOff[l] + o * iN + i] += delta[l + 1][o] * a[l][i];
      }
      if (l > 0) {
        for (let i = 0; i < iN; i++) {
          let acc = 0;
          for (let o = 0; o < oN; o++) acc += w[plan.wOff[l] + o * iN + i] * delta[l + 1][o];
          delta[l][i] = acc * (z[l][i] > 0 ? 1 : 0);
        }
      }
    }
  }
  return { gw, gb };
}

// CPU reference for dL/dX (the gradient w.r.t. the RAW network input). The input
// layer is LINEAR, so dL/dX[S,i] = Σ_o W[0][o,i]·δ₁[S,o] with NO relu' factor —
// this is the upstream signal the NRC hash-grid encode-backward scatters into the
// trainable feature tables (Müller 2022 Instant-NGP §4). Returns [B × rawInW].
function cpuInputGrads(
  w: Float32Array, b: Float32Array, x: Float32Array, y: Float32Array,
  spec: FusedNetSpec, plan: { wOff: number[]; bOff: number[]; inW: number[]; outW: number[]; wlayers: number; totalW: number; totalB: number },
  B: number,
): Float32Array {
  const { W, outW, inW: rawInW } = spec;
  const wl = plan.wlayers;
  const node = spec.hidden + 2;
  const dXall = new Float32Array(B * rawInW);
  for (let S = 0; S < B; S++) {
    const a: number[][] = [], z: number[][] = [];
    for (let nl = 0; nl < node; nl++) { a.push(new Array(W).fill(0)); z.push(new Array(W).fill(0)); }
    for (let i = 0; i < W; i++) a[0][i] = (i < rawInW) ? x[S * rawInW + i] : 0;
    for (let l = 0; l < wl; l++) {
      const iN = plan.inW[l], oN = plan.outW[l], isOut = l === wl - 1;
      for (let o = 0; o < oN; o++) {
        let acc = b[plan.bOff[l] + o];
        for (let i = 0; i < iN; i++) acc += w[plan.wOff[l] + o * iN + i] * a[l][i];
        z[l + 1][o] = acc;
        a[l + 1][o] = isOut ? acc : Math.max(0, acc);
      }
    }
    const delta: number[][] = [];
    for (let nl = 0; nl < node; nl++) delta.push(new Array(W).fill(0));
    for (let o = 0; o < outW; o++) delta[node - 1][o] = (a[node - 1][o] - y[S * outW + o]) / B;
    for (let l = wl - 1; l >= 1; l--) {
      const iN = plan.inW[l], oN = plan.outW[l];
      for (let i = 0; i < iN; i++) {
        let acc = 0;
        for (let o = 0; o < oN; o++) acc += w[plan.wOff[l] + o * iN + i] * delta[l + 1][o];
        delta[l][i] = acc * (z[l][i] > 0 ? 1 : 0);
      }
    }
    // l==0: LINEAR input → dL/dX[S,i] = Σ_o W[0][o,i]·δ₁[o].
    const iN = plan.inW[0], oN = plan.outW[0];
    for (let i = 0; i < Math.min(iN, rawInW); i++) {
      let acc = 0;
      for (let o = 0; o < oN; o++) acc += w[plan.wOff[0] + o * iN + i] * delta[1][o];
      dXall[S * rawInW + i] = acc;
    }
  }
  return dXall;
}

function relErr(a: Float32Array, b: Float32Array) {
  let maxRel = 0, maxAbs = 0;
  let worstRelIdx = -1, worstAbsIdx = -1;
  // Relative error is only meaningful where the gradient magnitude is
  // non-negligible; for near-zero cells FD round-off dominates. We gate the
  // relErr denominator with a floor and ALSO track the worst meaningful cell
  // (one whose magnitude is above a small fraction of the max gradient).
  let maxMag = 0;
  for (let i = 0; i < a.length; i++) maxMag = Math.max(maxMag, Math.abs(a[i]), Math.abs(b[i]));
  const floor = Math.max(1e-4, maxMag * 1e-3); // ignore cells below 0.1% of peak
  let maxRelMeaningful = 0;
  for (let i = 0; i < a.length; i++) {
    const abs = Math.abs(a[i] - b[i]);
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i])) + 1e-6;
    const rel = abs / denom;
    if (abs > maxAbs) { maxAbs = abs; worstAbsIdx = i; }
    if (rel > maxRel) { maxRel = rel; worstRelIdx = i; }
    if (denom > floor && rel > maxRelMeaningful) maxRelMeaningful = rel;
  }
  return { maxRel, maxAbs, maxRelMeaningful, worstRelIdx, worstAbsIdx, maxMag, floor };
}

async function main() {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) { console.log(JSON.stringify({ ok: false, reason: "no navigator.gpu" })); Deno.exit(1); }
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { console.log(JSON.stringify({ ok: false, reason: "no adapter" })); Deno.exit(1); }
  const info = (adapter as unknown as { info?: Record<string, string> }).info ?? {};
  const hasF16 = adapter.features.has("shader-f16");
  const useF16 = wantF16 && hasF16;

  const required: GPUFeatureName[] = useF16 ? ["shader-f16"] : [];
  // Request the workgroup-shared-memory + storage-buffer headroom the fused
  // kernel needs (lavapipe exposes far more than the WebGPU defaults of
  // 16384 B shared / 8 storage buffers). On a real adapter the harness should
  // clamp these to adapter.limits; here lavapipe grants them outright.
  const want = (k: string, fallback: number) =>
    Math.min((adapter.limits as unknown as Record<string, number>)[k] ?? fallback, fallback);
  const requiredLimits: Record<string, number> = {
    maxComputeWorkgroupStorageSize: want("maxComputeWorkgroupStorageSize", 32768),
    maxStorageBuffersPerShaderStage: want("maxStorageBuffersPerShaderStage", 10),
  };
  const device = await adapter.requestDevice({ requiredFeatures: required, requiredLimits });
  device.addEventListener?.("uncapturederror", (e: unknown) => {
    const ev = e as { error?: { message?: string } };
    console.error("WGPU ERROR:", ev?.error?.message ?? e);
  });

  console.log("=== NRC FUSED/TILED MLP KERNEL — VALIDATION ===");
  console.log("adapter:", info.description ?? info.vendor ?? "(unknown)");
  console.log("shader-f16 supported:", hasF16, "| requested:", wantF16, "| ACTIVE precision:", useF16 ? "f16 (mixed)" : "f32");
  console.log("maxComputeWorkgroupStorageSize:", device.limits.maxComputeWorkgroupStorageSize, "bytes");

  // ── PART 1: finite-difference gradient check on a TINY fused net ──
  // Tiny so the FD sweep (one extra forward per param) is cheap. We still use
  // the FULL fused forward+backward kernels — same code path as the big net.
  console.log("\n--- PART 1: finite-difference gradient check (FUSED kernel) ---");
  // tiny net: W=8 hidden, 2 hidden layers, outW=3, inW=4. TILE_B small.
  const tinySpec: FusedNetSpec = { inW: 4, W: 8, outW: 3, hidden: 2 };
  // tileB must satisfy 3*tileB*W*scBytes <= shared limit; W=8 is tiny so 8 is safe.
  const tinyTileB = 8;
  const B_fd = 5;
  const tiny = new FusedMlpTrainer(device, tinySpec, { useF16, tileB: tinyTileB });
  await tiny.build(B_fd);
  const tinyProbe = new FusedMlpTrainerProbe(tiny);

  const init = heInit(tiny);
  tiny.setWeights(init.w, init.b);
  const seed = { s: 777 };
  const batch = makeBatch(B_fd, tinySpec.inW, seed, tinySpec.outW);
  tiny.setBatch(batch.x, batch.y);

  tinyProbe.computeGradsStep();
  const { gw, gb } = await tinyProbe.readGrads();

  const plan = tiny.layerPlan;

  // PRIMARY oracle: CPU exact analytic grads for the SAME net. GPU must match.
  const cpu = cpuGrads(init.w, init.b, batch.x, batch.y, tinySpec, plan, B_fd);
  const ecw = relErr(gw, cpu.gw);
  const ecb = relErr(gb, cpu.gb);
  console.log("GPU-vs-CPU-analytic weight grad: maxAbsErr=", ecw.maxAbs.toExponential(3),
    " maxRelErr(meaningful)=", ecw.maxRelMeaningful.toExponential(3));
  console.log("GPU-vs-CPU-analytic bias   grad: maxAbsErr=", ecb.maxAbs.toExponential(3),
    " maxRelErr(meaningful)=", ecb.maxRelMeaningful.toExponential(3));
  // GPU must match the CPU analytic to tight tolerance (f16 looser).
  const cpuTol = useF16 ? 2e-2 : 1e-4;
  const cpuMatch = ecw.maxAbs < cpuTol * (ecw.maxMag + 1) && ecb.maxAbs < cpuTol * (ecb.maxMag + 1);
  console.log("GPU==CPU-ANALYTIC:", cpuMatch ? "PASS (kernel matches exact backprop)" : "FAIL");

  // ── PART 1b: dL/dX (INPUT gradient) GPU == CPU analytic (the NRC encode-
  // backward upstream signal). The input layer is linear (no ReLU kink), so this
  // is a CLEAN check — GPU must match the CPU input-grad oracle tightly. ──
  const gdx = await tinyProbe.readInputGrads(); // [B_fd × inW]
  const cpuDX = cpuInputGrads(init.w, init.b, batch.x, batch.y, tinySpec, plan, B_fd);
  const edx = relErr(gdx, cpuDX);
  console.log("GPU-vs-CPU dL/dX: maxAbsErr=", edx.maxAbs.toExponential(3),
    " maxRelErr(meaningful)=", edx.maxRelMeaningful.toExponential(3));
  const dxTol = useF16 ? 2e-2 : 1e-4;
  const dxMatch = edx.maxAbs < dxTol * (edx.maxMag + 1);
  console.log("dL/dX GPU==CPU-ANALYTIC:", dxMatch ? "PASS (encode-backward upstream signal correct)" : "FAIL");
  // Match the spike's FD step (1e-3): large enough to beat f32 round-off, small
  // enough that few cells straddle a ReLU kink. (f16 uses a slightly larger h.)
  const h = useF16 ? 5e-3 : 1e-3;
  const fdGW = new Float32Array(plan.totalW);
  for (let k = 0; k < plan.totalW; k++) {
    const wp = init.w.slice(); wp[k] += h; tiny.setWeights(wp, init.b);
    tiny.setBatch(batch.x, batch.y);
    const lp = await tinyProbe.computeLoss();
    const wm = init.w.slice(); wm[k] -= h; tiny.setWeights(wm, init.b);
    tiny.setBatch(batch.x, batch.y);
    const lm = await tinyProbe.computeLoss();
    fdGW[k] = (lp - lm) / (2 * h);
  }
  const fdGB = new Float32Array(plan.totalB);
  for (let k = 0; k < plan.totalB; k++) {
    const bp = init.b.slice(); bp[k] += h; tiny.setWeights(init.w, bp);
    tiny.setBatch(batch.x, batch.y);
    const lp = await tinyProbe.computeLoss();
    const bm = init.b.slice(); bm[k] -= h; tiny.setWeights(init.w, bm);
    tiny.setBatch(batch.x, batch.y);
    const lm = await tinyProbe.computeLoss();
    fdGB[k] = (lp - lm) / (2 * h);
  }
  tiny.setWeights(init.w, init.b);

  const ew = relErr(gw, fdGW);
  const eb = relErr(gb, fdGB);
  console.log("weight grad: maxRelErr(meaningful cells)=", ew.maxRelMeaningful.toExponential(3),
    " maxAbsErr=", ew.maxAbs.toExponential(3), " (raw maxRel=", ew.maxRel.toExponential(2),
    "at idx", ew.worstRelIdx, "where |g|=", Math.abs(gw[ew.worstRelIdx]).toExponential(2), ")");
  console.log("bias   grad: maxRelErr(meaningful cells)=", eb.maxRelMeaningful.toExponential(3),
    " maxAbsErr=", eb.maxAbs.toExponential(3));
  console.log("sample analytic gW[0..4]:", Array.from(gw.slice(0, 5)).map((v) => v.toFixed(5)));
  console.log("sample FD       gW[0..4]:", Array.from(fdGW.slice(0, 5)).map((v) => v.toFixed(5)));
  console.log(`worst-abs cell idx=${ew.worstAbsIdx}: analytic=${gw[ew.worstAbsIdx].toExponential(4)} FD=${fdGW[ew.worstAbsIdx].toExponential(4)} (peak |g|=${ew.maxMag.toExponential(3)}, floor=${ew.floor.toExponential(2)})`);
  // FD is a SECONDARY sanity check. It carries two error sources the CPU-analytic
  // oracle does not: (1) central-difference truncation/round-off on near-zero
  // cells, and (2) ReLU-KINK error — when the FD step h crosses a ReLU activation
  // boundary, the symmetric difference straddles a derivative discontinuity and
  // is simply wrong for that cell, regardless of how correct the backprop is.
  // So we report FD agreement on meaningful, non-kink cells but DO NOT gate the
  // verdict on it — the GPU==CPU-analytic match is the real correctness proof.
  console.log(`FD secondary check: weight maxRelErr(meaningful)=${ew.maxRelMeaningful.toExponential(3)} ` +
    `(FD carries ReLU-kink error on boundary cells; CPU-analytic match above is the oracle)`);
  const gradOK = cpuMatch && dxMatch;
  console.log("GRADIENT CHECK:", gradOK ? "PASS (GPU kernel == exact backprop, incl. dL/dX)" : "FAIL");

  // ── PART 2: actually LEARN the target function (full fused train loop) ──
  console.log("\n--- PART 2: fit a known function (FUSED train loop) ---");
  const trSpec: FusedNetSpec = { inW: 2, W: 64, outW: 3, hidden: 6 }; // the Müller core sizing
  // tileB: f16 fits 64 (3*64*64*2=24576<32768); f32 needs <=32 (3*32*64*4=24576).
  const trTileB = useF16 ? 64 : 32;
  const B_tr = 512;
  const trainer = new FusedMlpTrainer(device, trSpec, { useF16, tileB: trTileB });
  await trainer.build(B_tr);
  const trainerProbe = new FusedMlpTrainerProbe(trainer);
  const ti = heInit(trainer);
  trainer.setWeights(ti.w, ti.b);

  const trSeed = { s: 2024 };
  const lossesAt: Record<number, number> = {};
  const steps = 400;
  const t0 = performance.now();
  for (let step = 0; step < steps; step++) {
    const b = makeBatch(B_tr, trSpec.inW, trSeed, trSpec.outW);
    trainer.setBatch(b.x, b.y);
    trainer.trainStep(0.01);
    if (step === 0 || step === 50 || step === 100 || step === 200 || step === steps - 1) {
      const l = await trainerProbe.computeLoss();
      lossesAt[step] = l;
    }
  }
  await device.queue.onSubmittedWorkDone?.();
  const t1 = performance.now();
  console.log("loss trajectory (MSE):");
  for (const k of Object.keys(lossesAt).map(Number).sort((a, b) => a - b)) {
    console.log(`  step ${k.toString().padStart(4)}: ${lossesAt[k].toExponential(4)}`);
  }
  const first = lossesAt[0], last = lossesAt[steps - 1];
  const learned = last < first * 0.25;
  console.log("LEARNING CHECK:", learned ? `PASS (loss fell ${(first / last).toFixed(1)}x)` : "FAIL (loss did not drop)");
  console.log(`wall time ${steps} steps (lavapipe=CPU, NOT perf-representative): ${(t1 - t0).toFixed(0)} ms`);

  // ── PART 3: global-memory-traffic model — fused vs the spike's per-layer ──
  console.log("\n--- PART 3: global-memory-traffic model (fused vs per-layer spike) ---");
  trafficModel(useF16);

  console.log("\n=== VERDICT INPUTS ===");
  console.log(JSON.stringify({
    precision: useF16 ? "f16" : "f32",
    gradientCheckPassed: gradOK,
    learningCheckPassed: learned,
    adapter: info.description ?? "",
  }));

  device.destroy?.();
  Deno.exit(gradOK && learned ? 0 : 1);
}

// Model the GLOBAL-memory traffic per training step for the Müller 6×64 core,
// fused vs the spike's per-layer dispatch. The FLOP count is unchanged by
// fusion (same arithmetic); the win is bytes moved through global memory.
function trafficModel(useF16: boolean) {
  const W = 64, HIDDEN = 6, OUT_W = 3, IN_W = 64; // padded
  const node = HIDDEN + 2;          // 8 node-layers
  const wlayers = HIDDEN + 1;       // 7 weight layers
  const sc = useF16 ? 2 : 4;        // bytes per activation scalar (fused tiles)
  // param count
  const widths = [IN_W];
  for (let i = 0; i < HIDDEN; i++) widths.push(W);
  widths.push(OUT_W);
  let totalW = 0, totalB = 0;
  for (let l = 0; l < wlayers; l++) { totalW += widths[l + 1] * widths[l]; totalB += widths[l + 1]; }
  const paramBytes = (totalW + totalB) * sc;

  for (const B of [4096, 16384, 65536]) {
    // ---- SPIKE (per-layer dispatch, f32) global traffic per step ----
    // Every layer writes acts[B×outW] + z[B×outW] to global and the next layer
    // reads acts[B×inW]; backprop reads/writes delta[B×W] per layer; grad reads
    // acts + delta again. We model the dominant activation/delta tensor traffic
    // (f32 in the spike) + weights read each layer.
    // Forward: per weight-layer  write a(B*outW) + write z(B*outW) + read a_prev(B*inW)
    // Backward delta: per layer   read delta_in(B*outW) + read z(B*inW) + write delta(B*inW)
    // Grad: per layer             read delta(B*outW) + read a_prev(B*inW)
    let spikeActFloats = 0;
    for (let l = 0; l < wlayers; l++) {
      const inW = widths[l], outW = widths[l + 1];
      spikeActFloats += B * outW + B * outW + B * inW;        // forward
      if (l > 0) spikeActFloats += B * outW + B * inW + B * inW; // backward delta
      spikeActFloats += B * outW + B * inW;                  // grad reads
    }
    const paramBytesF32 = (totalW + totalB) * 4;
    const spikeBytes = spikeActFloats * 4 + paramBytesF32 * 3;  // spike is f32; weights touched ~3x

    // ---- FUSED kernel global traffic per step ----
    // Activation tiles stay in shared across layers. Global activation traffic
    // is: forward SAVES a[node×W] + z[node×W] ONCE per sample (for backward);
    // backward READS them back ONCE. No inter-layer activation round-trips.
    // = save(node*W) + save(node*W) + read(node*W [acts]) + read(node*W [z]) per sample.
    const fusedActFloats = B * (node * W + node * W + node * W + node * W);
    // Pure-fusion win: same f32 precision, just no inter-layer round-trips.
    const fusedBytesF32 = fusedActFloats * 4 + paramBytesF32 * 2;
    // Combined win: fusion + f16 activation tiles (weights downcast f16 too).
    const fusedBytesF16 = fusedActFloats * 2 + paramBytes * 2; // paramBytes uses sc

    const fuseOnly = spikeBytes / fusedBytesF32;
    const combined = spikeBytes / fusedBytesF16;
    console.log(
      `  batch ${B.toString().padStart(6)}: spike(f32)=${(spikeBytes / 1e6).toFixed(1)} MB ` +
      `| fused-f32=${(fusedBytesF32 / 1e6).toFixed(1)} MB (${fuseOnly.toFixed(2)}x) ` +
      `| fused-f16=${(fusedBytesF16 / 1e6).toFixed(1)} MB (${combined.toFixed(2)}x)`);
  }
  console.log("  FLOP count is identical fused-vs-per-layer — the win is bytes through global memory.");
  console.log("  Attribution: ~1.5x from FUSION alone (no inter-LAYER activation round-trips);");
  console.log("  ~2x further from f16 tiles -> ~3x combined global-traffic cut (this version).");
  console.log("  Resident shared-mem/workgroup: 3 tiles × TILE_B × W ×", sc, "B =",
    3 * (useF16 ? 64 : 32) * 64 * sc, "B (≤ 32768 limit).");
  // Honest ceiling: this first version still SAVES a+z to global for backward,
  // which dominates the residual traffic. The full tiny-cuda-nn recipe avoids
  // even that by recomputing the forward inside the backward dispatch (keeping
  // only the input tile in shared). Modeling that:
  {
    const B = 65536;
    const paramBytesF32 = (totalW + totalB) * 4;
    // recompute-backward fused: forward saves nothing extra; backward re-reads
    // only inputs (B*IN_W) + writes final output (B*OUT_W). Activations live in
    // shared for both forward and the recomputed-forward-in-backward.
    const recomputeFloats = B * (IN_W /*read input fwd*/ + OUT_W /*write out*/ +
      IN_W /*read input bwd*/ + OUT_W /*read tgt*/);
    const recomputeF16 = recomputeFloats * 2 + paramBytesF32 * 2;
    const spikeRef = 825.5e6;
    console.log(`  CEILING (recompute-in-backward, f16, B=${B}): ~${(recomputeF16 / 1e6).toFixed(1)} MB ` +
      `(~${(spikeRef / recomputeF16).toFixed(0)}x vs spike) — the next-step target.`);
  }
  console.log("  Real frame perf requires the WSL-GPU/hardware adapter; lavapipe is CPU correctness only.");
}

await main();
