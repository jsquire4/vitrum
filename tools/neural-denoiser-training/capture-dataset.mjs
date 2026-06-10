#!/usr/bin/env node
/**
 * capture-dataset.mjs — deterministic noisy/clean training-pair capture for the
 * vitrum neural denoiser PIPELINE smoke.
 *
 * SCOPE: this produces a *tiny, format-correct* dataset to exercise the
 * capture → train → export → load path end-to-end. It is NOT a real training
 * dataset. The radiance source here is a CPU brute-force path tracer
 * (a self-contained port of packages/pt-webgpu/src/__tests__/cpuTracer.ts's
 * `integratePath`), used because the GPU pt-webgpu worker (lavapipe via
 * ~/projects/wsl-gpu/capture-worker/render-pt-webgpu.ts) cannot be driven
 * cleanly headless from every box. For a REAL dataset see "GPU capture" in
 * tools/neural-denoiser-training/README.md.
 *
 * Output layout matches dataset_spec.md exactly:
 *   <out>/<scene>/noisy/frame_NNNN.png          (low-spp colour, Reinhard LDR)
 *   <out>/<scene>/noisy/frame_NNNN_albedo.png   (primary-hit albedo)
 *   <out>/<scene>/noisy/frame_NNNN_normal.png   (world normal, n*0.5+0.5)
 *   <out>/<scene>/clean/frame_NNNN.png          (high-spp colour, Reinhard LDR)
 *
 * Determinism: a single Park-Miller LCG seeded per (scene, frame, spp-pass)
 * from the CLI --seed. Re-running with the same flags is bit-identical.
 *
 * Usage:
 *   node capture-dataset.mjs --out data_smoke --pairs 4 --size 128 \
 *     --noisy-spp 1 --clean-spp 256 --seed 1984
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';

// ── CLI ────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const args = {
    out: 'data_smoke',
    pairs: 4,
    size: 128,
    noisySpp: 1,
    cleanSpp: 256,
    maxBounces: 4,
    seed: 1984,
    scene: 'cornell_box',
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--out') args.out = next();
    else if (a === '--pairs') args.pairs = parseInt(next(), 10);
    else if (a === '--size') args.size = parseInt(next(), 10);
    else if (a === '--noisy-spp') args.noisySpp = parseInt(next(), 10);
    else if (a === '--clean-spp') args.cleanSpp = parseInt(next(), 10);
    else if (a === '--max-bounces') args.maxBounces = parseInt(next(), 10);
    else if (a === '--seed') args.seed = parseInt(next(), 10);
    else if (a === '--scene') args.scene = next();
    else if (a === '--help' || a === '-h') { printHelp(); process.exit(0); }
    else { console.error(`Unknown arg: ${a}`); process.exit(2); }
  }
  return args;
}
function printHelp() {
  console.log('capture-dataset.mjs — see header comment for full docs.');
  console.log('  --out <dir>        output root          (default data_smoke)');
  console.log('  --pairs <n>        noisy/clean pairs    (default 4)');
  console.log('  --size <px>        square resolution    (default 128)');
  console.log('  --noisy-spp <n>    samples/pixel noisy  (default 1)');
  console.log('  --clean-spp <n>    samples/pixel clean  (default 256)');
  console.log('  --seed <n>         base RNG seed        (default 1984)');
}

// ── Deterministic LCG (Park-Miller, mirrors cpuTracer.ts lcg) ────────────────
function makeRng(seed) { return { v: (seed >>> 0) || 1 }; }
function lcg(rng) {
  rng.v = (Math.imul(rng.v, 1664525) + 1013904223) >>> 0;
  return rng.v / 0x100000000;
}

// ── Vec3 ─────────────────────────────────────────────────────────────────────
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, b) => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const len3 = (v) => Math.sqrt(dot(v, v));
function normalize(v) { const l = len3(v); return l < 1e-8 ? [0, 1, 0] : scale(v, 1 / l); }
const maxComp = (v) => Math.max(v[0], Math.max(v[1], v[2]));

// ── Möller-Trumbore (mirror of cpuTracer.ts intersectTriangleMT) ─────────────
function intersectTri(o, d, v0, v1, v2, eps = 1e-5) {
  const e1 = sub(v1, v0), e2 = sub(v2, v0);
  const n = cross(e1, e2);
  const det = -dot(d, n);
  if (Math.abs(det) < eps) return null;
  const invDet = 1 / det;
  const AO = sub(o, v0);
  const DAO = cross(AO, d);
  const u = dot(e2, DAO) * invDet;
  const v = -dot(e1, DAO) * invDet;
  const t = dot(AO, n) * invDet;
  const w = 1 - u - v;
  if (u < -eps || v < -eps || w < -eps || t < eps) return null;
  return t;
}

// ── Cosine-hemisphere sample (mirror) ────────────────────────────────────────
function buildOnb(n) {
  const lensq = n[0] * n[0] + n[1] * n[1];
  let t;
  if (lensq > 1e-10) { const inv = 1 / Math.sqrt(lensq); t = [-n[1] * inv, n[0] * inv, 0]; }
  else t = [1, 0, 0];
  return { t, b: cross(n, t) };
}
function cosineSample(rng, n) {
  const u1 = lcg(rng), u2 = lcg(rng);
  const r = Math.sqrt(u1), phi = 2 * Math.PI * u2;
  const lx = r * Math.cos(phi), ly = r * Math.sin(phi), lz = Math.sqrt(Math.max(0, 1 - u1));
  const { t, b } = buildOnb(n);
  return normalize([
    lx * t[0] + ly * b[0] + lz * n[0],
    lx * t[1] + ly * b[1] + lz * n[1],
    lx * t[2] + ly * b[2] + lz * n[2],
  ]);
}

// ── Scene: a tiny Cornell-ish box (≤ 32 tris), Lambertian only ───────────────
// Box spans [-1,1]^3, opening toward -z (camera). Ceiling emitter.
function buildScene() {
  const materials = [
    { albedo: [0.73, 0.73, 0.73], emission: [0, 0, 0] }, // 0 white
    { albedo: [0.65, 0.05, 0.05], emission: [0, 0, 0] }, // 1 red (left)
    { albedo: [0.12, 0.45, 0.15], emission: [0, 0, 0] }, // 2 green (right)
    { albedo: [0.0, 0.0, 0.0],    emission: [12, 12, 12] }, // 3 light
  ];
  const tris = [];
  const quad = (a, b, c, d, m) => { tris.push({ v: [a, b, c], m }); tris.push({ v: [a, c, d], m }); };
  const L = 1;
  // floor (white), ceiling (white), back (white)
  quad([-L, -L, -L], [L, -L, -L], [L, -L, L], [-L, -L, L], 0);   // floor
  quad([-L, L, L], [L, L, L], [L, L, -L], [-L, L, -L], 0);       // ceiling
  quad([-L, -L, -L], [-L, L, -L], [L, L, -L], [L, -L, -L], 0);   // back wall
  // left red, right green
  quad([-L, -L, L], [-L, L, L], [-L, L, -L], [-L, -L, -L], 1);   // left
  quad([L, -L, -L], [L, L, -L], [L, L, L], [L, -L, L], 2);       // right
  // ceiling emitter (small inset quad just below ceiling)
  const e = 0.4, y = L - 0.02;
  quad([-e, y, -e], [e, y, -e], [e, y, e], [-e, y, e], 3);
  // a diffuse box (two stacked quads as a crude occluder) in the middle
  quad([-0.3, -L, -0.2], [0.3, -L, -0.2], [0.3, -0.2, -0.2], [-0.3, -0.2, -0.2], 0);
  return { tris, materials, env: () => [0, 0, 0] };
}

function traceClosest(scene, o, d, tMin, tMax) {
  let best = null, bestT = tMax;
  for (let i = 0; i < scene.tris.length; i++) {
    const tr = scene.tris[i];
    const t = intersectTri(o, d, tr.v[0], tr.v[1], tr.v[2]);
    if (t === null || t < tMin || t >= bestT) continue;
    bestT = t;
    const gn = normalize(cross(sub(tr.v[1], tr.v[0]), sub(tr.v[2], tr.v[0])));
    const front = dot(gn, d) < 0;
    best = { i, t, n: front ? gn : scale(gn, -1), m: tr.m };
  }
  return best;
}

// Primary-hit aux extraction: albedo + world normal at the first surface.
function primaryAux(scene, o, d) {
  const hit = traceClosest(scene, o, d, 1e-4, 1e30);
  if (!hit) return { albedo: [0, 0, 0], normal: [0, 0, 0] };
  return { albedo: scene.materials[hit.m].albedo, normal: hit.n };
}

// Path integrator — NEE-free, pure cosine-bounce w/ emitter accumulation.
// (Simple + unbiased; high-spp converges to the same image as low-spp's mean.)
function integrate(scene, o0, d0, rng, maxBounces) {
  let tp = [1, 1, 1], rad = [0, 0, 0], o = o0, d = d0;
  for (let b = 0; b < maxBounces; b++) {
    const hit = traceClosest(scene, o, d, 1e-4, 1e30);
    if (!hit) { rad = add(rad, mul(tp, scene.env(d))); break; }
    const mat = scene.materials[hit.m];
    if (maxComp(mat.emission) > 0) rad = add(rad, mul(tp, mat.emission));
    const hitPos = add(o, scale(d, hit.t));
    const wi = cosineSample(rng, hit.n);
    tp = mul(tp, mat.albedo); // Lambert: (albedo/π)·cosθ/(cosθ/π) = albedo
    if (b > 2) {
      const surv = Math.min(Math.max(maxComp(tp), 0.1), 0.95);
      if (lcg(rng) > surv) break;
      tp = scale(tp, 1 / surv);
    }
    o = add(hitPos, scale(hit.n, 1e-3));
    d = wi;
  }
  return rad;
}

// ── Camera: simple pinhole, looking down -z from outside the box opening ──────
function makeCamera(size, frameIdx, pairs) {
  // Camera sits INSIDE the box (near the open -z... actually inside) looking
  // toward the back wall (+z back wall at z=-1). Vary the eye position slightly
  // per frame so the pairs are diverse cameras (dataset_spec.md §1).
  const jitterX = ((frameIdx / Math.max(1, pairs)) - 0.5) * 0.6; // [-0.3,0.3]
  const jitterY = ((frameIdx % 2) ? 0.15 : -0.15);
  const eye = [jitterX, jitterY, 0.9];   // inside, near the front, off-centre
  const target = [jitterX * 0.3, 0, -1]; // look at the back wall
  const fwd = normalize(sub(target, eye));
  const right = normalize(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  const fov = 1.1; // radians vertical (wide so walls fill frame)
  const tanH = Math.tan(fov / 2);
  return { eye, fwd, right, up, tanH };
}
function rayFor(cam, px, py, size) {
  // NDC in [-1,1], y up.
  const ndcX = (px + 0.5) / size * 2 - 1;
  const ndcY = 1 - (py + 0.5) / size * 2;
  const aspect = 1;
  const dir = normalize(add(cam.fwd, add(
    scale(cam.right, ndcX * cam.tanH * aspect),
    scale(cam.up, ndcY * cam.tanH),
  )));
  return { o: cam.eye, d: dir };
}

// ── Tonemap + encode ─────────────────────────────────────────────────────────
const reinhard = (l) => l / (1 + l);
function toByte(x) { return Math.max(0, Math.min(255, Math.round(x * 255))); }

// Minimal PNG encoder (RGB, 8-bit, no interlace). Uses node zlib.
function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xEDB88320 & -(c & 1));
  }
  return (~c) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const body = Buffer.concat([typeBuf, data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}
function encodePNG(rgb, w, h) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0; // RGB
  // raw with per-row filter byte 0
  const stride = w * 3;
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0;
    rgb.copy ? rgb.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride)
             : raw.set(rgb.subarray(y * stride, y * stride + stride), y * (stride + 1) + 1);
  }
  const idat = deflateSync(raw);
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}

// ── Render one image (color, given spp) ──────────────────────────────────────
function renderColor(scene, cam, size, spp, rng, maxBounces) {
  const out = Buffer.alloc(size * size * 3);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let acc = [0, 0, 0];
      for (let s = 0; s < spp; s++) {
        const { o, d } = rayFor(cam, px, py, size);
        acc = add(acc, integrate(scene, o, d, rng, maxBounces));
      }
      const c = scale(acc, 1 / spp);
      const i = (py * size + px) * 3;
      out[i] = toByte(reinhard(c[0]));
      out[i + 1] = toByte(reinhard(c[1]));
      out[i + 2] = toByte(reinhard(c[2]));
    }
  }
  return out;
}
function renderAux(scene, cam, size) {
  const albedo = Buffer.alloc(size * size * 3);
  const normal = Buffer.alloc(size * size * 3);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const { o, d } = rayFor(cam, px, py, size);
      const { albedo: a, normal: n } = primaryAux(scene, o, d);
      const i = (py * size + px) * 3;
      albedo[i] = toByte(a[0]); albedo[i + 1] = toByte(a[1]); albedo[i + 2] = toByte(a[2]);
      // encode world normal n*0.5+0.5
      normal[i] = toByte(n[0] * 0.5 + 0.5);
      normal[i + 1] = toByte(n[1] * 0.5 + 0.5);
      normal[i + 2] = toByte(n[2] * 0.5 + 0.5);
    }
  }
  return { albedo, normal };
}

// ── Main ─────────────────────────────────────────────────────────────────────
function main() {
  const args = parseArgs(process.argv);
  const scene = buildScene();
  const sceneDir = join(args.out, args.scene);
  const noisyDir = join(sceneDir, 'noisy');
  const cleanDir = join(sceneDir, 'clean');
  mkdirSync(noisyDir, { recursive: true });
  mkdirSync(cleanDir, { recursive: true });

  console.log(`[capture] scene=${args.scene} pairs=${args.pairs} size=${args.size} ` +
              `noisy=${args.noisySpp}spp clean=${args.cleanSpp}spp seed=${args.seed}`);

  for (let f = 0; f < args.pairs; f++) {
    const tag = `frame_${String(f + 1).padStart(4, '0')}`;
    const cam = makeCamera(args.size, f, args.pairs);

    // Deterministic per-pass seeds.
    const noisyRng = makeRng(args.seed * 1000003 + f * 31 + 1);
    const cleanRng = makeRng(args.seed * 1000003 + f * 31 + 2);

    const noisy = renderColor(scene, cam, args.size, args.noisySpp, noisyRng, args.maxBounces);
    const clean = renderColor(scene, cam, args.size, args.cleanSpp, cleanRng, args.maxBounces);
    const { albedo, normal } = renderAux(scene, cam, args.size);

    writeFileSync(join(noisyDir, `${tag}.png`), encodePNG(noisy, args.size, args.size));
    writeFileSync(join(noisyDir, `${tag}_albedo.png`), encodePNG(albedo, args.size, args.size));
    writeFileSync(join(noisyDir, `${tag}_normal.png`), encodePNG(normal, args.size, args.size));
    writeFileSync(join(cleanDir, `${tag}.png`), encodePNG(clean, args.size, args.size));
    console.log(`[capture] wrote ${tag} (noisy+albedo+normal+clean)`);
  }
  console.log(`[capture] done → ${sceneDir}`);
}

main();
