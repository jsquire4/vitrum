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
 *   <out>/<scene>/noisy/frame_NNNN.bin          (low-spp colour, linear-HDR VHDR)
 *   <out>/<scene>/noisy/frame_NNNN_albedo.png   (primary-hit albedo, 8-bit)
 *   <out>/<scene>/noisy/frame_NNNN_normal.png   (world normal, n*0.5+0.5, 8-bit)
 *   <out>/<scene>/clean/frame_NNNN.bin          (high-spp colour, linear-HDR VHDR)
 *
 * Determinism: a single Park-Miller LCG seeded per (scene, frame, spp-pass)
 * from the CLI --seed. Re-running with the same flags is bit-identical.
 *
 * Usage:
 *   node capture-dataset.mjs --out data_smoke --pairs 4 --size 128 \
 *     --noisy-spp 1 --clean-spp 256 --seed 1984
 *
 * Random scene mode (--scene random):
 *   Each pair renders a freshly randomized scene — random geometry (boxes and
 *   sphere-approximated meshes), random diffuse/metal-rough materials, random
 *   emitter placement/size/intensity, optional sky radiance, and a random camera
 *   always aimed at the scene centre. Seeded per-frame for full reproducibility.
 *   This mode is designed for producing a diverse training set so the denoiser
 *   sees varied normals, depth distributions, and albedo statistics.
 *
 *   Example:
 *     node capture-dataset.mjs --out data_real --pairs 128 --size 128 \
 *       --noisy-spp 1 --clean-spp 256 --seed 42 --scene random
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
  console.log('  --scene <name>     scene type: cornell_box | random (default cornell_box)');
  console.log('                     random = per-pair randomized geometry/materials/lights/camera');
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

// ── Random scene generator (--scene random) ───────────────────────────────────
// Builds a fully seeded, randomized scene each call. No two frames look the
// same: geometry layout, material palette, emitter, sky tint all vary.
//
// Primitives supported by this CPU tracer:
//   • Axis-aligned box (6 quads = 12 triangles).  Supports any position/size.
//   • Sphere approximation: uv-sphere subdivided to nLat×nLon quads (24 tris
//     at 3×8 — enough for plausible curved-surface normals at 128px).
//   • Floor + enclosing walls always present as the base room.
//
// Materials:
//   • Diffuse only in the path integrator (integratePath uses Lambertian),
//     but the albedo buffer carries chromatically varied colours the denoiser
//     must learn to handle — matte walls, metallic-look dark/bright combos,
//     saturated hues.
//
// Emitter:
//   Exactly one area light per scene (rect or ceiling strip), position +
//   size + intensity all randomized within physically reasonable ranges.
//
// Sky:
//   ~50% chance of a coloured sky (env radiance function returns a tinted
//   constant, not black). Drives contrast between sky-lit open gaps and
//   enclosed lit scenes — diversifies the background/normal distribution.

/**
 * Append triangles for an axis-aligned box [cx±hx, cy±hy, cz±hz] with
 * material index m into the given tris array.
 */
function addBox(tris, cx, cy, cz, hx, hy, hz, m) {
  const quad = (a, b, c, d) => {
    tris.push({ v: [a, b, c], m });
    tris.push({ v: [a, c, d], m });
  };
  const x0 = cx - hx, x1 = cx + hx;
  const y0 = cy - hy, y1 = cy + hy;
  const z0 = cz - hz, z1 = cz + hz;
  // -x face
  quad([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]);
  // +x face
  quad([x1, y0, z1], [x1, y1, z1], [x1, y1, z0], [x1, y0, z0]);
  // -y face (bottom)
  quad([x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]);
  // +y face (top)
  quad([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]);
  // -z face
  quad([x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z0]);
  // +z face
  quad([x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]);
}

/**
 * Append triangles approximating a sphere centred at (cx,cy,cz) with radius r,
 * material index m. Uses 3 latitude × 8 longitude bands (24 quads = 48 tris).
 * Sufficient for plausible curved-surface normals at 128px.
 */
function addSphere(tris, cx, cy, cz, r, m) {
  const nLat = 3, nLon = 8;
  // Build vertex grid [nLat+1][nLon].
  const pts = [];
  for (let i = 0; i <= nLat; i++) {
    const theta = (Math.PI * i) / nLat; // 0..π
    const row = [];
    for (let j = 0; j < nLon; j++) {
      const phi = (2 * Math.PI * j) / nLon;
      row.push([
        cx + r * Math.sin(theta) * Math.cos(phi),
        cy + r * Math.cos(theta),
        cz + r * Math.sin(theta) * Math.sin(phi),
      ]);
    }
    pts.push(row);
  }
  for (let i = 0; i < nLat; i++) {
    for (let j = 0; j < nLon; j++) {
      const j1 = (j + 1) % nLon;
      const a = pts[i][j], b = pts[i][j1], c = pts[i + 1][j1], d = pts[i + 1][j];
      tris.push({ v: [a, b, c], m });
      tris.push({ v: [a, c, d], m });
    }
  }
}

/**
 * Build a randomized scene seeded by `sceneSeed`.
 *
 * Every structural choice — room colour, number of objects, their kind
 * (box vs sphere), position, size, material, emitter placement / intensity,
 * presence of sky — is derived from the seed so the result is bit-identical
 * for the same seed.
 */
function buildRandomScene(sceneSeed) {
  const rng = makeRng(sceneSeed);
  const r = () => lcg(rng);
  const rRange = (lo, hi) => lo + r() * (hi - lo);
  const rInt = (lo, hi) => Math.floor(rRange(lo, hi + 0.9999));

  const materials = [];
  const tris = [];

  // ── Room walls (always present) ────────────────────────────────────────────
  // Room spans [-1,1]^3. Use random muted colour for floor/ceiling/back wall
  // and two contrasting colours for left/right walls (like Cornell but varied).
  const mkDiffuse = (r_, g_, b_) => ({ albedo: [r_, g_, b_], emission: [0, 0, 0] });
  const mkEmitter = (intensity, r_, g_, b_) => ({
    albedo: [0, 0, 0],
    emission: [intensity * r_, intensity * g_, intensity * b_],
  });

  // Wall colours: floor/ceiling/back = neutral; left/right = coloured pair
  const floorGrey = rRange(0.5, 0.85);
  materials.push(mkDiffuse(floorGrey, floorGrey, floorGrey));  // 0 floor/ceiling/back

  // Left wall: random hue (dominated channel), mid-brightness
  const lh = rInt(0, 2); // 0=R,1=G,2=B dominant
  const lv = rRange(0.4, 0.8);
  const leftAlbedo = [lh === 0 ? lv : rRange(0.05, 0.25), lh === 1 ? lv : rRange(0.05, 0.25), lh === 2 ? lv : rRange(0.05, 0.25)];
  materials.push(mkDiffuse(...leftAlbedo));    // 1 left wall

  // Right wall: orthogonal hue
  const rh = (lh + 1 + rInt(0, 1)) % 3;
  const rv = rRange(0.4, 0.8);
  const rightAlbedo = [rh === 0 ? rv : rRange(0.05, 0.25), rh === 1 ? rv : rRange(0.05, 0.25), rh === 2 ? rv : rRange(0.05, 0.25)];
  materials.push(mkDiffuse(...rightAlbedo));   // 2 right wall

  // ── Emitter ────────────────────────────────────────────────────────────────
  // Random area light: position on ceiling (y close to +1) or on back wall,
  // intensity in [5,20], tinted white/warm/cool.
  const emitOnCeiling = r() > 0.3; // 70% ceiling, 30% back-wall
  const emitIntensity = rRange(5, 20);
  const emitTint = rInt(0, 2); // 0=neutral,1=warm,2=cool
  const emitR = emitTint === 1 ? 1.0 : emitTint === 2 ? 0.7 : 1.0;
  const emitG = emitTint === 1 ? 0.9 : emitTint === 2 ? 0.85 : 1.0;
  const emitB = emitTint === 1 ? 0.6 : emitTint === 2 ? 1.0 : 1.0;
  materials.push(mkEmitter(emitIntensity, emitR, emitG, emitB)); // 3 emitter

  const L = 1;
  const quad = (a, b, c, d, m) => {
    tris.push({ v: [a, b, c], m });
    tris.push({ v: [a, c, d], m });
  };

  // Room shell (floor, ceiling, back, left, right — open toward +z for camera)
  quad([-L, -L, -L], [L, -L, -L], [L, -L, L], [-L, -L, L], 0); // floor
  quad([-L,  L,  L], [L,  L,  L], [L,  L, -L], [-L,  L, -L], 0); // ceiling
  quad([-L, -L, -L], [-L,  L, -L], [L,  L, -L], [L, -L, -L], 0); // back wall
  quad([-L, -L,  L], [-L,  L,  L], [-L,  L, -L], [-L, -L, -L], 1); // left
  quad([ L, -L, -L], [ L,  L, -L], [ L,  L,  L], [ L, -L,  L], 2); // right

  // Emitter geometry
  const ez = rRange(0.15, 0.5);
  const eOff = [rRange(-0.5, 0.5), rRange(-0.5, 0.5)];
  if (emitOnCeiling) {
    const ey = L - 0.02;
    quad(
      [eOff[0] - ez, ey, eOff[1] - ez],
      [eOff[0] + ez, ey, eOff[1] - ez],
      [eOff[0] + ez, ey, eOff[1] + ez],
      [eOff[0] - ez, ey, eOff[1] + ez],
      3
    );
  } else {
    // Back-wall strip emitter
    const eyLo = rRange(-0.6, 0.2), eyHi = eyLo + rRange(0.2, 0.5);
    const exHalf = rRange(0.15, 0.6);
    const ez2 = -L + 0.02;
    quad(
      [eOff[0] - exHalf, eyLo, ez2],
      [eOff[0] + exHalf, eyLo, ez2],
      [eOff[0] + exHalf, eyHi, ez2],
      [eOff[0] - exHalf, eyHi, ez2],
      3
    );
  }

  // ── Objects ────────────────────────────────────────────────────────────────
  // 2–5 objects: mix of boxes and spheres with random bright/dark materials
  const nObjs = rInt(2, 5);
  const objMats = [];
  for (let k = 0; k < nObjs; k++) {
    // Random saturated or near-neutral albedo
    const h = r() * 6; // [0,6) hue
    const s = rRange(0.0, 1.0); // saturation
    const v = rRange(0.25, 0.9); // value
    // HSV to RGB (simple inline)
    const i_ = Math.floor(h);
    const f = h - i_;
    const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    let ro, go, bo;
    switch (i_ % 6) {
      case 0: ro=v; go=t; bo=p; break;
      case 1: ro=q; go=v; bo=p; break;
      case 2: ro=p; go=v; bo=t; break;
      case 3: ro=p; go=q; bo=v; break;
      case 4: ro=t; go=p; bo=v; break;
      default: ro=v; go=p; bo=q;
    }
    objMats.push(materials.length);
    materials.push(mkDiffuse(Math.max(0.02, ro), Math.max(0.02, go), Math.max(0.02, bo)));
  }

  // Place objects on the floor, non-overlapping (approximate — just spaced)
  const placed = []; // [{cx,cz,r}] for overlap check
  for (let k = 0; k < nObjs; k++) {
    const isSphere = r() > 0.5;
    const sz = rRange(0.12, 0.38); // half-size / radius
    const m = objMats[k];

    // Try a few random positions; pick first that doesn't obviously overlap
    let cx = 0, cz = 0, ok = false;
    for (let attempt = 0; attempt < 12; attempt++) {
      cx = rRange(-0.7 + sz, 0.7 - sz);
      cz = rRange(-0.8 + sz, 0.3 - sz);
      let clear = true;
      for (const p of placed) {
        const dx = cx - p.cx, dz = cz - p.cz;
        if (Math.sqrt(dx * dx + dz * dz) < sz + p.r + 0.08) { clear = false; break; }
      }
      if (clear) { ok = true; break; }
    }
    if (!ok) continue; // skip if no non-overlapping position found

    placed.push({ cx, cz, r: sz });
    if (isSphere) {
      const cy = -L + sz;
      addSphere(tris, cx, cy, cz, sz, m);
    } else {
      // random height box
      const hy = rRange(sz * 0.6, sz * 1.8);
      addBox(tris, cx, -L + hy, cz, sz, hy, sz, m);
    }
  }

  // ── Sky / environment ──────────────────────────────────────────────────────
  // 50% of scenes have a non-black sky — provides directional ambient variety.
  let envFn;
  if (r() > 0.5) {
    // Coloured sky: gradient based on y-component of direction
    const skyR = rRange(0.1, 0.8), skyG = rRange(0.1, 0.8), skyB = rRange(0.2, 1.0);
    const intensity = rRange(0.3, 2.0);
    envFn = (d) => {
      const t = Math.max(0, d[1]) * intensity;
      return [skyR * t, skyG * t, skyB * t];
    };
  } else {
    envFn = () => [0, 0, 0];
  }

  return { tris, materials, env: envFn };
}

/**
 * Build a random camera looking at the scene centre for a randomized scene.
 * Camera always looks roughly toward the origin (where objects are placed),
 * from a random position on a hemisphere outside the front of the room.
 *
 * @param {object} rng - seeded LCG state (consumed in place)
 */
function makeRandomCamera(rng) {
  const r_ = () => lcg(rng);
  const rRange = (lo, hi) => lo + r_() * (hi - lo);

  // Random camera position on the +z hemisphere at distance [0.5,1.2]
  // and height [-0.4,0.6], lateral offset [-0.5,0.5]
  const eyeX = rRange(-0.5, 0.5);
  const eyeY = rRange(-0.3, 0.5);
  const eyeZ = rRange(0.6, 1.2);
  const eye = [eyeX, eyeY, eyeZ];

  // Aim at a random point near the scene centre (not exactly origin — varies focus)
  const targetX = rRange(-0.25, 0.25);
  const targetY = rRange(-0.4, 0.2);
  const targetZ = rRange(-0.6, 0.0);
  const target = [targetX, targetY, targetZ];

  const fwd = normalize(sub(target, eye));
  // Use world up = [0,1,0]; if camera looks straight up degenerate case — handle
  const worldUp = Math.abs(fwd[1]) < 0.95 ? [0, 1, 0] : [1, 0, 0];
  const right = normalize(cross(fwd, worldUp));
  const up = cross(right, fwd);

  // FOV: random between 50°–90° vertical
  const fovDeg = rRange(50, 90);
  const tanH = Math.tan((fovDeg * Math.PI / 180) / 2);

  return { eye, fwd, right, up, tanH };
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
// (color targets are written as linear HDR .bin — see encodeVHDR below; toByte
// is used only for the albedo/normal 8-bit PNG G-buffers.)
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
    if (rgb.copy) {
      rgb.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride);
    } else {
      raw.set(rgb.subarray(y * stride, y * stride + stride), y * (stride + 1) + 1);
    }
  }
  const idat = deflateSync(raw);
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}

// ── Linear-HDR color container (VHDR .bin) ────────────────────────────────────
// The runtime pack shader (neuralPack.wgsl) feeds RAW LINEAR HDR color to the
// denoiser (no tonemap). Training must match that, so color targets are written
// as linear float32 RGB — NOT reinhard-tonemapped 8-bit PNG. train.py loads this
// via _load_hdr / _load_hdr_np. Albedo + normal G-buffers stay 8-bit PNG.
//
// Header (little-endian): u32 magic ('VHDR' = 0x52444856), u32 version (1),
// u32 width, u32 height, then width*height*3 float32 linear radiance (row-major
// interleaved RGB).
const VHDR_MAGIC = 0x52444856;
const VHDR_VERSION = 1;
function encodeVHDR(linearRgb, w, h) {
  const header = Buffer.alloc(16);
  header.writeUInt32LE(VHDR_MAGIC, 0);
  header.writeUInt32LE(VHDR_VERSION, 4);
  header.writeUInt32LE(w, 8);
  header.writeUInt32LE(h, 12);
  const body = Buffer.alloc(w * h * 3 * 4);
  for (let i = 0; i < linearRgb.length; i++) body.writeFloatLE(linearRgb[i], i * 4);
  return Buffer.concat([header, body]);
}

// ── Render one image (color, given spp) → linear HDR float32 RGB ──────────────
function renderColor(scene, cam, size, spp, rng, maxBounces) {
  const out = new Float32Array(size * size * 3);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let acc = [0, 0, 0];
      for (let s = 0; s < spp; s++) {
        const { o, d } = rayFor(cam, px, py, size);
        acc = add(acc, integrate(scene, o, d, rng, maxBounces));
      }
      const c = scale(acc, 1 / spp);
      const i = (py * size + px) * 3;
      // Store RAW LINEAR radiance (no reinhard, no 8-bit quantization) so the
      // training color matches the runtime raw-linear denoiser input.
      out[i] = c[0];
      out[i + 1] = c[1];
      out[i + 2] = c[2];
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
  const isRandom = args.scene === 'random';

  if (!isRandom && args.scene !== 'cornell_box') {
    console.error(`Unknown --scene value: "${args.scene}". Valid options: cornell_box | random`);
    process.exit(2);
  }

  // For cornell_box: scene is built once, camera jitters per frame.
  // For random: scene + camera are rebuilt per frame from a per-frame seed.
  const cornellScene = isRandom ? null : buildScene();

  const sceneDir = join(args.out, args.scene);
  const noisyDir = join(sceneDir, 'noisy');
  const cleanDir = join(sceneDir, 'clean');
  mkdirSync(noisyDir, { recursive: true });
  mkdirSync(cleanDir, { recursive: true });

  console.log(`[capture] scene=${args.scene} pairs=${args.pairs} size=${args.size} ` +
              `noisy=${args.noisySpp}spp clean=${args.cleanSpp}spp seed=${args.seed}`);

  for (let f = 0; f < args.pairs; f++) {
    const tag = `frame_${String(f + 1).padStart(4, '0')}`;

    // Per-frame scene and camera.
    let scene, cam;
    if (isRandom) {
      // Each frame has its own scene seed (derived from base seed + frame index)
      // and a separate camera seed. Both are fully deterministic.
      const sceneSeed = args.seed * 1000003 + f * 7919;
      const camSeed   = args.seed * 999983  + f * 6271 + 3;
      scene = buildRandomScene(sceneSeed);
      cam   = makeRandomCamera(makeRng(camSeed));
    } else {
      scene = cornellScene;
      cam   = makeCamera(args.size, f, args.pairs);
    }

    // Deterministic per-pass seeds (noisy/clean independent).
    const noisyRng = makeRng(args.seed * 1000003 + f * 31 + 1);
    const cleanRng = makeRng(args.seed * 1000003 + f * 31 + 2);

    const noisy = renderColor(scene, cam, args.size, args.noisySpp, noisyRng, args.maxBounces);
    const clean = renderColor(scene, cam, args.size, args.cleanSpp, cleanRng, args.maxBounces);
    const { albedo, normal } = renderAux(scene, cam, args.size);

    // Color targets → linear-HDR .bin; albedo/normal G-buffers → 8-bit PNG.
    writeFileSync(join(noisyDir, `${tag}.bin`), encodeVHDR(noisy, args.size, args.size));
    writeFileSync(join(noisyDir, `${tag}_albedo.png`), encodePNG(albedo, args.size, args.size));
    writeFileSync(join(noisyDir, `${tag}_normal.png`), encodePNG(normal, args.size, args.size));
    writeFileSync(join(cleanDir, `${tag}.bin`), encodeVHDR(clean, args.size, args.size));
    if (f % 16 === 0 || f === args.pairs - 1) {
      console.log(`[capture] wrote ${tag} (noisy+albedo+normal+clean)  [${f + 1}/${args.pairs}]`);
    }
  }
  console.log(`[capture] done → ${sceneDir}`);
}

main();
