#!/usr/bin/env node
/**
 * diff-baselines.mjs — compare two directories of reference-render PNGs.
 *
 * Modes (degrade gracefully based on what's installed):
 *
 *   1. **byte-identical check (always available)** — SHA-256 hash + file size.
 *      Catches bit-exact regressions. For pure structural refactors (W1, most
 *      of W2) the expectation is byte-identical: anything else is a bug.
 *
 *   2. **pixel-diff (when `pngjs` is installed)** — decode both PNGs, compute
 *      mean absolute difference per channel + max-pixel-Δ + percent-different.
 *      Catches small numerical drift that's allowed by FP rounding.
 *
 * Exit code: 0 if every candidate matches its baseline within tolerance;
 *            1 if any candidate diverges or is missing a baseline counterpart.
 *
 * Usage:
 *   node tools/reference-renders/diff-baselines.mjs \
 *     --candidate tools/reference-renders/session-20260517 \
 *     --baseline  tools/reference-renders/baseline \
 *     [--tolerance 0.001]   # mean-abs-diff threshold in [0,1] (default 0.001)
 *
 * Output:
 *   A summary table to stdout, ready to paste into a PR body or sweep-diff
 *   report. Sample line:
 *     OK   cornell-glass.png   sha=eq  size=1234567 → 1234567
 *     DIFF cornell-glass.png   sha=ne  size=1234567 → 1234580   mae=0.0023  px>1%=0.4%
 *     MISS hero-product-viz.png   (baseline missing)
 */
import { readFile, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';

// ── arg parsing ──────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = { tolerance: 0.001 };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--candidate') out.candidate = argv[++i];
    else if (arg === '--baseline') out.baseline = argv[++i];
    else if (arg === '--tolerance') out.tolerance = Number(argv[++i]);
    else if (arg === '-h' || arg === '--help') {
      console.log(
        'Usage: node tools/reference-renders/diff-baselines.mjs --candidate DIR --baseline DIR [--tolerance F]',
      );
      process.exit(0);
    }
  }
  if (!out.candidate || !out.baseline) {
    console.error('Both --candidate and --baseline are required.');
    process.exit(2);
  }
  return out;
}

// ── PNG decode (best-effort via pngjs if available) ──────────────────────────

let PNGCtor = null;
try {
  const pngjs = await import('pngjs');
  PNGCtor = pngjs.PNG;
} catch {
  // pngjs not installed — fall back to size+hash only.
}

async function decodePng(path) {
  if (!PNGCtor) return null;
  const buf = await readFile(path);
  return new Promise((res, rej) => {
    new PNGCtor().parse(buf, (err, png) => {
      if (err) rej(err);
      else res(png);
    });
  });
}

function pixelDiff(a, b) {
  // a and b are pngjs.PNG instances with .width, .height, .data (RGBA Uint8).
  if (a.width !== b.width || a.height !== b.height) {
    return { kind: 'shape-mismatch', a: [a.width, a.height], b: [b.width, b.height] };
  }
  const n = a.data.length; // RGBA bytes
  let sumAbs = 0;
  let maxPix = 0;
  let nDiffPx = 0;
  const PX_DIFF_THRESHOLD = 8; // out of 255 per channel
  for (let i = 0; i < n; i += 4) {
    const dr = Math.abs(a.data[i]     - b.data[i]);
    const dg = Math.abs(a.data[i + 1] - b.data[i + 1]);
    const db = Math.abs(a.data[i + 2] - b.data[i + 2]);
    sumAbs += dr + dg + db;
    const px = Math.max(dr, dg, db);
    if (px > maxPix) maxPix = px;
    if (px > PX_DIFF_THRESHOLD) nDiffPx++;
  }
  const totalPx = (a.width * a.height);
  const mae = sumAbs / (totalPx * 3 * 255);
  return {
    kind: 'ok',
    mae,
    maxPix,
    pctDiff: 100 * nDiffPx / totalPx,
    totalPx,
  };
}

// ── main ─────────────────────────────────────────────────────────────────────

const args = parseArgs(process.argv);
const candidateDir = resolve(args.candidate);
const baselineDir  = resolve(args.baseline);

console.log(`candidate: ${candidateDir}`);
console.log(`baseline:  ${baselineDir}`);
console.log(`pngjs:     ${PNGCtor ? 'available (pixel-diff enabled)' : 'NOT installed (size+hash only)'}`);
console.log(`tolerance: mae < ${args.tolerance}`);
console.log('');

let candidates;
try {
  candidates = (await readdir(candidateDir)).filter((f) => f.endsWith('.png'));
} catch (err) {
  console.error(`Candidate dir not readable: ${err.message}`);
  process.exit(2);
}

if (candidates.length === 0) {
  console.error('No .png files in candidate directory.');
  process.exit(2);
}

let nFail = 0;

for (const name of candidates.sort()) {
  const cPath = join(candidateDir, name);
  const bPath = join(baselineDir, name);

  let bSize;
  try {
    bSize = (await stat(bPath)).size;
  } catch {
    console.log(`MISS ${name}   (no baseline at ${bPath})`);
    nFail++;
    continue;
  }

  const [cBuf, bBuf] = await Promise.all([readFile(cPath), readFile(bPath)]);
  const cSha = createHash('sha256').update(cBuf).digest('hex').slice(0, 12);
  const bSha = createHash('sha256').update(bBuf).digest('hex').slice(0, 12);
  const cSize = cBuf.length;

  if (cSha === bSha) {
    console.log(`OK   ${name.padEnd(36)} sha=eq  size=${cSize}`);
    continue;
  }

  // Diverged — try pixel diff if pngjs is available.
  if (PNGCtor) {
    try {
      const [cPng, bPng] = await Promise.all([decodePng(cPath), decodePng(bPath)]);
      const diff = pixelDiff(cPng, bPng);
      if (diff.kind === 'shape-mismatch') {
        console.log(
          `DIFF ${name.padEnd(36)} SHAPE c=${diff.a.join('x')} vs b=${diff.b.join('x')}`,
        );
        nFail++;
      } else if (diff.mae < args.tolerance) {
        console.log(
          `OK*  ${name.padEnd(36)} sha=ne  mae=${diff.mae.toFixed(5)} pctDiff=${diff.pctDiff.toFixed(2)}% maxPix=${diff.maxPix} (within tolerance)`,
        );
      } else {
        console.log(
          `DIFF ${name.padEnd(36)} sha=ne  size=${bSize}→${cSize}  mae=${diff.mae.toFixed(5)} pctDiff=${diff.pctDiff.toFixed(2)}% maxPix=${diff.maxPix}`,
        );
        nFail++;
      }
    } catch (err) {
      console.log(`DIFF ${name.padEnd(36)} sha=ne  size=${bSize}→${cSize}  (pngjs decode failed: ${err.message})`);
      nFail++;
    }
  } else {
    console.log(`DIFF ${name.padEnd(36)} sha=ne  size=${bSize}→${cSize}  (install pngjs for pixel-diff)`);
    nFail++;
  }
}

console.log('');
console.log(nFail === 0 ? '[diff-baselines] all match' : `[diff-baselines] ${nFail} divergence(s)`);
process.exit(nFail === 0 ? 0 : 1);
