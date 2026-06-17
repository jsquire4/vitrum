#!/usr/bin/env node
/**
 * Full-repo line-by-line gap signal scanner.
 * Usage: node tools/gap-scan/scan.mjs
 * Outputs: plan/.gap-scan-raw.json, plan/.gap-scan-prod-high.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const OUT_DIR = path.join(ROOT, 'plan');
const ROOTS = [
  path.join(ROOT, 'packages'),
  path.join(ROOT, 'tools'),
  path.join(ROOT, 'examples'),
];
const SKIP_DIRS = new Set(['node_modules', 'coverage', 'dist', '.vite', '__snapshots__']);
const EXTS = new Set(['.ts', '.tsx', '.js', '.mjs']);

const PATTERNS = [
  ['throw', /\bthrow\s+new\s+Error\b/],
  ['throw_generic', /\bthrow\s+/],
  ['unsupported', /unsupported/i],
  ['approximate', /approximate/i],
  ['not_implemented', /not\s+implement/i],
  ['todo', /\bTODO\b|\bFIXME\b|\bHACK\b|\bXXX\b/],
  ['reserved', /@reserved|reserved\s+for\s+phase|Phase\s*2/i],
  ['stub', /\bstub\b|\bplaceholder\b|\bno-?op\b/i],
  ['deferred', /\bdeferred\b|\bfollow-?up\b|\bremaining\b|\btail\b/i],
  ['fallback', /fallback-rebuild|fallbackRebuild|fallback\s*:/i],
  ['downgrade', /downgrade/i],
  ['warn', /onWarning|EngineWarning|console\.warn|warnOnce/i],
  ['unreachable', /unreachable|not\s+yet|not\s+supported|does\s+not\s+support/i],
];

function isProd(file) {
  return !file.includes('/__tests__/') && !file.endsWith('.test.ts') && !file.endsWith('.test.tsx');
}

function walk(dir, files = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    if (ent.isDirectory()) {
      if (SKIP_DIRS.has(ent.name)) continue;
      walk(path.join(dir, ent.name), files);
    } else {
      const ext = path.extname(ent.name);
      if (!EXTS.has(ext) && !ent.name.endsWith('.wgsl.ts')) continue;
      files.push(path.join(dir, ent.name));
    }
  }
  return files;
}

const findings = [];
const fileStats = { files: 0, lines: 0 };

for (const root of ROOTS) {
  if (!fs.existsSync(root)) continue;
  for (const filePath of walk(root)) {
    const rel = path.relative(ROOT, filePath);
    const parts = rel.split(path.sep);
    const pkg = parts[0] === 'packages' && parts.length > 1 ? parts[1] : parts[0];
    const text = fs.readFileSync(filePath, 'utf8');
    const lines = text.split('\n');
    fileStats.files += 1;
    fileStats.lines += lines.length;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const stripped = line.trim();
      if (!stripped) continue;
      for (const [kind, pat] of PATTERNS) {
        if (!pat.test(line)) continue;
        if (kind === 'approximate' || kind === 'unsupported') {
          if (rel.includes('/__tests__/') && line.includes('expect')) continue;
        }
        findings.push({ file: rel, line: i + 1, kind, text: stripped.slice(0, 240), pkg });
      }
    }
  }
}

const prod = findings.filter((x) => isProd(x.file));
const HIGH = new Set(['throw', 'not_implemented', 'todo', 'reserved', 'downgrade', 'fallback']);
const seen = new Set();
const highProd = [];
for (const x of prod) {
  if (!HIGH.has(x.kind)) continue;
  const k = `${x.file}:${x.line}:${x.kind}`;
  if (seen.has(k)) continue;
  seen.add(k);
  highProd.push(x);
}

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUT_DIR, '.gap-scan-raw.json'), JSON.stringify({ fileStats, findings }, null, 0));
fs.writeFileSync(path.join(OUT_DIR, '.gap-scan-prod-high.json'), JSON.stringify({ count: highProd.length, items: highProd }, null, 2));

console.log(`gap-scan: ${fileStats.files} files, ${fileStats.lines} lines, ${findings.length} signals, ${highProd.length} high-signal production`);
