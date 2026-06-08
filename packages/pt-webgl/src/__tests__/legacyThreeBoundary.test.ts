import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const THREE_EDGE =
  /from\s+['"](?:three(?:\/|['"])|three-gpu-pathtracer['"])|@vitrum\/three-bindings/;

function walkFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      out.push(...walkFiles(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

function normalizePath(file: string, root: string): string {
  return path.relative(root, file).split(path.sep).join('/');
}

describe('@vitrum/pt-webgl legacy THREE boundary', () => {
  it('keeps THREE/fork adapter imports in the legacy quarantine or engine core', () => {
    const srcRoot = fileURLToPath(new URL('..', import.meta.url));
    const allowedRootFiles = new Set(['ptEngineWebGL2.ts']);

    const offenders = walkFiles(srcRoot)
      .filter((file) => file.endsWith('.ts'))
      .map((file) => ({ file, rel: normalizePath(file, srcRoot) }))
      .filter(({ rel }) => !rel.startsWith('__tests__/'))
      .filter(({ rel }) => !rel.startsWith('legacy/three/'))
      .filter(({ rel }) => !allowedRootFiles.has(rel))
      .filter(({ file }) => THREE_EDGE.test(readFileSync(file, 'utf8')))
      .map(({ rel }) => rel);

    expect(offenders).toEqual([]);
  });
});
