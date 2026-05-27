/**
 * 64×64 mechanical PNGs for Sprint 10c / 14 A/B harness (not GPU captures).
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { PNG } from 'pngjs';
import { getRepoRoot } from './repoRoot.mjs';

const repoRoot = getRepoRoot(import.meta.url);
const outDir = resolve(repoRoot, 'tools/reference-renders/bdpt-layered-mechanical');

function solidPng(width, height, rgb) {
  const img = new PNG({ width, height });
  for (let i = 0; i < width * height; i += 1) {
    const o = i * 4;
    img.data[o] = rgb[0];
    img.data[o + 1] = rgb[1];
    img.data[o + 2] = rgb[2];
    img.data[o + 3] = 255;
  }
  return PNG.sync.write(img);
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const layered = resolve(outDir, 'cornell-layered.png');
  const bdpt = resolve(outDir, 'cornell-layered-bdpt.png');
  await writeFile(layered, solidPng(64, 64, [55, 90, 140]));
  await writeFile(bdpt, solidPng(64, 64, [200, 110, 50]));
  console.log(`[bdpt-fixtures] wrote ${layered}`);
  console.log(`[bdpt-fixtures] wrote ${bdpt}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
