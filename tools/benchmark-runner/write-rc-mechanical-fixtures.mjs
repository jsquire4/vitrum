/**
 * Writes 64×64 mechanical RC acceptance PNGs (not GPU captures).
 * Used by `benchmark:rc-acceptance-mechanical` and default CI until a GPU
 * GPU captures live in `W8-rc-{off,on}/` and are refreshed via
 * `benchmark:rc-acceptance-gpu` only.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { PNG } from 'pngjs';
import { getRepoRoot } from './repoRoot.mjs';

const repoRoot = getRepoRoot(import.meta.url);

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
  const offDir = resolve(repoRoot, 'tools/reference-renders/W8-rc-mechanical-off');
  const onDir = resolve(repoRoot, 'tools/reference-renders/W8-rc-mechanical-on');
  await mkdir(offDir, { recursive: true });
  await mkdir(onDir, { recursive: true });
  const offPath = resolve(offDir, 'cornell-walkaround-rc-off.png');
  const onPath = resolve(onDir, 'cornell-walkaround-rc-on.png');
  await writeFile(offPath, solidPng(64, 64, [40, 42, 48]));
  await writeFile(onPath, solidPng(64, 64, [220, 140, 60]));
  console.log(`[rc-fixtures] wrote ${offPath}`);
  console.log(`[rc-fixtures] wrote ${onPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
