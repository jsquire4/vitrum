/**
 * PR-D6 polish — smoke PNG references under tools/reference-renders/PR-hybrid/.
 *
 * Requires hybrid-capable WebGPU adapter (≥16 storage buffers / stage).
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchDevServer, stopDevServer, waitForServerReady } from './devServer.mjs';
import { launchWebGpuBrowser } from './launchWebGpuBrowser.mjs';
import { getRepoRoot } from './repoRoot.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = getRepoRoot(import.meta.url);
const refRoot = resolve(repoRoot, 'tools/reference-renders/PR-hybrid');

const SCENARIOS = [
  {
    scenarioId: 'PR-hybrid-tlas-10-inst',
    dir: 'tlas-on',
    query: {
      mode: 'walkaround',
      scene: 'tlas10inst',
      bvhMode: 'tlas',
      samplesTarget: '32',
      prBench: 'frame-sample',
      prBenchAuto: '1',
      prBenchFrames: '24',
    },
  },
  {
    scenarioId: 'PR-hybrid-material-churn',
    dir: 'material-edit',
    query: {
      mode: 'walkaround',
      samplesTarget: '32',
      prBench: 'material-churn',
      prBenchAuto: '1',
      prBenchIters: '20',
    },
  },
  {
    scenarioId: 'PR-hybrid-200k-static',
    dir: '200k-static',
    query: {
      mode: 'walkaround',
      scene: 'bench200k',
      targetTriangles: '200000',
      samplesTarget: '16',
      prBench: 'frame-sample',
      prBenchAuto: '1',
      prBenchFrames: '24',
    },
  },
];

const benchPort = process.env.VITRUM_BENCH_DEV_PORT ?? '5175';
const startServer = process.env.VITRUM_PR_REF_START_SERVER !== '0';

async function main() {
  let devServer = null;
  let base = `http://127.0.0.1:${benchPort}/`;
  if (startServer) {
    devServer = launchDevServer(
      `npm run dev --workspace @vitrum-examples/two-engines-one-scene -- --host 127.0.0.1 --port ${benchPort}`,
      repoRoot,
    );
    const ready = await waitForServerReady(devServer, base, 90_000, 500);
    base = ready.url.endsWith('/') ? ready.url : `${ready.url}/`;
  }

  const { chromium } = await import('playwright');
  const { browser, caps } = await launchWebGpuBrowser(chromium, base);
  if (!caps.hybridCanRun) {
    console.warn(
      `[pr-ref-capture] adapter insufficient for hybrid (buffers=${caps.maxStorageBuffersPerShaderStage}); skipping PNG capture.`,
    );
    await browser.close();
    if (devServer) stopDevServer(devServer);
    process.exit(0);
  }

  const manifest = [];
  for (const row of SCENARIOS) {
    const outDir = resolve(refRoot, row.dir);
    await mkdir(outDir, { recursive: true });
    const pngPath = resolve(outDir, `${row.scenarioId}.png`);
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const u = new URL(`${base}walkaround.html`);
    for (const [k, v] of Object.entries(row.query)) {
      u.searchParams.set(k, v);
    }
    try {
      await page.goto(u.toString(), { waitUntil: 'domcontentloaded', timeout: 90_000 });
      await page.waitForFunction(
        () => globalThis.__vitrum?.walkaround?.state === 'ready' || globalThis.__vitrumPrBenchLast != null,
        null,
        { timeout: 180_000, polling: 300 },
      );
      const canvas = page.locator('#c-wgpu');
      await canvas.waitFor({ timeout: 15_000 });
      await canvas.screenshot({ path: pngPath });
      const hash = createHash('sha256').update(await readFile(pngPath)).digest('hex');
      manifest.push({ scenarioId: row.scenarioId, pngPath, hash });
      console.log(`[pr-ref-capture] wrote ${pngPath} sha256=${hash.slice(0, 12)}`);
    } catch (e) {
      console.error(`[pr-ref-capture] ${row.scenarioId} failed:`, e);
      manifest.push({ scenarioId: row.scenarioId, error: String(e) });
    } finally {
      await page.close();
    }
  }

  await browser.close();
  if (devServer) stopDevServer(devServer);

  const manifestPath = resolve(refRoot, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify({ generatedAt: new Date().toISOString(), manifest }, null, 2)}\n`);
  console.log(`[pr-ref-capture] manifest ${manifestPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
