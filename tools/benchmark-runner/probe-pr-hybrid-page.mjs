/**
 * Quick poll of walkaround.html bench globals (Windows GPU host).
 */
import { launchWebGpuBrowser } from './launchWebGpuBrowser.mjs';

const port = process.env.VITRUM_BENCH_DEV_PORT ?? '5176';
const url =
  process.env.VITRUM_PROBE_BENCH_URL ??
  `http://127.0.0.1:${port}/walkaround.html?prBench=material-churn&prBenchAuto=1&mode=walkaround&prBenchIters=5`;

const { chromium } = await import('playwright');
const origin = new URL(url).origin + '/';
const { browser } = await launchWebGpuBrowser(chromium, origin);
const page = await browser.newPage();
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });

for (let i = 0; i < 36; i += 1) {
  await page.waitForTimeout(5_000);
  const snap = await page.evaluate(() => ({
    status: document.querySelector('#status')?.textContent?.slice(0, 240) ?? '',
    eng: globalThis.__vitrumWalkaround?.state ?? null,
    tel: globalThis.__vitrum?.walkaround?.state ?? null,
    prLast: globalThis.__vitrumPrBenchLast ?? null,
    prApi: globalThis.__vitrumPrBench != null,
  }));
  console.log(JSON.stringify({ sec: (i + 1) * 5, ...snap }));
  if (snap.prLast != null) break;
  if (snap.eng === 'ready' && snap.prApi && i >= 2) {
    const run = await page.evaluate(async () => {
      const api = globalThis.__vitrumPrBench;
      if (api == null) return { err: 'no api' };
      return api.run('material-churn');
    });
    console.log('manual run:', JSON.stringify(run));
    break;
  }
}

await browser.close();
