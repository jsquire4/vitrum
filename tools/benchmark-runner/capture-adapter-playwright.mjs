import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';

const outputPng = process.env.VITRUM_OUTPUT_PNG;
if (!outputPng) {
  console.error('VITRUM_OUTPUT_PNG is required.');
  process.exit(2);
}

const captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5173/';
function captureUrlWithScenarioParams() {
  const scenarioId = process.env.VITRUM_SCENARIO_ID;
  const seed = process.env.VITRUM_SEED;
  const w = process.env.VITRUM_WIDTH;
  const h = process.env.VITRUM_HEIGHT;
  const bounces = process.env.VITRUM_BOUNCES;
  const spp = process.env.VITRUM_SPP;
  const caustic = process.env.VITRUM_CAUSTIC_STRATEGY;
  try {
    const u = new URL(captureUrlBase);
    if (scenarioId) u.searchParams.set('vitrumScenario', scenarioId);
    if (seed) u.searchParams.set('vitrumSeed', seed);
    if (w) u.searchParams.set('vitrumWidth', w);
    if (h) u.searchParams.set('vitrumHeight', h);
    if (bounces) u.searchParams.set('vitrumBounces', bounces);
    if (spp) u.searchParams.set('vitrumSpp', spp);
    u.searchParams.set('vitrumAutoStart', '1');
    if (caustic && caustic !== 'candidate' && caustic !== 'baseline') {
      u.searchParams.set('vitrumCaustic', caustic);
    }
    return u.toString();
  } catch {
    return captureUrlBase;
  }
}
const captureSelector = process.env.VITRUM_CAPTURE_SELECTOR ?? 'canvas';
const settleMs = Number(process.env.VITRUM_CAPTURE_SETTLE_MS ?? '2500');
const timeoutMs = Number(process.env.VITRUM_CAPTURE_TIMEOUT_MS ?? '30000');

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  console.error(
    'Playwright is not installed in this workspace. Install it, or provide a different VITRUM_CAPTURE_CMD adapter.',
  );
  console.error(String(error));
  process.exit(3);
}

const browser = await chromium.launch({
  headless: true,
  args: [
    '--disable-dev-shm-usage',
    '--js-flags=--max-old-space-size=1024',
  ],
});
try {
  const page = await browser.newPage({
    viewport: {
      width: Math.max(1, Number(process.env.VITRUM_WIDTH ?? '1280')),
      height: Math.max(1, Number(process.env.VITRUM_HEIGHT ?? '720')),
    },
  });
  const captureUrl = captureUrlWithScenarioParams();
  await page.goto(captureUrl, { waitUntil: 'networkidle', timeout: timeoutMs });
  if (settleMs > 0) {
    await page.waitForTimeout(settleMs);
  }
  await page
    .waitForFunction(() => globalThis.VITRUM_CAPTURE_READY === true, null, {
      timeout: Math.max(1000, timeoutMs - settleMs),
    })
    .catch(async () => {
      // Best effort; many pages won't expose this sentinel.
      await page.waitForTimeout(1500);
    });

  const locator = page.locator(captureSelector).first();
  await locator.waitFor({ timeout: Math.max(1000, timeoutMs / 2) });
  await mkdir(dirname(outputPng), { recursive: true });
  await locator.screenshot({ path: outputPng });

  const telemetry = await page
    .evaluate(() => {
      const msPerSample = (globalThis).VITRUM_MS_PER_SAMPLE;
      const extra = (globalThis).VITRUM_CAPTURE_TELEMETRY;
      if (typeof msPerSample !== 'number' || !Number.isFinite(msPerSample)) return null;
      return {
        ...(extra && typeof extra === 'object' ? extra : {}),
        msPerSample,
      };
    })
    .catch(() => null);

  if (telemetry != null) {
    console.log(JSON.stringify(telemetry));
  }
} finally {
  await browser.close();
}
