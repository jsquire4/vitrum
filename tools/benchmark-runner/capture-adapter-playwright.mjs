import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';

const outputPng = process.env.VITRUM_OUTPUT_PNG;
if (!outputPng) {
  console.error('VITRUM_OUTPUT_PNG is required.');
  process.exit(2);
}

const captureUrl = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5173/';
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

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto(captureUrl, { waitUntil: 'networkidle', timeout: timeoutMs });
  await page.waitForTimeout(Math.max(0, settleMs));
  const captureReady = await page
    .evaluate(() => {
      const maybeReady = (globalThis).VITRUM_CAPTURE_READY;
      return maybeReady === true;
    })
    .catch(() => false);
  if (!captureReady) {
    // Best effort; many pages won't expose this sentinel.
    await page.waitForTimeout(1500);
  }

  const locator = page.locator(captureSelector).first();
  await locator.waitFor({ timeout: Math.max(1000, timeoutMs / 2) });
  await mkdir(dirname(outputPng), { recursive: true });
  await locator.screenshot({ path: outputPng });

  const telemetry = await page
    .evaluate(() => {
      const value = (globalThis).VITRUM_MS_PER_SAMPLE;
      return typeof value === 'number' && Number.isFinite(value) ? value : null;
    })
    .catch(() => null);

  if (telemetry != null) {
    console.log(JSON.stringify({ msPerSample: telemetry }));
  }
} finally {
  await browser.close();
}
