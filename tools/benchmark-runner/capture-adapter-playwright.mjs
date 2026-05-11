import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';

const outputPng = process.env.VITRUM_OUTPUT_PNG;
if (!outputPng) {
  console.error('VITRUM_OUTPUT_PNG is required.');
  process.exit(2);
}

const captureUrlBase = process.env.VITRUM_CAPTURE_URL ?? 'http://127.0.0.1:5173/';

/**
 * Mapping table: ENV name → URL query key + optional validator.
 * The validator returns true when the env value should be forwarded as-is,
 * or false to skip. Default validator: non-empty string.
 */
const ENV_TO_QUERY = [
  { env: 'VITRUM_SCENARIO_ID',         query: 'vitrumScenario' },
  { env: 'VITRUM_SEED',                query: 'vitrumSeed' },
  { env: 'VITRUM_WIDTH',               query: 'vitrumWidth' },
  { env: 'VITRUM_HEIGHT',              query: 'vitrumHeight' },
  { env: 'VITRUM_BOUNCES',             query: 'vitrumBounces' },
  { env: 'VITRUM_SPP',                 query: 'vitrumSpp' },
  {
    env: 'VITRUM_CAUSTIC_STRATEGY',
    query: 'vitrumCaustic',
    validate: (v) => v !== 'candidate' && v !== 'baseline',
  },
  {
    env: 'VITRUM_DISPLAY',
    query: 'vitrumDisplay',
    validate: (v) => ['raw', 'bilateral', 'oidn', 'wgsl', 'svgf'].includes(v),
  },
  { env: 'VITRUM_OIDN_MODEL',          query: 'vitrumOidnModel' },
  { env: 'VITRUM_WGSL_SIGMA',          query: 'vitrumWgslSigma' },
  {
    env: 'VITRUM_WEBGPU_SHARED',
    query: 'vitrumWebGpuShared',
    validate: (v) => v === '0' || v === '1',
  },
  { env: 'VITRUM_SVGF_ATROUS',         query: 'vitrumSvgfAtrous' },
];

function captureUrlWithScenarioParams() {
  try {
    const u = new URL(captureUrlBase);
    for (const { env, query, validate } of ENV_TO_QUERY) {
      const val = process.env[env];
      if (!val || val.length === 0) continue;
      if (validate && !validate(val)) continue;
      u.searchParams.set(query, val);
    }
    u.searchParams.set('vitrumAutoStart', '1');
    // Special case: SVGF frame count accepts two env aliases.
    const svgfFrames = process.env.VITRUM_SVGF_FRAME_COUNT ?? process.env.VITRUM_SVGF_FRAMES;
    if (svgfFrames && svgfFrames.length > 0) {
      u.searchParams.set('vitrumSvgfFrameCount', svgfFrames);
    }
    return u.toString();
  } catch {
    return captureUrlBase;
  }
}
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

const jsHeapMb = Number(process.env.VITRUM_JS_HEAP_MB ?? '4096');
const browser = await chromium.launch({
  headless: true,
  args: [
    '--disable-dev-shm-usage',
    // 1 GiB default is too small for the OIDN model + large viewport reads;
    // raise to 4 GiB by default and let the env override either direction.
    `--js-flags=--max-old-space-size=${jsHeapMb}`,
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

  const selectorFromEnv = process.env.VITRUM_CAPTURE_SELECTOR;
  // Cornell sets globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR when raw vs denoise canvas should be snapped.
  const selector =
    selectorFromEnv != null && selectorFromEnv.length > 0
      ? selectorFromEnv
      : await page.evaluate(() => {
          const s = globalThis.VITRUM_CAPTURE_CANVAS_SELECTOR;
          return typeof s === 'string' && s.length > 0 ? s : 'canvas';
        });

  const locator = page.locator(selector).first();
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
