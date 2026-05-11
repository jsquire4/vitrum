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
  /** Matches Cornell query `vitrumDisplay` (raw | bilateral | oidn | wgsl | svgf). */
  const display = process.env.VITRUM_DISPLAY;
  /** ONNX model URL for OIDN (`vitrumOidnModel`). */
  const oidnModel = process.env.VITRUM_OIDN_MODEL;
  /** WebGPU bilateral sigma (`vitrumWgslSigma`). */
  const wgslSigma = process.env.VITRUM_WGSL_SIGMA;
  /** `0` disables shared WebGPU device (`vitrumWebGpuShared`). */
  const webgpuShared = process.env.VITRUM_WEBGPU_SHARED;
  const svgfFrames = process.env.VITRUM_SVGF_FRAME_COUNT ?? process.env.VITRUM_SVGF_FRAMES;
  const svgfAtrous = process.env.VITRUM_SVGF_ATROUS;
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
    if (display === 'raw' || display === 'bilateral' || display === 'oidn' || display === 'wgsl' || display === 'svgf') {
      u.searchParams.set('vitrumDisplay', display);
    }
    if (oidnModel != null && oidnModel.length > 0) {
      u.searchParams.set('vitrumOidnModel', oidnModel);
    }
    if (wgslSigma != null && wgslSigma.length > 0) {
      u.searchParams.set('vitrumWgslSigma', wgslSigma);
    }
    if (webgpuShared === '0' || webgpuShared === '1') {
      u.searchParams.set('vitrumWebGpuShared', webgpuShared);
    }
    if (svgfFrames != null && svgfFrames.length > 0) {
      u.searchParams.set('vitrumSvgfFrameCount', svgfFrames);
    }
    if (svgfAtrous != null && svgfAtrous.length > 0) {
      u.searchParams.set('vitrumSvgfAtrous', svgfAtrous);
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
