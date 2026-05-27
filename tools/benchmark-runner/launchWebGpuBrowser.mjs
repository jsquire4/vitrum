/**
 * Launch Chromium with the best available WebGPU adapter for this host.
 * Probing runs on a secure origin (default http://127.0.0.1:5175/) because
 * navigator.gpu is unavailable on about:blank.
 */

import { WEBGPU_CHROMIUM_LAUNCH_CANDIDATES } from './playwrightWebGpu.mjs';

function tierScore(caps) {
  if (!caps?.ok) return -1;
  let score = 0;
  if (caps.ptWebgpuFullTier) score += 100;
  if (caps.hybridCanRun) score += 10;
  if (caps.ptWebgpuLiteTier) score += 1;
  return score;
}

/**
 * @param {import('playwright').Page} page
 */
export async function readWebGpuAdapterCaps(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) {
      return { ok: false, reason: 'no navigator.gpu' };
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { ok: false, reason: 'requestAdapter returned null' };
    }
    let vendor = '';
    let architecture = '';
    let description = '';
    try {
      const info = await adapter.requestAdapterInfo();
      vendor = info.vendor ?? '';
      architecture = info.architecture ?? '';
      description = info.description ?? '';
    } catch {
      /* optional API */
    }
    const limits = adapter.limits;
    return {
      ok: true,
      vendor,
      architecture,
      description,
      maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
      maxStorageTexturesPerShaderStage: limits.maxStorageTexturesPerShaderStage,
      ptWebgpuFullTier:
        limits.maxStorageBuffersPerShaderStage >= 10 &&
        limits.maxStorageTexturesPerShaderStage >= 5,
      ptWebgpuLiteTier:
        limits.maxStorageBuffersPerShaderStage >= 8 &&
        limits.maxStorageTexturesPerShaderStage >= 4,
      hybridCanRun:
        limits.maxStorageBuffersPerShaderStage >= 16 &&
        limits.maxStorageTexturesPerShaderStage >= 8,
    };
  });
}

/**
 * @param {import('playwright').Chromium} chromium
 * @param {string} [probeUrl] secure origin for adapter probe
 */
export async function launchWebGpuBrowser(chromium, probeUrl = 'http://127.0.0.1:5175/') {
  const attempts = [];

  for (let i = 0; i < WEBGPU_CHROMIUM_LAUNCH_CANDIDATES.length; i += 1) {
    const profile = i === 0 ? 'hardware' : 'swiftshader';
    const launchOptions = WEBGPU_CHROMIUM_LAUNCH_CANDIDATES[i];
    let browser;
    try {
      browser = await chromium.launch(launchOptions);
    } catch (err) {
      attempts.push({
        profile,
        ok: false,
        error: String(err instanceof Error ? err.message : err),
      });
      continue;
    }

    const page = await browser.newPage();
    try {
      await page.goto(probeUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    } catch (err) {
      attempts.push({
        profile,
        ok: false,
        error: `goto failed: ${err instanceof Error ? err.message : err}`,
      });
      await browser.close();
      continue;
    }

    // eslint-disable-next-line no-await-in-loop
    const caps = await readWebGpuAdapterCaps(page);
    attempts.push({ profile, ...caps });
    await browser.close();

    if (!caps.ok) continue;
    if (tierScore(caps) >= tierScore(attempts.find((a) => a.profile === 'hardware' && a.ok))) {
      /* keep scanning — want best profile */
    }
  }

  const ranked = attempts
    .filter((a) => a.ok)
    .sort((a, b) => tierScore(b) - tierScore(a));
  const best = ranked[0];
  if (best == null) {
    throw new Error(
      `No WebGPU adapter usable at ${probeUrl}. Attempts:\n${JSON.stringify(attempts, null, 2)}`,
    );
  }

  const launchIndex = best.profile === 'hardware' ? 0 : 1;
  const browser = await chromium.launch(WEBGPU_CHROMIUM_LAUNCH_CANDIDATES[launchIndex]);
  const page = await browser.newPage();
  return { browser, page, profile: best.profile, caps: best, attempts };
}
