/**
 * Dumps Chromium's chrome://gpu WebGPU/Vulkan diagnostic so we can see which
 * backend Dawn actually selected and why hardware/lavapipe was rejected.
 */

const { chromium } = await import('playwright');
const args = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-vulkan',
  '--ignore-gpu-blocklist',
  '--disable-gpu-sandbox',
  '--no-sandbox',
  '--disable-dev-shm-usage',
];
const browser = await chromium.launch({ headless: true, args, env: { ...process.env } });
const page = await browser.newPage();
await page.goto('chrome://gpu', { waitUntil: 'domcontentloaded', timeout: 60_000 }).catch(() => {});
await page.waitForTimeout(2000);
const text = await page.evaluate(() => document.body?.innerText ?? '').catch(() => '');
// Print only the lines we care about.
const wanted = /vulkan|webgpu|dawn|swiftshader|software|gl_renderer|gl_vendor|adapter|sandbox|llvmpipe|lavapipe/i;
const lines = text.split('\n').map((l) => l.trim()).filter((l) => l && wanted.test(l));
console.log(lines.slice(0, 80).join('\n') || '(no matching lines; chrome://gpu may be empty in this build)');
await browser.close();
