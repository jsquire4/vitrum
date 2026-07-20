/**
 * Headless WebGPU adapter probe for WSL2.
 *
 * Serves tools/gpu-env/probe.html on a secure (localhost) origin and launches
 * Playwright's bundled Chromium with a configurable WebGPU/Vulkan backend, then
 * prints adapter limits + the repo's tier booleans (ptWebgpuFullTier /
 * ptWebgpuLiteTier / hybridCanRun).
 *
 * Backends (env VITRUM_GPU_BACKEND):
 *   swiftshader  — Chromium's built-in software Vulkan (baseline; below the
 *                  pt-webgpu lite floor of 8 storage buffers / 4 storage textures,
 *                  see packages/pt-webgpu/src/webgpuLimits.ts)
 *   vulkan       — Dawn over the system Vulkan loader, honouring VK_ICD_FILENAMES
 *                  / VK_DRIVER_FILES (point these at lavapipe to get its limits)
 *   default      — no special flags beyond --enable-unsafe-webgpu
 *
 * Useful env:
 *   VK_ICD_FILENAMES / VK_DRIVER_FILES — restrict the Vulkan loader to one ICD
 *   VITRUM_HEADLESS=0 — run headed (needs a display)
 *   VITRUM_EXTRA_FLAGS="--a,--b" — extra Chromium flags, comma-separated
 */
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const backend = (process.env.VITRUM_GPU_BACKEND ?? 'vulkan').toLowerCase();
const headless = process.env.VITRUM_HEADLESS !== '0';
const PORT = Number(process.env.VITRUM_PROBE_PORT ?? '0'); // 0 = OS-assigned free port

function backendArgs() {
  const common = [
    '--enable-unsafe-webgpu',
    '--disable-dev-shm-usage',
  ];
  if (backend === 'swiftshader') {
    return [
      ...common,
      '--enable-features=Vulkan',
      '--use-vulkan=swiftshader',
      '--enable-unsafe-swiftshader',
      '--use-angle=swiftshader',
    ];
  }
  if (backend === 'vulkan') {
    // Dawn picks the Vulkan backend; the Vulkan loader then resolves the ICD
    // from VK_ICD_FILENAMES / VK_DRIVER_FILES (e.g. lavapipe).
    return [
      ...common,
      '--enable-features=Vulkan',
      '--use-vulkan',
      '--disable-vulkan-surface',
      '--ignore-gpu-blocklist',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ];
  }
  return [...common, '--enable-features=Vulkan'];
}

function startServer() {
  const server = http.createServer(async (req, res) => {
    try {
      const name = (req.url ?? '/').split('?')[0] === '/' ? 'probe.html' : (req.url ?? '').slice(1);
      const body = await readFile(join(here, name.split('?')[0]));
      const ct = name.endsWith('.html') ? 'text/html' : 'application/octet-stream';
      res.writeHead(200, { 'content-type': ct });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end('not found');
    }
  });
  return new Promise((res) => server.listen(PORT, '127.0.0.1', () => res(server)));
}

function serverPort(server) {
  const a = server.address();
  return typeof a === 'object' && a ? a.port : PORT;
}

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (e) {
  console.error('Playwright not available:', String(e));
  process.exit(3);
}

const extra = (process.env.VITRUM_EXTRA_FLAGS ?? '').split(',').map((s) => s.trim()).filter(Boolean);
const args = [...backendArgs(), ...extra];
const server = await startServer();
const port = serverPort(server);

const launchEnv = { ...process.env };
console.error(`[probe] backend=${backend} headless=${headless}`);
console.error(`[probe] flags: ${args.join(' ')}`);
console.error(`[probe] VK_ICD_FILENAMES=${launchEnv.VK_ICD_FILENAMES ?? '(unset)'} VK_DRIVER_FILES=${launchEnv.VK_DRIVER_FILES ?? '(unset)'}`);

let browser;
const consoleLines = [];
try {
  browser = await chromium.launch({ headless, args, env: launchEnv });
  const page = await browser.newPage();
  page.on('console', (m) => consoleLines.push(`[console:${m.type()}] ${m.text()}`));
  await page.goto(`http://127.0.0.1:${port}/probe.html`, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  const result = await page.waitForFunction(() => globalThis.VITRUM_PROBE_RESULT, null, { timeout: 30_000 })
    .then((h) => h.jsonValue())
    .catch(() => ({ ok: false, reason: 'probe timed out' }));
  console.log(JSON.stringify({ backend, headless, flags: args, ...result }, null, 2));
  process.exitCode = result?.ok ? 0 : 1;
} catch (e) {
  console.error('launch/probe failed:', String(e));
  if (consoleLines.length) console.error(consoleLines.join('\n'));
  process.exitCode = 4;
} finally {
  if (browser) await browser.close();
  server.close();
}
