#!/usr/bin/env -S deno run --sloppy-imports --allow-read --allow-env --allow-run --allow-net=127.0.0.1 --allow-sys --allow-write
// @ts-nocheck -- browser-side WebGPU types are evaluated by Chromium.
/**
 * Chromium/Tint compile + real pipeline-creation gate for portable shared-BVH
 * compositions and the shipped ReSTIR-GI/NRC composition.
 */
import { chromium } from "../../node_modules/playwright/index.mjs";
import { composeWgsl } from "../../packages/walkaround-hybrid/src/pipeline/wgslComposer.ts";
import { WGSL_MODULES } from "../../packages/walkaround-hybrid/src/pipeline/wgslModules.ts";
import { buildRisGiNrcModule } from "../../packages/walkaround-hybrid/src/shaders/risGiNrc.wgsl.ts";
import { SHARED_BVH_PORTABLE_COMPOSITIONS } from "./sharedBvhPortableCompositions.mjs";

const nrcConfig = {
  levels: 16,
  featuresPerEntry: 2,
  oneBlobBins: 4,
  width: 64,
  outWidth: 3,
  hidden: 5,
};

function requiredBindGroupCount(code) {
  const groups = [...code.matchAll(/@group\s*\(\s*(\d+)\s*\)/g)]
    .map((match) => Number(match[1]));
  return groups.length === 0 ? 0 : Math.max(...groups) + 1;
}

const nrcCode = composeWgsl(buildRisGiNrcModule(nrcConfig), WGSL_MODULES);
const sharedOnly = Deno.args.includes("--shared-only");
const compositions = [
  ...SHARED_BVH_PORTABLE_COMPOSITIONS.map((entry) => ({
    ...entry,
    requiredMaxBindGroups: requiredBindGroupCount(entry.code),
  })),
  ...sharedOnly ? [] : [{
    name: "walkaround-hybrid/risGiNrc-verbatim",
    code: nrcCode,
    entryPoint: "risGiMain",
    requiredMaxBindGroups: requiredBindGroupCount(nrcCode),
  }],
];

const configuredBrowser = Deno.env.get("VITRUM_TINT_BROWSER");
const browserCandidates = [
  configuredBrowser,
  "/snap/bin/chromium",
  "/usr/bin/chromium",
  "/usr/bin/google-chrome",
  "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
  "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
].filter((path, index, all) => path && all.indexOf(path) === index);

const launchProfiles = [
  {
    name: "vulkan",
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-angle=vulkan",
      "--disable-vulkan-surface",
      "--disable-gpu-sandbox",
      "--no-sandbox",
    ],
  },
  {
    name: "native",
    args: ["--enable-unsafe-webgpu", "--disable-gpu-sandbox", "--no-sandbox"],
  },
  {
    name: "swiftshader",
    args: [
      "--enable-unsafe-webgpu",
      "--enable-unsafe-swiftshader",
      "--use-angle=swiftshader",
      "--disable-gpu-sandbox",
      "--no-sandbox",
    ],
  },
];

async function exists(path) {
  try {
    return (await Deno.stat(path)).isFile;
  } catch {
    return false;
  }
}

const serverAbort = new AbortController();
const server = Deno.serve({
  hostname: "127.0.0.1",
  port: 0,
  signal: serverAbort.signal,
  onListen() {},
}, () => new Response("<!doctype html><meta charset=utf-8><title>Vitrum Tint gate</title>", {
  headers: { "content-type": "text/html; charset=utf-8" },
}));
const pageUrl = `http://127.0.0.1:${server.addr.port}/`;
const failures = [];
let bestPartial;
let success;

try {
  browserSearch:
  for (const executablePath of browserCandidates) {
    if (!(await exists(executablePath))) continue;
    for (const profile of launchProfiles) {
      let browser;
      try {
        browser = await chromium.launch({
          executablePath,
          headless: true,
          timeout: 30_000,
          args: profile.args,
        });
        const page = await browser.newPage();
        await page.goto(pageUrl, { waitUntil: "load", timeout: 15_000 });
        const result = await page.evaluate(async (entries) => {
          if (!globalThis.isSecureContext) {
            throw new Error("loopback page is not a secure context");
          }
          if (!navigator.gpu) throw new Error("navigator.gpu is unavailable");
          const adapter = await navigator.gpu.requestAdapter();
          if (!adapter) throw new Error("requestAdapter() returned null");
          const supported = entries.filter((entry) =>
            (entry.requiredMaxBindGroups ?? 0) <= adapter.limits.maxBindGroups
          );
          const blocked = entries.filter((entry) =>
            (entry.requiredMaxBindGroups ?? 0) > adapter.limits.maxBindGroups
          );
          const maxRequiredBindGroups = Math.max(
            4,
            ...supported.map((entry) => entry.requiredMaxBindGroups ?? 0),
          );
          const device = await adapter.requestDevice({
            requiredLimits: { maxBindGroups: maxRequiredBindGroups },
          });
          const compiled = [];
          for (const entry of supported) {
            const module = device.createShaderModule({ label: entry.name, code: entry.code });
            const info = await module.getCompilationInfo();
            const compileErrors = info.messages.filter((message) => message.type === "error");
            if (compileErrors.length > 0) {
              throw new Error(
                `${entry.name} Tint compile: ` +
                compileErrors.slice(0, 3).map((message) =>
                  `${message.lineNum}:${message.linePos} ${message.message}`).join(" | "),
              );
            }
            device.pushErrorScope("validation");
            let pipelineFailure;
            try {
              await device.createComputePipelineAsync({
                label: `${entry.name}/${entry.entryPoint}`,
                layout: "auto",
                compute: { module, entryPoint: entry.entryPoint },
              });
            } catch (error) {
              pipelineFailure = String(error?.message ?? error);
            }
            const scopedError = await device.popErrorScope();
            if (pipelineFailure || scopedError) {
              throw new Error(
                `${entry.name} Tint pipeline: ` +
                `${pipelineFailure ?? scopedError?.message ?? scopedError}`,
              );
            }
            compiled.push(entry.name);
          }
          return {
            compiled,
            blocked: blocked.map((entry) => ({
              name: entry.name,
              requiredMaxBindGroups: entry.requiredMaxBindGroups,
            })),
            adapter: adapter.info ? {
              vendor: adapter.info.vendor,
              architecture: adapter.info.architecture,
              device: adapter.info.device,
              description: adapter.info.description,
            } : null,
            maxBindGroups: adapter.limits.maxBindGroups,
          };
        }, compositions);

        const attempt = {
          ...result,
          browserVersion: browser.version(),
          executablePath,
          profileName: profile.name,
        };
        if (!bestPartial || attempt.compiled.length > bestPartial.compiled.length) {
          bestPartial = attempt;
        }
        if (result.blocked.length === 0) {
          success = attempt;
          break browserSearch;
        }
        failures.push(
          `${executablePath} (${profile.name}): ` +
          result.blocked.map((entry) =>
            `${entry.name} needs maxBindGroups=${entry.requiredMaxBindGroups}; ` +
            `adapter exposes ${result.maxBindGroups}`
          ).join(" | "),
        );
      } catch (error) {
        failures.push(`${executablePath} (${profile.name}): ${String(error?.message ?? error)}`);
      } finally {
        await browser?.close().catch(() => {});
      }
    }
  }
} finally {
  serverAbort.abort();
  await server.finished.catch(() => {});
}

if (success) {
  console.log("[tint-gate] PASS");
  console.log(`  browser       : ${success.browserVersion}`);
  console.log(`  executable    : ${success.executablePath}`);
  console.log(`  launch profile: ${success.profileName}`);
  console.log(`  pipelines     : ${success.compiled.length}/${compositions.length}`);
  console.log(`  maxBindGroups : ${success.maxBindGroups}`);
  if (success.adapter) console.log(`  adapter       : ${JSON.stringify(success.adapter)}`);
  for (const name of success.compiled) console.log(`  TINT+PIPE     : ${name}`);
  Deno.exit(0);
}

console.error("[tint-gate] FAIL: no local Chromium launch produced all pipelines");
if (bestPartial) {
  console.error(`  best partial  : ${bestPartial.compiled.length}/${compositions.length}`);
  for (const name of bestPartial.compiled) console.error(`  TINT+PIPE     : ${name}`);
  for (const entry of bestPartial.blocked) {
    console.error(
      `  CAPABILITY    : ${entry.name} needs maxBindGroups=${entry.requiredMaxBindGroups}; ` +
      `adapter exposes ${bestPartial.maxBindGroups}`,
    );
  }
}
for (const failure of failures) console.error(`  ${failure}`);
Deno.exit(1);
