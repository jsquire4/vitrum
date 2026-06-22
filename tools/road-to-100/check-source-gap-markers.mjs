#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Scans production package source for high-risk gap markers. Each allowed hit
// must name a deliberate compatibility/future-contract boundary, so future
// audits do not repeatedly rediscover the same intentional rows as code gaps.

const ROOTS = ["packages"];
const SOURCE_EXTENSIONS = [".ts", ".tsx", ".wgsl.ts", ".glsl.js"];
const MARKER_RE = /\b(TODO|FIXME|not implemented|stub|stubs)\b/i;

const ALLOWED_MARKERS = Object.freeze([
  {
    path: "packages/gltf-adapter/src/compression.ts",
    includes: "stub), else warn and leave the data unresolved",
    reason: "EXT_meshopt fallback-stub contract comment; real decoder-hook handling is implemented.",
  },
  {
    path: "packages/gltf-adapter/src/compression.ts",
    includes: "Keep gltf.buffers index-consistent (pad any gap with zero-length stubs).",
    reason: "Internal buffer-index preservation for compression fallback analysis.",
  },
  {
    path: "packages/gltf-adapter/src/compression.ts",
    includes: "unless that buffer is a `fallback: true` stub",
    reason: "glTF meshopt fallback-buffer semantics, not a no-op implementation.",
  },
  {
    path: "packages/gltf-adapter/src/gltfTypes.ts",
    includes: "EXT_meshopt_compression `fallback: true` stubs",
    reason: "Schema field documentation for meshopt fallback buffers.",
  },
  {
    path: "packages/dev/src/react/DenoiserABToggle.tsx",
    includes: "Other backends render a stub badge and log once if the debug surface is absent.",
    reason: "Dev overlay intentionally degrades when an engine lacks the optional debug surface.",
  },
  {
    path: "packages/dev/src/react/DenoiserABToggle.tsx",
    includes: "engine.debug.setDenoiserEnabled() is not implemented on this backend",
    reason: "Optional debug control warning, not a core rendering feature gap.",
  },
  {
    path: "packages/dev/src/react/DenoiserABToggle.tsx",
    includes: "? 'denoiser [stub]'",
    reason: "Dev overlay badge for optional debug-control absence.",
  },
  {
    path: "packages/engine/src/progressiveHandoff.ts",
    includes: "fully unit-testable with stub engines",
    reason: "Testability comment for orchestration code.",
  },
  {
    path: "packages/engine/src/adapterProfile.ts",
    includes: "`{ limits }` stub. `selectPtWebgpuTraceTier`",
    reason: "Adapter profile helper accepts minimal limit objects for tests/hosts.",
  },
  {
    path: "packages/engine/src/adapterProfile.ts",
    includes: "test stubs handle that",
    reason: "Capability-probe documentation for test doubles.",
  },
  {
    path: "packages/walkaround-hybrid/src/HybridEngine.ts",
    includes: "Non-'none' strategies are not implemented for this backend.",
    reason: "Structured caustic-strategy rejection on walkaround; PT backends own those paths.",
  },
  {
    path: "packages/walkaround-hybrid/src/neural/nrc/nrcQueryHarness.ts",
    includes: "Stub the spread-termination + reservoir symbols the query module does NOT",
    reason: "Standalone NRC validation harness isolates nrcQuery forward math from unrelated GI spread/reservoir code.",
  },
  {
    path: "packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts",
    includes: "line acknowledged this as a stub). This module replaces that",
    reason: "Historical comment documenting a removed PPG stub.",
  },
  {
    path: "packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts",
    includes: "no more uniform-grid stub",
    reason: "Historical comment documenting the real flat-buffer PPG implementation.",
  },
  {
    path: "packages/walkaround-hybrid/src/pipeline/BvhUpdateSink.ts",
    includes: "tested with a lightweight stub instead of a real GPU pipeline",
    reason: "Dependency-injection testability comment.",
  },
  {
    path: "packages/walkaround-hybrid/src/environment/equirectDirectional.ts",
    includes: "the fork carries a TODO",
    reason: "Comment notes this implementation fixed the inherited TODO by including solid angle.",
  },
  {
    path: "packages/pt-webgl2/src/scene/equirectHdrInfo.ts",
    includes: "unlike the original fork TODO",
    reason: "Comment notes this implementation fixed the inherited TODO by including solid angle.",
  },
  {
    path: "packages/pt-webgpu/src/bdpt/bdptLightPathBufferWebGPU.ts",
    includes: "tests use stub devices without global GPUBufferUsage",
    reason: "Host/test-device compatibility comment.",
  },
  {
    path: "packages/pt-webgpu/src/wgsl/pathTrace/causticLite.wgsl.ts",
    includes: "Lite caustic stubs",
    reason: "Lite-tier caustic paths are intentionally disabled and capability-gated.",
  },
  {
    path: "packages/pt-webgpu/src/wgsl/pathTrace/connectCore.wgsl.ts",
    includes: "lite tier appends its stub / procedural-only implementations",
    reason: "WGSL include comment for lite-tier compatibility implementations.",
  },
]);

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[road-to-100-source-gap-scan] ${message}`);
}

/** @param {string} path */
function isProductionSource(path) {
  if (!path.includes("/src/")) return false;
  if (path.includes("/__tests__/")) return false;
  if (path.endsWith(".test.ts") || path.endsWith(".test.tsx") || path.endsWith(".d.ts")) return false;
  return SOURCE_EXTENSIONS.some((ext) => path.endsWith(ext));
}

/**
 * @param {string} dir
 * @returns {AsyncGenerator<string>}
 */
async function* walk(dir) {
  for await (const entry of Deno.readDir(repoUrl(dir))) {
    const path = `${dir}/${entry.name}`;
    if (entry.isDirectory) {
      if (entry.name === "node_modules" || entry.name === "dist" || entry.name === "coverage") continue;
      yield* walk(path);
    } else if (entry.isFile && isProductionSource(path)) {
      yield path;
    }
  }
}

const allowByPath = new Map();
for (const marker of ALLOWED_MARKERS) {
  const rows = allowByPath.get(marker.path) ?? [];
  rows.push(marker);
  allowByPath.set(marker.path, rows);
}

/** @type {{ path: string; line: number; text: string }[]} */
const unclassified = [];
/** @type {Set<string>} */
const seenAllowed = new Set();

for (const root of ROOTS) {
  for await (const path of walk(root)) {
    const text = await Deno.readTextFile(repoUrl(path));
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (!MARKER_RE.test(line)) continue;
      const allowed = (allowByPath.get(path) ?? []).find((marker) => line.includes(marker.includes));
      if (allowed) {
        seenAllowed.add(`${allowed.path}\n${allowed.includes}`);
      } else {
        unclassified.push({ path, line: i + 1, text: line.trim() });
      }
    }
  }
}

if (unclassified.length > 0) {
  const details = unclassified
    .map((hit) => `  ${hit.path}:${hit.line}: ${hit.text}`)
    .join("\n");
  fail(`unclassified production gap markers found:\n${details}`);
}

const staleAllowed = ALLOWED_MARKERS.filter((marker) => !seenAllowed.has(`${marker.path}\n${marker.includes}`));
if (staleAllowed.length > 0) {
  const details = staleAllowed
    .map((marker) => `  ${marker.path}: ${marker.includes}`)
    .join("\n");
  fail(`allowed marker list is stale; expected marker text disappeared:\n${details}`);
}

console.log(
  `[road-to-100-source-gap-scan] PASS (${ALLOWED_MARKERS.length} classified production markers, ` +
  `${unclassified.length} unclassified)`,
);
