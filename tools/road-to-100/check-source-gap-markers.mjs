#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Scans production package source for high-risk gap markers. Each allowed hit
// must name a deliberate compatibility/future-contract boundary, so future
// audits do not repeatedly rediscover the same intentional rows as code gaps.

const ROOTS = ["packages"];
const SOURCE_EXTENSIONS = [".ts", ".tsx", ".js", ".mjs"];
const MARKER_RE = /\b(?:TODO|FIXME|TBD|unimplemented|not[ -]implemented|partial(?:ly)?[ -]implemented|partial[ -]implementation|incomplete[ -]implementation|implementation[ -]incomplete|unfinished[ -]implementation|stub|stubs)\b/i;

export const ALLOWED_MARKERS = Object.freeze([
  {
    path: "packages/engine/src/progressiveHandoff.ts",
    line: "// no GPU calls of its own — so it is fully unit-testable with stub engines.",
    reason: "Testability comment for orchestration code.",
  },
  {
    path: "packages/engine/src/adapterProfile.ts",
    line: "*  `{ limits }` stub. `selectPtWebgpuTraceTier` reads only the two storage",
    reason: "Adapter profile helper accepts minimal limit objects for tests/hosts.",
  },
  {
    path: "packages/engine/src/adapterProfile.ts",
    line: "*  assumed to support WebGL2 (test stubs handle that). */",
    reason: "Capability-probe documentation for test doubles.",
  },
  {
    path: "packages/walkaround-hybrid/src/neural/nrc/nrcQueryHarness.ts",
    line: "// Stub the spread-termination + reservoir symbols the query module does NOT",
    reason: "Standalone NRC validation harness isolates nrcQuery forward math from unrelated GI spread/reservoir code.",
  },
  {
    path: "packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts",
    line: "* `_ = vIdx` line acknowledged this as a stub). This module replaces that",
    reason: "Historical comment documenting a removed PPG stub.",
  },
  {
    path: "packages/walkaround-hybrid/src/ppg/ppgUpdate.wgsl.ts",
    line: "// W9: real flat-buffer leaf location (no more uniform-grid stub).",
    reason: "Historical comment documenting the real flat-buffer PPG implementation.",
  },
  {
    path: "packages/walkaround-hybrid/src/pipeline/BvhUpdateSink.ts",
    line: "* can be tested with a lightweight stub instead of a real GPU pipeline.",
    reason: "Dependency-injection testability comment.",
  },
  {
    path: "packages/walkaround-hybrid/src/environment/equirectDirectional.ts",
    line: "* NOTE vs that port: we include the sinθ solid-angle term (the fork carries a TODO",
    reason: "Comment notes this implementation fixed the inherited TODO by including solid angle.",
  },
  {
    path: "packages/pt-webgl2/src/scene/equirectHdrInfo.ts",
    line: "// NOTE: unlike the original fork TODO, the per-pixel importance weight includes",
    reason: "Comment notes this implementation fixed the inherited TODO by including solid angle.",
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

/** @typedef {{ path: string, line: string, reason: string }} AllowedMarker */

/**
 * Validate all source entries against an exact-line allowlist. Trimming only
 * ignores indentation/trailing whitespace; adding any other text — including a
 * second gap marker — makes the line unclassified and fails closed.
 *
 * @param {{ path: string, text: string }[]} entries
 * @param {readonly AllowedMarker[]} [allowedMarkers]
 * @returns {{ classifiedCount: number, unclassifiedCount: number }}
 */
export function checkSourceEntries(entries, allowedMarkers = ALLOWED_MARKERS) {
  /** @type {Map<string, AllowedMarker[]>} */
  const allowByPath = new Map();
  /** @type {Set<string>} */
  const allowedMarkerKeys = new Set();
  for (const marker of allowedMarkers) {
    if (
      typeof marker.path !== "string" || marker.path.trim() === "" ||
      typeof marker.line !== "string" || marker.line.trim() === "" ||
      typeof marker.reason !== "string" || marker.reason.trim() === ""
    ) {
      fail("each allowed marker requires a nonempty path, exact trimmed full line, and reason");
    }
    if (marker.line !== marker.line.trim()) {
      fail(`allowed marker line must already be trimmed: ${marker.path}: ${marker.line}`);
    }
    if (!MARKER_RE.test(marker.line)) {
      fail(`allowed marker line does not contain a recognized gap marker: ${marker.path}: ${marker.line}`);
    }
    const key = `${marker.path}\n${marker.line}`;
    if (allowedMarkerKeys.has(key)) {
      fail(`duplicate allowed marker: ${marker.path}: ${marker.line}`);
    }
    allowedMarkerKeys.add(key);
    const rows = allowByPath.get(marker.path) ?? [];
    rows.push(marker);
    allowByPath.set(marker.path, rows);
  }

  /** @type {{ path: string; line: number; text: string }[]} */
  const unclassified = [];
  /** @type {Map<string, number>} */
  const allowedHitCounts = new Map();

  for (const { path, text } of entries) {
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (!MARKER_RE.test(line)) continue;
      const allowed = (allowByPath.get(path) ?? []).filter((marker) =>
        line.trim() === marker.line
      );
      if (allowed.length > 1) {
        fail(`ambiguous allowed marker match at ${path}:${i + 1}`);
      }
      if (allowed.length === 1) {
        const key = `${allowed[0].path}\n${allowed[0].line}`;
        const count = (allowedHitCounts.get(key) ?? 0) + 1;
        if (count > 1) fail(`allowed marker matched more than once: ${allowed[0].path}: ${allowed[0].line}`);
        allowedHitCounts.set(key, count);
      } else {
        unclassified.push({ path, line: i + 1, text: line.trim() });
      }
    }
  }
  if (unclassified.length > 0) {
    const details = unclassified
      .map((hit) => `  ${hit.path}:${hit.line}: ${hit.text}`)
      .join("\n");
    fail(`unclassified production gap markers found:\n${details}`);
  }

  const staleAllowed = allowedMarkers.filter((marker) =>
    (allowedHitCounts.get(`${marker.path}\n${marker.line}`) ?? 0) !== 1
  );
  if (staleAllowed.length > 0) {
    const details = staleAllowed
      .map((marker) => `  ${marker.path}: ${marker.line}`)
      .join("\n");
    fail(`allowed marker list is stale; expected exact line disappeared:\n${details}`);
  }

  return {
    classifiedCount: allowedMarkers.length,
    unclassifiedCount: unclassified.length,
  };
}

if (import.meta.main) {
  /** @type {{ path: string, text: string }[]} */
  const entries = [];
  for (const root of ROOTS) {
    for await (const path of walk(root)) {
      entries.push({ path, text: await Deno.readTextFile(repoUrl(path)) });
    }
  }
  const result = checkSourceEntries(entries);
  console.log(
    `[road-to-100-source-gap-scan] PASS (${result.classifiedCount} classified production markers, ` +
    `${result.unclassifiedCount} unclassified)`,
  );
}
