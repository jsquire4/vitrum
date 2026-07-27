// @ts-check
// Shared provenance for committed radiometric A/B result snapshots.

export const RADIOMETRIC_RESULT_PROVENANCE_SCHEMA = "vitrum.radiometric-ab.result-provenance.v1";
export const RADIOMETRIC_SOURCE_MANIFEST_SCHEMA = "vitrum.radiometric-ab.source-manifest.v1";
export const PT_RADIOMETRIC_STATUS_PROVENANCE_SCHEMA = "vitrum.pt-radiometric-ab.status-provenance.v1";
export const RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH = "tools/radiometric-ab/resultProvenance.mjs";
export const PT_RADIOMETRIC_AB_WRAPPER_PATH = "tools/radiometric-ab/run-pt-ab.mjs";
export const WALKAROUND_RESULT_PROVENANCE_SCHEMA = "vitrum.walkaround-ab.result-provenance.v1";
export const WALKAROUND_STATUS_PROVENANCE_SCHEMA = "vitrum.walkaround-ab.status-provenance.v1";
export const WALKAROUND_AB_WRAPPER_PATH = "tools/radiometric-ab/run-walkaround-ab.mjs";
export const WALKAROUND_AB_HARNESS_PATH = "tools/radiometric-ab/walkaround-ab.mjs";
export const PT_RADIOMETRIC_RUNTIME_SOURCE_ROOTS = Object.freeze([
  "packages/core/src",
  "packages/pt-webgpu/src",
  "packages/shared-bvh/src",
  "packages/shared-denoisers/src",
  "packages/shared-samplers/src",
]);

const RUNTIME_SOURCE_SUFFIXES = Object.freeze([
  ".ts",
  ".tsx",
  ".js",
  ".mjs",
  ".json",
  ".wgsl",
  ".glsl",
]);

/** @param {URL | string} url */
async function readFileBytes(url) {
  if (globalThis.Deno != null) return await globalThis.Deno.readFile(url);
  const { readFile } = await import("node:fs/promises");
  return new Uint8Array(await readFile(url));
}

/** @param {Uint8Array} bytes */
async function sha256Hex(bytes) {
  if (globalThis.crypto?.subtle != null) {
    const owned = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(owned).set(bytes);
    const digest = await globalThis.crypto.subtle.digest("SHA-256", owned);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }
  const { createHash } = await import("node:crypto");
  return createHash("sha256").update(bytes).digest("hex");
}

/** @param {URL | string} url */
async function sha256FileUrl(url) {
  return await sha256Hex(await readFileBytes(url));
}

/**
 * @param {string} repoRootImportMetaUrl
 * @param {string} path
 */
async function sha256RepoPath(repoRootImportMetaUrl, path) {
  return await sha256FileUrl(new URL(path, repoRootImportMetaUrl));
}

/** @param {string[]} sourceRoots */
function normalizedSourceRoots(sourceRoots) {
  if (!Array.isArray(sourceRoots) || sourceRoots.length === 0) {
    throw new TypeError("radiometric sourceRoots must contain at least one repository-relative directory");
  }
  const roots = [...new Set(sourceRoots.map((root) => {
    if (typeof root !== "string") throw new TypeError("radiometric source root must be a string");
    const normalized = root.replaceAll("\\", "/").replace(/^\.\//, "").replace(/\/+$/, "");
    if (
      normalized.length === 0 ||
      normalized.startsWith("/") ||
      normalized === ".." ||
      normalized.startsWith("../") ||
      normalized.includes("/../")
    ) {
      throw new TypeError(`invalid repository-relative radiometric source root: ${root}`);
    }
    return normalized;
  }))].sort();
  return roots;
}

/** @param {string} path */
function isRuntimeSourcePath(path) {
  return !path.includes("/__tests__/") &&
    !path.includes("/test/") &&
    !/\.(?:test|spec)\.[cm]?[jt]sx?$/.test(path) &&
    !path.endsWith(".d.ts") &&
    RUNTIME_SOURCE_SUFFIXES.some((suffix) => path.endsWith(suffix));
}

/** @param {URL} directoryUrl */
async function sortedDirectoryEntries(directoryUrl) {
  if (globalThis.Deno != null) {
    const entries = [];
    for await (const entry of globalThis.Deno.readDir(directoryUrl)) {
      entries.push({
        name: entry.name,
        isDirectory: entry.isDirectory,
        isFile: entry.isFile,
      });
    }
    return entries.sort((a, b) => a.name.localeCompare(b.name));
  }
  const { readdir } = await import("node:fs/promises");
  const entries = await readdir(directoryUrl, { withFileTypes: true });
  return entries
    .map((entry) => ({
      name: entry.name,
      isDirectory: entry.isDirectory(),
      isFile: entry.isFile(),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Hash the exact runtime source tree used by the pt-webgpu proof harnesses.
 * Tests and declarations are excluded because they are not loaded by the
 * renderer. Paths and digests are sorted before the aggregate digest is made,
 * so Node and Deno produce byte-identical manifests.
 *
 * @param {string} repoRootImportMetaUrl
 * @param {string[]} sourceRoots
 */
export async function buildRadiometricSourceManifest(repoRootImportMetaUrl, sourceRoots) {
  const roots = normalizedSourceRoots(sourceRoots);
  /** @type {string[]} */
  const paths = [];

  /** @param {string} relativeDirectory */
  async function walk(relativeDirectory) {
    const directoryUrl = new URL(`${relativeDirectory}/`, repoRootImportMetaUrl);
    const entries = await sortedDirectoryEntries(directoryUrl);
    for (const entry of entries) {
      const relativePath = `${relativeDirectory}/${entry.name}`;
      if (entry.isDirectory) await walk(relativePath);
      else if (entry.isFile && isRuntimeSourcePath(relativePath)) paths.push(relativePath);
    }
  }

  for (const root of roots) await walk(root);
  paths.sort();
  const files = await Promise.all(paths.map(async (path) => ({
    path,
    sha256: await sha256RepoPath(repoRootImportMetaUrl, path),
  })));
  const identity = {
    schema: RADIOMETRIC_SOURCE_MANIFEST_SCHEMA,
    roots,
    files,
  };
  return {
    ...identity,
    digestSha256: await sha256Hex(new TextEncoder().encode(JSON.stringify(identity))),
  };
}

/**
 * @param {string} scriptImportMetaUrl
 * @param {string} scriptPath
 * @param {string} resultPath
 * @param {{ repoRootImportMetaUrl: string, sourceRoots: string[] }} options
 */
export async function radiometricResultProvenance(
  scriptImportMetaUrl,
  scriptPath,
  resultPath,
  options,
) {
  if (options == null || typeof options.repoRootImportMetaUrl !== "string") {
    throw new TypeError("radiometric result provenance requires repoRootImportMetaUrl");
  }
  return {
    schema: RADIOMETRIC_RESULT_PROVENANCE_SCHEMA,
    scriptPath,
    resultPath,
    scriptSha256: await sha256FileUrl(new URL(scriptImportMetaUrl)),
    helperPath: RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH,
    helperSha256: await sha256FileUrl(new URL(import.meta.url)),
    sourceManifest: await buildRadiometricSourceManifest(
      options.repoRootImportMetaUrl,
      options.sourceRoots,
    ),
    generatedBy: "vitrum radiometric A/B harness",
  };
}

/** @param {string} repoRootImportMetaUrl */
async function ptRadiometricSourceIdentity(repoRootImportMetaUrl) {
  return {
    wrapperPath: PT_RADIOMETRIC_AB_WRAPPER_PATH,
    wrapperSha256: await sha256RepoPath(repoRootImportMetaUrl, PT_RADIOMETRIC_AB_WRAPPER_PATH),
    helperPath: RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH,
    helperSha256: await sha256FileUrl(new URL(import.meta.url)),
  };
}

/**
 * @param {string} repoRootImportMetaUrl
 * @param {string} statusPath
 * @param {string[]} resultPaths
 */
export async function ptRadiometricStatusProvenance(repoRootImportMetaUrl, statusPath, resultPaths) {
  return {
    schema: PT_RADIOMETRIC_STATUS_PROVENANCE_SCHEMA,
    ...(await ptRadiometricSourceIdentity(repoRootImportMetaUrl)),
    statusPath,
    resultPaths,
    resultSha256: await Promise.all(resultPaths.map(async (path) => ({
      path,
      sha256: await sha256RepoPath(repoRootImportMetaUrl, path),
    }))),
    generatedBy: "vitrum pt radiometric A/B wrapper",
  };
}

/** @param {string} repoRootImportMetaUrl */
async function walkaroundSourceIdentity(repoRootImportMetaUrl) {
  return {
    wrapperPath: WALKAROUND_AB_WRAPPER_PATH,
    wrapperSha256: await sha256RepoPath(repoRootImportMetaUrl, WALKAROUND_AB_WRAPPER_PATH),
    harnessPath: WALKAROUND_AB_HARNESS_PATH,
    harnessSha256: await sha256RepoPath(repoRootImportMetaUrl, WALKAROUND_AB_HARNESS_PATH),
    helperPath: RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH,
    helperSha256: await sha256FileUrl(new URL(import.meta.url)),
  };
}

/**
 * @param {string} repoRootImportMetaUrl
 * @param {string} resultPath
 */
export async function walkaroundResultProvenance(repoRootImportMetaUrl, resultPath) {
  return {
    schema: WALKAROUND_RESULT_PROVENANCE_SCHEMA,
    ...(await walkaroundSourceIdentity(repoRootImportMetaUrl)),
    resultPath,
    generatedBy: "vitrum walkaround radiometric A/B harness",
  };
}

/**
 * @param {string} repoRootImportMetaUrl
 * @param {string} statusPath
 * @param {string} resultPath
 */
export async function walkaroundStatusProvenance(repoRootImportMetaUrl, statusPath, resultPath) {
  return {
    schema: WALKAROUND_STATUS_PROVENANCE_SCHEMA,
    ...(await walkaroundSourceIdentity(repoRootImportMetaUrl)),
    statusPath,
    resultPath,
    generatedBy: "vitrum walkaround radiometric A/B wrapper",
  };
}
