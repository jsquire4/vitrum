// @ts-check
// Shared provenance for committed radiometric A/B result snapshots.

export const RADIOMETRIC_RESULT_PROVENANCE_SCHEMA = "vitrum.radiometric-ab.result-provenance.v1";
export const RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH = "tools/radiometric-ab/resultProvenance.mjs";
export const WALKAROUND_RESULT_PROVENANCE_SCHEMA = "vitrum.walkaround-ab.result-provenance.v1";
export const WALKAROUND_STATUS_PROVENANCE_SCHEMA = "vitrum.walkaround-ab.status-provenance.v1";
export const WALKAROUND_PROMOTION_PROVENANCE_SCHEMA = "vitrum.walkaround-ab.promotion-provenance.v1";
export const WALKAROUND_AB_WRAPPER_PATH = "tools/radiometric-ab/run-walkaround-ab.mjs";
export const WALKAROUND_AB_HARNESS_PATH = "tools/radiometric-ab/walkaround-ab.mjs";

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

/**
 * @param {string} scriptImportMetaUrl
 * @param {string} scriptPath
 * @param {string} resultPath
 */
export async function radiometricResultProvenance(scriptImportMetaUrl, scriptPath, resultPath) {
  return {
    schema: RADIOMETRIC_RESULT_PROVENANCE_SCHEMA,
    scriptPath,
    resultPath,
    scriptSha256: await sha256FileUrl(new URL(scriptImportMetaUrl)),
    helperPath: RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH,
    helperSha256: await sha256FileUrl(new URL(import.meta.url)),
    generatedBy: "vitrum radiometric A/B harness",
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

/**
 * @param {string} repoRootImportMetaUrl
 * @param {string} statusPath
 * @param {string[]} sourceStatusPaths
 * @param {string[]} sourceResultPaths
 */
export async function walkaroundPromotionProvenance(
  repoRootImportMetaUrl,
  statusPath,
  sourceStatusPaths,
  sourceResultPaths,
) {
  return {
    schema: WALKAROUND_PROMOTION_PROVENANCE_SCHEMA,
    ...(await walkaroundSourceIdentity(repoRootImportMetaUrl)),
    statusPath,
    sourceStatusPaths,
    sourceStatusSha256: await Promise.all(sourceStatusPaths.map(async (path) => ({
      path,
      sha256: await sha256RepoPath(repoRootImportMetaUrl, path),
    }))),
    sourceResultPaths,
    sourceResultSha256: await Promise.all(sourceResultPaths.map(async (path) => ({
      path,
      sha256: await sha256RepoPath(repoRootImportMetaUrl, path),
    }))),
    generatedBy: "vitrum walkaround radiometric A/B promotion wrapper",
  };
}
