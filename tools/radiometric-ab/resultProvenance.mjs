// @ts-check
// Shared provenance for committed radiometric A/B result snapshots.

export const RADIOMETRIC_RESULT_PROVENANCE_SCHEMA = "vitrum.radiometric-ab.result-provenance.v1";
export const RADIOMETRIC_RESULT_PROVENANCE_HELPER_PATH = "tools/radiometric-ab/resultProvenance.mjs";

/** @param {Uint8Array} bytes */
async function sha256Hex(bytes) {
  const owned = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(owned).set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", owned);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/** @param {URL | string} url */
async function sha256FileUrl(url) {
  return await sha256Hex(await Deno.readFile(url));
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
