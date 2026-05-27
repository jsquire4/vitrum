import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Repo root for benchmark-runner spawns (honours VITRUM_REPO_ROOT from Windows pushd wrapper). */
export function getRepoRoot(importMetaUrl) {
  const fromEnv = process.env.VITRUM_REPO_ROOT?.trim();
  if (fromEnv != null && fromEnv.length > 0) {
    return fromEnv;
  }
  const here = dirname(fileURLToPath(importMetaUrl));
  return resolve(here, '..', '..');
}
