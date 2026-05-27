import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

/**
 * Merge PR-hybrid manifest entries (perf + PNG) under tools/reference-renders/PR-hybrid/.
 */
export async function readPrHybridManifest(refRoot) {
  const manifestPath = resolve(refRoot, 'manifest.json');
  try {
    const raw = JSON.parse(await readFile(manifestPath, 'utf8'));
    const list = Array.isArray(raw.manifest) ? raw.manifest : [];
    return { manifestPath, prior: raw, list };
  } catch {
    return { manifestPath, prior: {}, list: [] };
  }
}

export async function writePrHybridManifest(refRoot, { perfEntries = [], pngEntries = [], note }) {
  const { manifestPath, prior, list } = await readPrHybridManifest(refRoot);
  const keep = list.filter((e) => {
    if (e?.kind === 'perf' && perfEntries.length > 0) return false;
    if (e?.pngPath != null && pngEntries.length > 0) {
      const id = e.scenarioId;
      return !pngEntries.some((p) => p.scenarioId === id);
    }
    return true;
  });
  const body = {
    generatedAt: new Date().toISOString(),
    note:
      note ??
      'Perf: npm run benchmark:pr-hybrid-gpu. PNGs: npm run benchmark:pr-hybrid-refs (or pr-hybrid-gpu-full).',
    latestPerf: 'perf/latest.json',
    manifest: [...keep, ...perfEntries, ...pngEntries],
  };
  if (prior.schema) body.benchSchema = prior.schema;
  await writeFile(manifestPath, `${JSON.stringify(body, null, 2)}\n`);
  return manifestPath;
}
