import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const checkerUrl = pathToFileURL(resolve(repoRoot, 'tools', 'road-to-100', 'check-ledger.mjs')).href;
const gapCheckerPath = resolve(repoRoot, 'tools', 'road-to-100', 'check-source-gap-markers.mjs');
const gapCheckerUrl = pathToFileURL(gapCheckerPath).href;
const packageJsonPath = resolve(repoRoot, 'package.json');
const {
  isProductionSource,
  provisionalTerm,
  REMOVED_ROAD_FILES,
  REQUIRED_ACTIVE_FILES,
  roadOpenRows,
  roadQueueRows,
  runRoadSourceCheck,
} = await import(checkerUrl);
const manifestUrl = pathToFileURL(
  resolve(repoRoot, 'tools', 'road-to-100', 'source-manifest.mjs'),
).href;
const { isRoadSourceManifestPath } = await import(manifestUrl);
const { ALLOWED_MARKERS, checkSourceEntries } = await import(gapCheckerUrl);

test('Road source classifier covers production code without scanning test fixtures', () => {
  assert.equal(isProductionSource('packages/core/src/scene/validation.ts'), true);
  assert.equal(isProductionSource('packages/pt-webgl2/src/glsl/materialStride.js'), true);
  assert.equal(isProductionSource('packages/example/src/runtime.mjs'), true);
  assert.equal(isProductionSource('packages/example/src/runtime.cjs'), false);
  assert.equal(isProductionSource('packages/core/src/scene/validation.test.ts'), false);
  assert.equal(isProductionSource('packages/core/src/__tests__/validation.ts'), false);
  assert.equal(isProductionSource('packages/core/src/index.d.ts'), false);
  assert.equal(isProductionSource('tools/road-to-100/check-ledger.mjs'), false);
});

test('Road source checker recognizes removed provisional terminology', () => {
  const removedWord = ['experi', 'mental'].join('');
  assert.equal(provisionalTerm(`mode: ${removedWord}`), removedWord);
  assert.equal(provisionalTerm(`${removedWord}_mode`), removedWord);
  assert.equal(provisionalTerm(`${removedWord}Feature`), removedWord);
  assert.equal(provisionalTerm('supported implementation'), undefined);
});

test('Road gap-marker scan covers ordinary JavaScript production source', async () => {
  const source = await readFile(gapCheckerPath, 'utf8');
  assert.match(source, /SOURCE_EXTENSIONS = \["\.ts", "\.tsx", "\.js", "\.mjs"\]/);
  assert.doesNotMatch(source, /"\.glsl\.js"|"\.wgsl\.ts"/);
});

test('Road gap-marker scan covers unfinished-implementation language and exact allowlisting', async () => {
  const source = await readFile(gapCheckerPath, 'utf8');
  for (const marker of [
    'TBD',
    'unimplemented',
    'partial[ -]implementation',
    'incomplete[ -]implementation',
    'unfinished[ -]implementation',
  ]) {
    assert.ok(source.includes(marker), `missing gap marker pattern: ${marker}`);
  }
  assert.match(source, /duplicate allowed marker/);
  assert.match(source, /ambiguous allowed marker match/);
  assert.match(source, /allowed marker matched more than once/);
  assert.match(source, /nonempty path, exact trimmed full line, and reason/);
  assert.match(source, /line\.trim\(\) === marker\.line/);
});

test('Road gap-marker allowlist rejects extra marker text appended to an allowed line', () => {
  const allowed = ALLOWED_MARKERS[0];
  assert.doesNotThrow(() =>
    checkSourceEntries([{ path: allowed.path, text: allowed.line }], [allowed]),
  );
  assert.throws(
    () => checkSourceEntries(
      [{ path: allowed.path, text: `${allowed.line} TODO unimplemented` }],
      [allowed],
    ),
    /unclassified production gap markers found/,
  );
});

test('Road final manifest covers implementation and verification roots but excludes outputs', () => {
  for (const path of [
    'package.json',
    'packages/core/src/scene/patchScene.ts',
    'packages/engine/__tests__/attachVitrumLoop.test.ts',
    'packages/gltf-adapter/src/assets/draco_decoder.wasm',
    'examples/shared/camera.ts',
    'examples/attach-vitrum/src/main.ts',
    'scripts/__tests__/road-to-100-source-check.test.mjs',
    'tools/behavioral-gate/gate.mjs',
    'tools/behavioral-gate/deno.lock',
    'tools/road-to-100/source-manifest.mjs',
    'eslint.config.js',
    'package-lock.json',
    'tsconfig.base.json',
  ]) {
    assert.equal(isRoadSourceManifestPath(path), true, `manifest omitted ${path}`);
  }
  for (const path of [
    'plan/road-to-100.md',
    'examples/attach-vitrum/dist/index.js',
    'examples/gltf-viewer/node_modules/dependency/index.js',
    'tools/benchmark-runner/results/result.json',
    'tools/reference-renders/baseline/frame.png',
  ]) {
    assert.equal(isRoadSourceManifestPath(path), false, `manifest included ${path}`);
  }
});

test('Road queue parser fails closed on every non-final status', () => {
  const valid = [
    '## Current code-gap queue',
    '',
    '| ID | Current-source gap | Primary source | Status |',
    '| --- | --- | --- | --- |',
    '| C0 | Gap | `source.ts` | Closed |',
    '',
    '## Closed implementation programs',
  ].join('\n');
  assert.equal(roadQueueRows(valid).length, 1);
  assert.equal(roadOpenRows(valid).length, 0);
  assert.equal(roadOpenRows(valid.replace('Closed', 'Open')).length, 1);
  assert.equal(roadOpenRows(valid.replace('Closed', 'In flight')).length, 1);
  for (const status of ['Pending', 'Blocked', 'Reopened', '', 'closed']) {
    assert.throws(
      () => roadQueueRows(valid.replace('Closed', status)),
      /unknown status/,
      `status ${JSON.stringify(status)} must fail closed`,
    );
  }
  assert.throws(
    () => roadQueueRows(valid.replace('| Closed |', '| Closed')),
    /missing trailing pipe/,
  );
  const indentedOpen = valid.replace(
    '| C0 | Gap | `source.ts` | Closed |',
    [
      '| C0 | Gap | `source.ts` | Closed |',
      '  | X1 | Indented gap | `source.ts` | Open |',
    ].join('\n'),
  );
  assert.equal(roadOpenRows(indentedOpen).length, 1);
});

test('canonical proof-check excludes committed host-status snapshot gates', async () => {
  const packageJson = JSON.parse(await readFile(packageJsonPath, 'utf8'));
  assert.equal(
    packageJson.scripts['proof-check'],
    [
      'npm run road-to-100-source-check',
      'npm run radiometric-ab:source-check',
      'npm run radiometric-ab:restir-pt-specialty',
      'npm run renderer-fidelity-proof-check',
    ].join(' && '),
  );
  assert.match(packageJson.scripts['radiometric-ab:source-check'], /--source-only/);
});

test('Road source contract converges on the current repository', async () => {
  assert.ok(REQUIRED_ACTIVE_FILES.length > 0);
  assert.ok(REMOVED_ROAD_FILES.length > 0);
  const road = await readFile(resolve(repoRoot, 'plan', 'road-to-100.md'), 'utf8');
  assert.equal(roadOpenRows(road).length, 0);
  await runRoadSourceCheck();
});
