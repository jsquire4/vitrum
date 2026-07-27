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
  runRoadSourceCheck,
} = await import(checkerUrl);
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
  await runRoadSourceCheck();
});
