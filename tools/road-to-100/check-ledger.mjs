#!/usr/bin/env node
// @ts-check

/**
 * Small, deterministic Road-to-100 source guard.
 *
 * This intentionally checks current production source and a finite set of active
 * documents. It does not pin prose snapshots, generated evidence, host results,
 * historical archives, or detector-test strings.
 */
import { access, readFile, readdir } from 'node:fs/promises';
import { constants } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeRoadSourceManifest } from './source-manifest.mjs';

export const PROVISIONAL_TERM_RE = /experimental/i;

export const REQUIRED_ACTIVE_FILES = Object.freeze([
  'plan/README.md',
  'plan/road-to-100.md',
  'plan/roadmap.md',
  'plan/renderer-fidelity-matrix.md',
]);

export const REMOVED_ROAD_FILES = Object.freeze([
  'plan/gap-closure-execution-plan.md',
  'plan/road-to-100-gap-ledger-2026-06-11.md',
  'plan/complexity-sweep-2026-07-20-plan.md',
  'plan/fidelity-promotion-playbook.md',
  'tools/road-to-100/check-validation-queue.mjs',
  'tools/road-to-100/validationQueueManifest.mjs',
  'tools/road-to-100/validation-queue.json',
  'tools/road-to-100/next-actions.mjs',
]);

export const ACTIVE_ROOT_DOCS = Object.freeze([
  'README.md',
  'AGENTS.md',
  'CLAUDE.md',
  'plan/README.md',
  'plan/road-to-100.md',
  'plan/roadmap.md',
  'plan/renderer-fidelity-matrix.md',
  'tools/radiometric-ab/README.md',
]);

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '../..');
const SOURCE_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.mjs']);

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[road-to-100-source-check] ${message}`);
}

/** @param {string} path */
async function exists(path) {
  try {
    await access(resolve(REPO_ROOT, path), constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

/** @param {string} path */
async function text(path) {
  return await readFile(resolve(REPO_ROOT, path), 'utf8');
}

/** @param {string} path */
export function isProductionSource(path) {
  const normalized = path.replaceAll('\\', '/');
  if (!normalized.startsWith('packages/') || !normalized.includes('/src/')) return false;
  if (normalized.includes('/__tests__/')) return false;
  if (/\.(test|spec)\.[cm]?[jt]sx?$/.test(normalized) || normalized.endsWith('.d.ts')) {
    return false;
  }
  return SOURCE_EXTENSIONS.has(extname(normalized));
}

/**
 * @param {string} dir
 * @returns {AsyncGenerator<string>}
 */
async function* walk(dir) {
  for (const entry of await readdir(resolve(REPO_ROOT, dir), { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name === 'coverage') {
        continue;
      }
      yield* walk(path);
    } else if (entry.isFile()) {
      yield path.replaceAll('\\', '/');
    }
  }
}

/** @param {string} source */
export function provisionalTerm(source) {
  const match = source.match(PROVISIONAL_TERM_RE);
  return match?.[0];
}

async function packageReadmes() {
  const paths = [];
  for (const entry of await readdir(resolve(REPO_ROOT, 'packages'), { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const path = `packages/${entry.name}/README.md`;
    if (await exists(path)) paths.push(path);
  }
  return paths.sort();
}

async function checkActiveFiles() {
  for (const path of REQUIRED_ACTIVE_FILES) {
    if (!(await exists(path))) fail(`required active file is missing: ${path}`);
  }
  for (const path of REMOVED_ROAD_FILES) {
    if (await exists(path)) fail(`superseded Road artifact still exists: ${path}`);
  }
}

async function checkPackageScripts() {
  const packageJson = JSON.parse(await text('package.json'));
  const scripts = packageJson.scripts ?? {};
  const sourceCheck = String(scripts['road-to-100-source-check'] ?? '');
  if (!sourceCheck.includes('tools/road-to-100/check-ledger.mjs')) {
    fail('package.json road-to-100-source-check does not run check-ledger.mjs');
  }
  if (!sourceCheck.includes('road-to-100-source-gap-scan')) {
    fail('package.json road-to-100-source-check does not run the source marker scan');
  }
  const expectedProofCheck = [
    'npm run road-to-100-source-check',
    'npm run radiometric-ab:source-check',
    'npm run radiometric-ab:restir-pt-specialty',
    'npm run renderer-fidelity-proof-check',
  ].join(' && ');
  if (scripts['proof-check'] !== expectedProofCheck) {
    fail('package.json proof-check must remain source/code-derived and exclude committed host-status snapshots');
  }
  for (const removed of ['road-to-100-validation-status', 'road-to-100-next-actions']) {
    if (Object.hasOwn(scripts, removed)) fail(`package.json still exposes removed script: ${removed}`);
  }
  const serialized = JSON.stringify(scripts);
  for (const path of REMOVED_ROAD_FILES) {
    if (serialized.includes(path)) fail(`package.json still references removed Road artifact: ${path}`);
  }
}

export const ROAD_ROW_STATUSES = Object.freeze(['Open', 'In flight', 'Closed']);
const ROAD_ROW_STATUS_SET = new Set(ROAD_ROW_STATUSES);

/** @param {string} road */
export function roadQueueRows(road) {
  const start = road.indexOf('## Current code-gap queue');
  if (start < 0) fail('plan/road-to-100.md is missing the current code-gap queue');
  const remainder = road.slice(start);
  const nextHeading = remainder.slice(1).search(/\r?\n##\s/);
  const section = nextHeading < 0 ? remainder : remainder.slice(0, nextHeading + 1);
  const rows = [];
  const ids = new Set();
  for (const line of section.split(/\r?\n/)) {
    const candidate = line.trim();
    if (!candidate.startsWith('|')) continue;
    if (/^\|\s*ID\s*\|/.test(candidate) || /^\|\s*:?-+/.test(candidate)) continue;
    if (!candidate.endsWith('|')) {
      fail(`malformed Road queue row (missing trailing pipe): ${candidate}`);
    }
    const fields = candidate.slice(1, -1).split('|').map((field) => field.trim());
    if (fields.length !== 4) {
      fail(`malformed Road queue row (expected four fields): ${candidate}`);
    }
    const [id, , , status] = fields;
    if (!/^[A-Z][A-Z0-9]*$/.test(id)) {
      fail(`malformed Road queue row identifier: ${id || '<blank>'}`);
    }
    if (ids.has(id)) fail(`duplicate Road queue row identifier: ${id}`);
    ids.add(id);
    if (!ROAD_ROW_STATUS_SET.has(status)) {
      fail(
        `Road queue row ${id} has unknown status ${JSON.stringify(status)}; ` +
        `expected one of ${ROAD_ROW_STATUSES.join(', ')}`,
      );
    }
    rows.push(Object.freeze({ id, status, line: candidate }));
  }
  if (rows.length === 0) fail('plan/road-to-100.md contains no queue rows');
  return Object.freeze(rows);
}

/** @param {string} road */
export function roadOpenRows(road) {
  return roadQueueRows(road)
    .filter((row) => row.status !== 'Closed')
    .map((row) => row.line);
}

async function checkRoadContract() {
  const road = await text('plan/road-to-100.md');
  for (const needle of [
    '**Authority:** current production source under `packages/*/src`',
    '## Current code-gap queue',
    '## Closed implementation programs',
    '`npm run build`',
    '## Reopen rule',
    'Do not create proof-only, host-only,',
    '**Final source manifest:**',
  ]) {
    if (!road.includes(needle)) fail(`plan/road-to-100.md is missing contract text: ${needle}`);
  }
  const openRows = roadOpenRows(road);
  const claimsNoOpenRows = road.includes('There are no open implementation rows.');
  if (openRows.length > 0 && claimsNoOpenRows) {
    fail('plan/road-to-100.md contradicts its open rows with the no-open sentinel');
  }
  if (openRows.length === 0 && !claimsNoOpenRows) {
    fail('plan/road-to-100.md has no open rows but is missing the no-open sentinel');
  }
  if (openRows.length > 0) {
    fail(`plan/road-to-100.md contains ${openRows.length} open implementation rows`);
  }

  const manifestMatch = road.match(
    /\*\*Final source manifest:\*\*\s+([\d,]+) files;\s+SHA-256\s+`([a-f0-9]{64})`;/,
  );
  if (manifestMatch == null) {
    fail('plan/road-to-100.md has no parseable final source manifest');
  }
  const declaredFiles = Number(manifestMatch[1].replaceAll(',', ''));
  const declaredSha256 = manifestMatch[2];
  const actual = await computeRoadSourceManifest(REPO_ROOT);
  if (declaredFiles !== actual.files || declaredSha256 !== actual.sha256) {
    fail(
      'plan/road-to-100.md final source manifest is stale: ' +
      `declared ${declaredFiles} files/${declaredSha256}, ` +
      `actual ${actual.files} files/${actual.sha256}`,
    );
  }
}

async function checkActiveDocs() {
  const paths = [...ACTIVE_ROOT_DOCS, ...(await packageReadmes())];
  for (const path of paths) {
    const source = await text(path);
    const provisional = provisionalTerm(source);
    if (provisional != null) {
      fail(`${path} uses removed provisional maturity terminology: ${provisional}`);
    }
    for (const removed of REMOVED_ROAD_FILES) {
      if (source.includes(removed)) fail(`${path} references removed Road artifact: ${removed}`);
    }
  }
}

async function checkProductionSource() {
  const violations = [];
  for await (const path of walk('packages')) {
    if (!isProductionSource(path)) continue;
    const source = await text(path);
    const provisional = provisionalTerm(source);
    if (provisional != null) violations.push(`${path}: ${provisional}`);
  }
  if (violations.length > 0) {
    fail(`production source contains removed provisional identifiers:\n  ${violations.join('\n  ')}`);
  }
}

export async function runRoadSourceCheck() {
  await checkActiveFiles();
  await checkPackageScripts();
  await checkRoadContract();
  await checkActiveDocs();
  await checkProductionSource();
}

const invokedPath = process.argv[1] == null ? '' : resolve(process.argv[1]);
if (invokedPath === fileURLToPath(import.meta.url)) {
  await runRoadSourceCheck();
  const packageCount = (await packageReadmes()).length;
  console.log(
    `[road-to-100-source-check] PASS (${REQUIRED_ACTIVE_FILES.length} active plans, ` +
    `${packageCount} package READMEs, production source clean)`,
  );
}
