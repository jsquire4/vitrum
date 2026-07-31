#!/usr/bin/env node
// @ts-check

import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { dirname, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '../..');
const NUL = Buffer.from([0]);

export const ROAD_ROOT_SOURCE_FILES = Object.freeze([
  'eslint.config.js',
  'package-lock.json',
  'package.json',
  'tsconfig.base.json',
  'tsconfig.json',
]);

export const ROAD_SOURCE_ROOTS = Object.freeze([
  'packages',
  'examples',
  'scripts',
  'tools',
]);

export const ROAD_SOURCE_EXTENSIONS = Object.freeze([
  '.cjs',
  '.css',
  '.glsl',
  '.html',
  '.js',
  '.json',
  '.lock',
  '.mjs',
  '.py',
  '.sh',
  '.ts',
  '.tsx',
  '.wasm',
  '.wgsl',
]);

export const ROAD_SOURCE_EXCLUDED_DIRECTORIES = Object.freeze([
  '.vite',
  '__pycache__',
  'build',
  'checkpoints',
  'coverage',
  'data_real',
  'data_smoke',
  'diagnostics',
  'dist',
  'node_modules',
  'reference-renders',
  'results',
]);

const SOURCE_EXTENSION_SET = new Set(ROAD_SOURCE_EXTENSIONS);
const EXCLUDED_DIRECTORY_SET = new Set(ROAD_SOURCE_EXCLUDED_DIRECTORIES);

/** @param {string} path */
export function isRoadSourceManifestPath(path) {
  const normalized = path.replaceAll('\\', '/');
  if (ROAD_ROOT_SOURCE_FILES.includes(normalized)) return true;
  const root = normalized.split('/', 1)[0];
  if (!ROAD_SOURCE_ROOTS.includes(root)) return false;
  if (
    normalized
      .split('/')
      .slice(0, -1)
      .some((part) => EXCLUDED_DIRECTORY_SET.has(part))
  ) {
    return false;
  }
  return SOURCE_EXTENSION_SET.has(extname(normalized));
}

/**
 * @param {string} root
 * @param {string} directory
 * @returns {Promise<string[]>}
 */
async function filesBelow(root, directory) {
  /** @type {string[]} */
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const absolute = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (EXCLUDED_DIRECTORY_SET.has(entry.name)) continue;
      files.push(...await filesBelow(root, absolute));
    } else if (entry.isFile()) {
      const path = relative(root, absolute).replaceAll('\\', '/');
      if (isRoadSourceManifestPath(path)) files.push(path);
    }
  }
  return files;
}

/**
 * Compute the canonical Road-to-100 remediation freeze:
 * repository-relative POSIX path, NUL, raw file content, NUL, in bytewise
 * path order, for executable source, tests, shaders, examples, proof tooling,
 * runtime WASM, dependency locks, and configuration below ROAD_SOURCE_ROOTS,
 * plus ROAD_ROOT_SOURCE_FILES. Generated builds, dependency caches, captured
 * evidence, training data, and other output-only directories are excluded
 * explicitly above.
 *
 * @param {string} [root]
 */
export async function computeRoadSourceManifest(root = REPO_ROOT) {
  const paths = [...ROAD_ROOT_SOURCE_FILES];
  for (const sourceRoot of ROAD_SOURCE_ROOTS) {
    paths.push(...await filesBelow(root, resolve(root, sourceRoot)));
  }
  paths.sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));

  const digest = createHash('sha256');
  for (const path of paths) {
    digest.update(path, 'utf8');
    digest.update(NUL);
    digest.update(await readFile(resolve(root, path)));
    digest.update(NUL);
  }
  return Object.freeze({
    algorithm: 'sha256(path-utf8 + NUL + raw-content + NUL)',
    files: paths.length,
    sha256: digest.digest('hex'),
  });
}

const invokedPath = process.argv[1] == null ? '' : resolve(process.argv[1]);
if (invokedPath === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await computeRoadSourceManifest(), null, 2));
}
