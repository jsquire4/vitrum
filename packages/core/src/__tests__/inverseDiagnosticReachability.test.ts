/**
 * Every `InverseSessionDiagnostic.code` the core contract advertises must be
 * reachable — i.e. some shipping backend must actually construct a diagnostic
 * carrying that code. A code hosts can switch on but no backend can produce is
 * a false promise: it creates dead UI branches and dead compatibility rows that
 * silently rot.
 *
 * This is a source-scanning ledger gate, not a renderer test. The backend
 * packages own the assertions about WHEN each diagnostic fires; this file only
 * proves that a producing branch exists at all, in both directions:
 *   - no advertised code without an emit site;
 *   - no emitted code missing from the advertised union (which would not even
 *     type-check, but pins the union as the single source of truth).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, type Dirent } from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(fileURLToPath(new URL('../../../../', import.meta.url)));
const CONTRACT_FILE = 'packages/core/src/inverse.ts';

/** Backend packages permitted to emit inverse diagnostics. */
const BACKEND_SOURCE_ROOTS = [
  'packages/pt-webgpu/src',
  'packages/pt-webgl2/src',
  'packages/walkaround-hybrid/src',
  'packages/engine/src',
] as const;

function collectTsFiles(absDir: string, out: string[] = []): string[] {
  let entries: Dirent[];
  try {
    entries = readdirSync(absDir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    const full = join(absDir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === 'dist') continue;
      collectTsFiles(full, out);
      continue;
    }
    if (!entry.name.endsWith('.ts')) continue;
    // Tests may reference a code without any production branch producing it.
    if (full.includes('__tests__') || entry.name.endsWith('.test.ts')) continue;
    out.push(full);
  }
  return out;
}

/** The `code:` union members declared by the public contract, in file order. */
function advertisedCodes(): string[] {
  const text = readFileSync(resolve(REPO_ROOT, CONTRACT_FILE), 'utf8');
  const start = text.indexOf('export interface InverseSessionDiagnostic');
  expect(start, 'InverseSessionDiagnostic must exist in the contract file').toBeGreaterThan(-1);
  const body = text.slice(start, text.indexOf('}', start));
  const codeStart = body.indexOf('readonly code:');
  expect(codeStart, 'InverseSessionDiagnostic must declare a `code` union').toBeGreaterThan(-1);
  const union = body.slice(codeStart, body.indexOf(';', codeStart));
  return [...union.matchAll(/'([^']+)'/g)].map((match) => match[1]!);
}

/** Comments can NAME a code without any branch producing it, so they are not
 *  evidence of reachability and are stripped before scanning. */
function stripComments(text: string): string {
  return text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/** Codes constructed as a `'path-replay-*'` string literal in production
 *  backend sources. Literals appear both directly (`code: 'x'`) and via ternary
 *  chains that select a code (see pt-webgpu emissivePathReplayDomain), so the
 *  scan matches the literal itself rather than a single syntactic shape. */
function emittedCodes(): Set<string> {
  const emitted = new Set<string>();
  for (const root of BACKEND_SOURCE_ROOTS) {
    for (const file of collectTsFiles(resolve(REPO_ROOT, root))) {
      const text = stripComments(readFileSync(file, 'utf8'));
      for (const match of text.matchAll(/'(path-replay-[a-z-]+)'/g)) {
        emitted.add(match[1]!);
      }
    }
  }
  return emitted;
}

describe('InverseSessionDiagnostic code reachability', () => {
  it('advertises a non-trivial union parsed from the contract source', () => {
    const codes = advertisedCodes();
    expect(codes.length).toBeGreaterThan(5);
    expect(new Set(codes).size, 'union members must be unique').toBe(codes.length);
    expect(codes).toContain('path-replay-unsupported-field');
  });

  it('has a production emit site for every advertised diagnostic code', () => {
    const emitted = emittedCodes();
    const unreachable = advertisedCodes().filter((code) => !emitted.has(code));

    expect(
      unreachable,
      `InverseSessionDiagnostic.code advertises codes no backend can emit: ${unreachable.join(', ')}. ` +
      'Either emit them from the responsible backend branch or remove them from the public union.',
    ).toEqual([]);
  });

  it('advertises every code the backends actually emit', () => {
    const advertised = new Set(advertisedCodes());
    const undeclared = [...emittedCodes()].filter((code) => !advertised.has(code));

    expect(undeclared, `backends emit codes absent from the public union: ${undeclared.join(', ')}`).toEqual([]);
  });

  it('does not resurrect the four codes that had no emit branch', () => {
    // These were advertised while every path-replay narrowing for normal /
    // environment / lighting / light-selection controls actually resolved to
    // 'path-replay-unsupported-field'. Reintroduce one only with its emitter.
    const advertised = new Set(advertisedCodes());
    for (const code of [
      'path-replay-unsupported-normal',
      'path-replay-unsupported-environment',
      'path-replay-unsupported-lighting',
      'path-replay-unsupported-light-selection',
    ]) {
      expect(advertised.has(code), `${code} must not be advertised without an emit site`).toBe(false);
    }
  });
});
