#!/usr/bin/env node
/**
 * install-git-hooks.mjs — idempotently point git at the committed hooks dir
 * (scripts/githooks) so the warn-only pre-push GPU gate is active after
 * `npm install`. Run via the `prepare` npm lifecycle script.
 *
 * No-op (exit 0) outside a git work tree — tarball installs, CI checkouts
 * without .git, etc. — so it never breaks `npm ci` / publish.
 */
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));

try {
  execFileSync('git', ['rev-parse', '--is-inside-work-tree'], { cwd: root, stdio: 'ignore' });
} catch {
  process.exit(0); // not a git work tree
}

if (!existsSync(resolve(root, 'scripts/githooks/pre-push'))) process.exit(0);

try {
  execFileSync('git', ['config', 'core.hooksPath', 'scripts/githooks'], { cwd: root, stdio: 'ignore' });
  process.stderr.write(
    '[install-git-hooks] core.hooksPath → scripts/githooks (warn-only GPU pre-push gate active)\n',
  );
} catch {
  /* non-fatal — hooks just stay default */
}
