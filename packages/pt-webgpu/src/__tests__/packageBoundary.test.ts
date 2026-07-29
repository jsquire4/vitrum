import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

describe('@vitrum/pt-webgpu package boundary', () => {
  it('exports only the package root and the narrow support-profile subpath', () => {
    const pkg = JSON.parse(
      readFileSync(fileURLToPath(new URL('../../package.json', import.meta.url)), 'utf8'),
    ) as {
      exports?: Readonly<Record<string, unknown>>;
    };

    expect(pkg.exports).toEqual({
      '.': './src/index.ts',
      './support-profile': './src/supportProfile.ts',
    });
    expect(pkg.exports).not.toHaveProperty('./src/*');
  });
});
