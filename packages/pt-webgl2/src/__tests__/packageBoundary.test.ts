import { readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// The whole point of @vitrum/pt-webgl2 is to be THREE-free. This test fails the
// build if any source file imports `three` (or three-mesh-bvh, or the fork).
function tsFiles(relUrl: string): string[] {
  const root = fileURLToPath(new URL(relUrl, import.meta.url));
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const full = `${dir}/${entry}`;
      if (statSync(full).isDirectory()) walk(full);
      else if (entry.endsWith('.ts')) out.push(full);
    }
  };
  walk(root);
  return out;
}

describe('@vitrum/pt-webgl2 package boundary', () => {
  it('no source file imports three / three-mesh-bvh / the fork', () => {
    const offenders = tsFiles('..').filter((p) => {
      const src = readFileSync(p, 'utf8');
      return /from ['"]three(?:\/[^'"]*)?['"]|from ['"]three-mesh-bvh['"]|from ['"]three-gpu-pathtracer['"]|@vitrum\/pt-webgl['"]/.test(
        src,
      );
    });
    expect(offenders).toEqual([]);
  });

  it('declares every directly imported Vitrum package as a runtime dependency', () => {
    const pkg = JSON.parse(
      readFileSync(fileURLToPath(new URL('../../package.json', import.meta.url)), 'utf8'),
    ) as {
      dependencies?: Readonly<Record<string, string>>;
    };

    expect(pkg.dependencies).toMatchObject({
      '@vitrum/core': expect.any(String),
      '@vitrum/shared-bvh': expect.any(String),
      '@vitrum/shared-denoisers': expect.any(String),
      '@vitrum/shared-samplers': expect.any(String),
    });
  });
});
