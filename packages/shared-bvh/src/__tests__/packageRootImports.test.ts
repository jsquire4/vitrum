import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

function readSource(relativeUrl: string): string {
  return readFileSync(fileURLToPath(new URL(relativeUrl, import.meta.url)), 'utf8');
}

function tsFiles(relativeUrl: string): string[] {
  const root = fileURLToPath(new URL(relativeUrl, import.meta.url));
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const full = `${dir}/${entry}`;
      const stat = statSync(full);
      if (stat.isDirectory()) {
        walk(full);
      } else if (entry.endsWith('.ts') || entry.endsWith('.tsx')) {
        out.push(full.replaceAll('\\', '/').replace(/\/+/g, '/'));
      }
    }
  };
  walk(root);
  return out;
}

function importedOrExportedSpecifiers(source: string): string[] {
  const specs: string[] = [];
  const re =
    /(?:^\s*import\s+(?:type\s+)?[\s\S]*?\s+from\s+['"]([^'"]+)['"]|^\s*import\s+['"]([^'"]+)['"]|^\s*export\s+(?:type\s+)?[\s\S]*?\s+from\s+['"]([^'"]+)['"])/gm;
  for (const match of source.matchAll(re)) {
    const spec = match[1] ?? match[2] ?? match[3];
    if (spec != null) specs.push(spec);
  }
  return specs;
}

describe('package root import boundaries', () => {
  it('keeps core and shared BVH roots free of the optional three-bindings bridge', () => {
    const roots = [
      ['@vitrum/core', '../../../core/src/index.ts'],
      ['@vitrum/shared-bvh', '../index.ts'],
    ] as const;

    for (const [pkg, relativeUrl] of roots) {
      const source = readSource(relativeUrl);
      expect(source, `${pkg} root must not import @vitrum/three-bindings`).not.toMatch(
        /@vitrum\/three-bindings|three-bindings/,
      );
    }
  });

  it('keeps shared-bvh root exports core-native', () => {
    const source = readSource('../index.ts');
    const specifiers = importedOrExportedSpecifiers(source);

    expect(specifiers).not.toContain('./bvhCommon.js');
    expect(specifiers).not.toContain('./legacy/three.js');
    expect(specifiers).not.toContain('three');
    expect(specifiers).not.toContain('three-mesh-bvh');
    expect(existsSync(fileURLToPath(new URL('../bvhCommon.ts', import.meta.url)))).toBe(false);
  });

  it('keeps SceneBvh root buffer fields host-neutral', () => {
    const source = readSource('../sceneBvh.ts');

    expect(source).toContain('sourceMaterials?: readonly unknown[]');
    expect(source).not.toContain('legacyThreeMaterials');
  });

  it('does not publish any legacy THREE ingestion subpath', () => {
    const pkg = JSON.parse(readSource('../../package.json')) as {
      exports?: Record<string, unknown>;
    };

    expect(pkg.exports?.['.']).toEqual({
      types: './src/index.ts',
      import: './src/index.ts',
    });
    expect(pkg.exports).not.toHaveProperty('./legacy/three');
    expect(pkg.exports).not.toHaveProperty('./bvhCommon');
    expect(pkg.exports).not.toHaveProperty('./sceneBvh');
  });

  it('has no direct three imports in source', () => {
    const offenders = tsFiles('..')
      .filter((path) => {
        const source = readFileSync(path, 'utf8');
        return /from ['"]three(?:\/webgpu)?['"]|import \* as THREE from ['"]three['"]/.test(source);
      });

    expect(offenders).toEqual([]);
  });
});
