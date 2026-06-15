import { readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

function readSource(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), 'utf8');
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

describe('@vitrum/walkaround-hybrid package boundary', () => {
  it('keeps optional THREE bridge APIs out of the package root', () => {
    const root = readSource('../index.ts');

    expect(root).not.toMatch(/applyDDGIShading|disposeApplyDDGIShadingCache|upgradeToNodeMaterial/);
    expect(root).not.toMatch(/WalkaroundBVHSceneRoot|WalkaroundDDGIScene|WalkaroundThreeHostScene/);
    expect(root).not.toMatch(/hostScene\/types/);
    expect(root).not.toMatch(/export \{ HybridEngine/);
  });

  it('keeps root-exported option types from importing three directly', () => {
    const options = readSource('../HybridEngineOptions.ts');

    expect(options).not.toMatch(/from ['"]three['"]/);
    expect(options).not.toMatch(/THREE\.Scene/);
    expect(options).not.toMatch(/threeScene|LegacyThreeSceneOption/);
  });

  it('does not expose a THREE bridge subpath', () => {
    const packageJson = JSON.parse(readSource('../../package.json')) as {
      exports?: Record<string, string>;
    };

    expect(packageJson.exports).not.toHaveProperty('./three');
  });

  it('has no direct three imports in source', () => {
    const offenders = tsFiles('..')
      .filter((path) => {
        const source = readFileSync(path, 'utf8');
        return /from ['"]three(?:\/webgpu|\/tsl)?['"]|import \* as THREE from ['"]three['"]/.test(source);
      });

    expect(offenders).toEqual([]);
  });

  it('keeps the README aligned with the raw WebGPU package boundary', () => {
    const readme = readSource('../../README.md');

    expect(readme).not.toMatch(/spatial sTree never splits|single global cell/);
    expect(readme).not.toMatch(/renderer\.backend\.device|StorageTexture|requires `?three\/webgpu/);
    expect(readme).toMatch(/splitOverflowLeaves/);
    expect(readme).toMatch(/has no direct\s+`three` \/ `three\/webgpu` dependency/);
  });
});
