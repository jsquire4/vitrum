import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
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
    expect(options).toMatch(/readonly threeScene\?: LegacyThreeSceneOption/);
  });

  it('exposes optional THREE bridge APIs through the ./three subpath', () => {
    const threeSubpath = readSource('../three.ts');
    const packageJson = JSON.parse(readSource('../../package.json')) as {
      exports?: Record<string, string>;
    };

    expect(packageJson.exports?.['./three']).toBe('./src/three.ts');
    expect(threeSubpath).toMatch(/applyDDGIShading/);
    expect(threeSubpath).toMatch(/disposeApplyDDGIShadingCache/);
    expect(threeSubpath).toMatch(/upgradeToNodeMaterial/);
    expect(threeSubpath).toMatch(/WalkaroundBVHSceneRoot|WalkaroundDDGIScene|WalkaroundThreeHostScene/);
    expect(threeSubpath).toMatch(/HybridEngineThreeOptions/);
    expect(threeSubpath).toMatch(/export \{ HybridEngine, createWalkaroundEngine_Hybrid \}/);
  });

  it('quarantines direct three imports to bridge and legacy adapter folders', () => {
    const offenders = tsFiles('..')
      .filter((path) => !path.includes('/src/three/'))
      .filter((path) => !path.includes('/src/legacy/three/'))
      .filter((path) => !path.includes('/src/__tests__/'))
      .filter((path) => !path.includes('/src/ddgi/__tests__/'))
      .filter((path) => !path.includes('/src/rc/__tests__/'))
      .filter((path) => !path.includes('/src/restir/__tests__/'))
      .filter((path) => {
        const source = readFileSync(path, 'utf8');
        return /from ['"]three(?:\/webgpu|\/tsl)?['"]|import \* as THREE from ['"]three['"]/.test(source);
      });

    expect(offenders).toEqual([]);
  });

  it('does not keep legacy THREE compatibility shims in runtime folders', () => {
    for (const path of [
      '../restir/bvhCompute.ts',
      '../restir/sceneBvhFromCore.ts',
      '../rc/bvhCompute.ts',
    ]) {
      expect(existsSync(fileURLToPath(new URL(path, import.meta.url))), path).toBe(false);
    }
  });
});
