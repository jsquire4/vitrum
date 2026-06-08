import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

function readEngineRoot(): string {
  return readFileSync(fileURLToPath(new URL('../index.ts', import.meta.url)), 'utf8');
}

function readEngineSource(name: string): string {
  return readFileSync(fileURLToPath(new URL(`../${name}`, import.meta.url)), 'utf8');
}

function readEnginePackageJson(): {
  dependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
} {
  return JSON.parse(readFileSync(fileURLToPath(new URL('../../package.json', import.meta.url)), 'utf8'));
}

describe('@vitrum/engine package root boundary', () => {
  it('does not directly re-export the optional three-bindings bridge', () => {
    expect(readEngineRoot()).not.toMatch(/@vitrum\/three-bindings|three-bindings/);
  });

  it('keeps legacy WebGL/THREE adapters off the hard dependency graph', () => {
    const pkg = readEnginePackageJson();
    expect(pkg.dependencies).not.toHaveProperty('@vitrum/pt-webgl');
    expect(pkg.dependencies).not.toHaveProperty('@vitrum/three-bindings');
    expect(pkg.optionalDependencies).toMatchObject({
      '@vitrum/pt-webgl': 'file:../pt-webgl',
      '@vitrum/three-bindings': 'file:../three-bindings',
    });
  });

  it('lazy-loads legacy WebGL/THREE paths without static type queries', () => {
    const createEngine = readEngineSource('createEngine.ts');
    const bridge = readEngineSource('threeSceneBridge.ts');

    expect(createEngine).not.toMatch(/^import\s+type[^\n]*@vitrum\/pt-webgl/m);
    expect(createEngine).not.toMatch(/typeof import\(['"]three['"]\)/);
    expect(createEngine).not.toMatch(/import\(['"]three['"]\)\.WebGLRenderer/);
    expect(bridge).not.toMatch(/typeof import\(['"]@vitrum\/three-bindings['"]\)/);
    expect(bridge).not.toMatch(/Parameters<[^>]*sceneFromThreeJS/);
  });
});
