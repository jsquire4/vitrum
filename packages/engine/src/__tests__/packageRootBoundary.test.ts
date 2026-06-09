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
  devDependencies?: Record<string, string>;
} {
  return JSON.parse(readFileSync(fileURLToPath(new URL('../../package.json', import.meta.url)), 'utf8'));
}

describe('@vitrum/engine package root boundary', () => {
  it('does not directly re-export the optional three-bindings bridge', () => {
    expect(readEngineRoot()).not.toMatch(/@vitrum\/three-bindings|three-bindings/);
  });

  it('depends on the native WebGL2 backend without legacy WebGL/THREE adapters', () => {
    const pkg = readEnginePackageJson();
    expect(pkg.dependencies).toMatchObject({
      '@vitrum/pt-webgl2': 'file:../pt-webgl2',
    });
    expect(pkg.dependencies).not.toHaveProperty('@vitrum/pt-webgl');
    expect(pkg.dependencies).not.toHaveProperty('@vitrum/three-bindings');
    expect(pkg.optionalDependencies ?? {}).not.toHaveProperty('@vitrum/pt-webgl');
    expect(pkg.optionalDependencies ?? {}).not.toHaveProperty('@vitrum/three-bindings');
    expect(pkg.peerDependencies ?? {}).not.toHaveProperty('three');
    expect(pkg.peerDependenciesMeta ?? {}).not.toHaveProperty('three');
    expect(pkg.devDependencies ?? {}).not.toHaveProperty('three');
    expect(pkg.devDependencies ?? {}).not.toHaveProperty('@types/three');
  });

  it('lazy-loads pt-webgl2 without retaining engine-level THREE bridge paths', () => {
    const createEngine = readEngineSource('createEngine.ts');

    expect(createEngine).toMatch(/import\(['"]@vitrum\/pt-webgl2['"]\)/);
    expect(createEngine).not.toMatch(/@vitrum\/pt-webgl(?!2)/);
    expect(createEngine).not.toMatch(/@vitrum\/three-bindings/);
    expect(createEngine).not.toMatch(/threeSceneBridge|ThreeSceneLike|sceneFromThreeSceneLike|isThreeScene/);
    expect(createEngine).not.toMatch(/typeof import\(['"]three['"]\)/);
    expect(createEngine).not.toMatch(/import\(['"]three['"]\)/);
  });
});
