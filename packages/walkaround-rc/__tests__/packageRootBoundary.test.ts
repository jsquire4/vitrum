import { readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

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

describe('@vitrum/walkaround-rc package boundary', () => {
  it('keeps the raw RC root free of direct three imports', () => {
    const root = readFileSync(fileURLToPath(new URL('../src/index.ts', import.meta.url)), 'utf8');
    expect(root).not.toMatch(/from ['"]three(?:\/webgpu|\/tsl)?['"]|import \* as THREE/);
    expect(root).not.toMatch(/THREE|TSL|\/three/);
  });

  it('has no direct three imports in source', () => {
    const offenders = tsFiles('../src')
      .filter((path) => {
        const source = readFileSync(path, 'utf8');
        return /from ['"]three(?:\/webgpu|\/tsl)?['"]|import \* as THREE from ['"]three['"]/.test(source);
      });

    expect(offenders).toEqual([]);
  });

  it('does not declare three-mesh-bvh for the raw RC package', () => {
    const pkg = JSON.parse(readFileSync(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8')) as {
      peerDependencies?: Record<string, string>;
      peerDependenciesMeta?: Record<string, unknown>;
      devDependencies?: Record<string, string>;
    };

    expect(pkg.peerDependencies ?? {}).not.toHaveProperty('three');
    expect(pkg.peerDependencies ?? {}).not.toHaveProperty('three-mesh-bvh');
    expect(pkg.peerDependenciesMeta ?? {}).not.toHaveProperty('three');
    expect(pkg.peerDependenciesMeta ?? {}).not.toHaveProperty('three-mesh-bvh');
    expect(pkg.devDependencies ?? {}).not.toHaveProperty('three');
    expect(pkg.devDependencies ?? {}).not.toHaveProperty('@types/three');
    expect(pkg.devDependencies).not.toHaveProperty('three-mesh-bvh');
  });

  it('keeps public docs aligned with the raw RC surface', () => {
    const packageReadme = readFileSync(fileURLToPath(new URL('../README.md', import.meta.url)), 'utf8');
    const cascadePyramid = readFileSync(fileURLToPath(new URL('../src/cascadePyramid.ts', import.meta.url)), 'utf8');
    const rootReadme = readFileSync(fileURLToPath(new URL('../../../README.md', import.meta.url)), 'utf8');

    expect(packageReadme).not.toMatch(/GIReceiver|buildWalkaroundLightingNode|TSL-side receiver helpers/);
    expect(cascadePyramid).not.toMatch(/GIReceiver|buildWalkaroundLightingNode|TSL-side receiver helpers/);
    expect(rootReadme).not.toMatch(/@vitrum\/walkaround-rc\s+.*receiver/);
    expect(packageReadme).toMatch(/old TSL receiver wrappers are\s+not shipped/);
  });
});
