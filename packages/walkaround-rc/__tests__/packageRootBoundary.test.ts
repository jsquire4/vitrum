import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import * as rcRoot from '../src/index.js';

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

  it('keeps CPU-only validation helpers out of the production root and src tree', () => {
    const runtimeRoot = rcRoot as unknown as Record<string, unknown>;
    const rootSource = readFileSync(
      fileURLToPath(new URL('../src/index.ts', import.meta.url)),
      'utf8',
    );
    for (const removed of [
      'allocateCascades',
      'disposeCascades',
      'computeOctahedralSolidAngles',
      'MAX_OCTAHEDRAL_SOLID_ANGLE_GRID_SIZE',
    ]) {
      expect(runtimeRoot).not.toHaveProperty(removed);
      expect(rootSource).not.toMatch(new RegExp(`\\b${removed}\\b`));
    }
    for (const removedType of ['CascadeBuffers', 'CascadeAABB']) {
      expect(rootSource).not.toMatch(new RegExp(`\\b${removedType}\\b`));
    }

    expect(
      existsSync(fileURLToPath(new URL('../src/cascadeBuffers.ts', import.meta.url))),
    ).toBe(false);
    expect(
      existsSync(fileURLToPath(new URL('../src/octahedralSolidAngles.ts', import.meta.url))),
    ).toBe(false);
    expect(
      existsSync(fileURLToPath(new URL('./support/cascadeBuffers.ts', import.meta.url))),
    ).toBe(true);
    expect(
      existsSync(fileURLToPath(new URL('./support/octahedralSolidAngles.ts', import.meta.url))),
    ).toBe(true);
  });

  it('retains only production-wired runtime entry points plus derived cascade metadata', () => {
    expect(rcRoot.CASCADE_COUNT).toBe(rcRoot.CASCADE_DIMS.length);

    const hybridRc = readFileSync(
      fileURLToPath(new URL('../../walkaround-hybrid/src/HybridEngineRC.ts', import.meta.url)),
      'utf8',
    );
    const hybridConfig = readFileSync(
      fileURLToPath(new URL('../../walkaround-hybrid/src/HybridEngineConfig.ts', import.meta.url)),
      'utf8',
    );
    const cascadeSampler = readFileSync(
      fileURLToPath(
        new URL(
          '../../walkaround-hybrid/src/shaders/sampleCascadeC0.wgsl.ts',
          import.meta.url,
        ),
      ),
      'utf8',
    );
    const rcDispatcher = readFileSync(
      fileURLToPath(new URL('../src/cascadeDispatch.ts', import.meta.url)),
      'utf8',
    );
    const cascadeMerge = readFileSync(
      fileURLToPath(new URL('../src/wgsl/cascadeMerge.wgsl.ts', import.meta.url)),
      'utf8',
    );

    expect(hybridRc).toMatch(
      /import\s*\{[^}]*RCDispatcher[^}]*CASCADE_DIMS[^}]*validateCascadeDims[^}]*\}\s*from '@vitrum\/walkaround-rc'/s,
    );
    expect(hybridConfig).toMatch(
      /import\s*\{[^}]*RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET[^}]*RC_MAX_TRANSMITTED_INTERFACE_BUDGET[^}]*RC_MIN_TRANSMITTED_INTERFACE_BUDGET[^}]*validateCascadeDims[^}]*\}\s*from '@vitrum\/walkaround-rc'/s,
    );
    expect(cascadeSampler).toMatch(
      /import\s*\{\s*RC_OCTAHEDRAL_STRATIFIED_SAMPLING_WGSL\s*\}\s*from '@vitrum\/walkaround-rc'/,
    );
    expect(rcDispatcher).toMatch(
      /import\s*\{\s*PROBE_RAY_CAST_WGSL\s*\}\s*from '.\/wgsl\/probeRayCast\.wgsl\.js'/,
    );
    expect(rcDispatcher).toMatch(
      /import\s*\{\s*CASCADE_MERGE_WGSL\s*\}\s*from '.\/wgsl\/cascadeMerge\.wgsl\.js'/,
    );
    expect(cascadeMerge).toMatch(
      /import\s*\{\s*RC_OCTAHEDRAL_SOLID_ANGLE_WGSL\s*\}\s*from '.\/octahedralSampling\.wgsl\.js'/,
    );
  });
});
