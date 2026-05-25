# npm publish-readiness audit — 2026-05-17

> Read-only audit of every workspace `package.json` plus root, enumerating every blocker that exists today before vitrum can ship `@vitrum/*` to npm. No changes were made to any `package.json` or source file. The project remains pre-alpha and private per [`CLAUDE.md`](../CLAUDE.md) and [`RELEASING.md`](../RELEASING.md).

## Status update — 2026-05-18

Several blockers identified below have since been resolved. Verified-current state:

- **README/LICENSE** (blocker #5) — **fully resolved**. All 12 packages (10 audited + `@vitrum/scene-lighting` + `@vitrum/walkaround-rc`, both added afterward) now have both a `README.md` and a `LICENSE` file. The follow-up fixes landed via `chore/missing-package-readmes` (`ec58f48`), `chore/per-package-license-files-20260517` (`cc5bc50`), the W13 README-audit pass (`fc882f6` + `0ab2d54`), and the W8 walkaround-rc extraction (`b065676` + this commit).
- **scene-lighting package metadata** (added post-audit) — brought to workspace parity 2026-05-18 (commit `63f207f` added keywords/author/license/files/repository/lint script).
- **walkaround-rc package metadata** (extracted post-audit) — README.md + LICENSE in place; package.json mirrors the walkaround-hybrid shape.
- Blockers #1 (no build step), #2 (private), #3 (`file:` fork dep), #4 (`file:` intra-workspace deps), and the per-package `homepage`/`bugs` gap remain open — they are intentional pre-alpha state per `RELEASING.md`. The per-package rows below are accurate for those fields but stale wrt README/LICENSE and predate the `scene-lighting` + `walkaround-rc` splits.

## Scope

Audited packages (10): root + 10 workspace packages under `packages/*`.

| # | Package | Path |
|---|---|---|
| 1 | (root workspace) | `package.json` |
| 2 | `@vitrum/core` | `packages/core` |
| 3 | `@vitrum/dev` | `packages/dev` |
| 4 | `@vitrum/engine` | `packages/engine` |
| 5 | `@vitrum/pt-webgl` | `packages/pt-webgl` |
| 6 | `@vitrum/pt-webgpu` | `packages/pt-webgpu` |
| 7 | `@vitrum/shared-bvh` | `packages/shared-bvh` |
| 8 | `@vitrum/shared-denoisers` | `packages/shared-denoisers` |
| 9 | `@vitrum/shared-samplers` | `packages/shared-samplers` |
| 10 | `@vitrum/three-bindings` | `packages/three-bindings` |
| 11 | `@vitrum/walkaround-hybrid` | `packages/walkaround-hybrid` |

Total publishable-candidate packages: **10** (root is workspace-only, will stay `private: true` forever).

`examples/*` and `tools/*` are also workspace members and out of scope for npm publish — they ship inside the repo, not to the registry.

---

## TL;DR — top blockers

1. **No package has a real build.** Every package's `main`/`types`/`exports` point at `./src/*.ts`. `dist/` directories exist but contain only `tsconfig.tsbuildinfo` — no `.js` and no `.d.ts`. Every package-level `tsconfig.json` (except `core`) sets `"noEmit": true`. There is no `build` script in any package. Until a build step lands, npm consumers cannot `import` from `@vitrum/*` (TypeScript-from-source assumes the consumer has a TS transpile step *and* the same `tsconfig` options — both unsafe to assume).
2. **Every package is `private: true`.** Intentional today (safety belt — `RELEASING.md` calls this out explicitly), but enumerated here as a publish-time flag that must be flipped per package.
3. **`@vitrum/pt-webgl` depends on `three-gpu-pathtracer` via `file:../three-gpu-pathtracer`** — the sibling-repo blocker is closed. Publish still needs a deliberate package identity/version for the absorbed renderer, or `pt-webgl` stays private.
4. **All intra-workspace deps use `file:../X` specifiers.** Every cross-package dep (`@vitrum/core`, `@vitrum/shared-*`, `@vitrum/three-bindings`, `@vitrum/walkaround-hybrid`, `@vitrum/pt-webgl`) must be rewritten to a real semver specifier (e.g. `^0.1.0-alpha.1`) before publish, in lockstep with bumping every package's `version` off `0.0.0`.
5. **Most packages have no README.md and no per-package LICENSE.** Only `pt-webgl`, `pt-webgpu`, and `walkaround-hybrid` have READMEs; none of the 10 packages have a `LICENSE` file (the repo root has one, but `npm publish` ships the package directory's tarball — root files are not included).

The rest of this document expands each row, then groups cross-cutting issues, then proposes a cheapest-wins-first fix order with effort sizing.

---

## Per-package readiness table

Columns:

- **Identity** — does it have name/version/description/keywords/author/license/repository? Missing = `homepage`/`bugs` not declared anywhere.
- **Entrypoint** — does `main`/`types`/`exports` resolve to something publishable?
- **Build** — does the package produce a `dist/` consumers can use?
- **`private`** — current safety-belt state. `T` = `private:true`.
- **Files** — `files` field correctness.
- **README/LICENSE** — present in package dir? (root files don't ship in package tarballs.)
- **Blockers (B = blocker count this package)**

| # | Package | Identity | Entrypoint | Build | private | files field | README/LICENSE | Blockers |
|---|---|---|---|---|---|---|---|---|
| 1 | (root) | OK + missing `homepage`/`bugs` | n/a (`private:true`) | n/a | T | n/a | repo-root only | 0 (intentionally private) |
| 2 | `@vitrum/core` | OK; missing `homepage`/`bugs` | points at `./src/*.ts` | no emit; only `tsbuildinfo` | T | ships `src` + `README.md` but **no README.md exists** | none in pkg | 4 |
| 3 | `@vitrum/dev` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build script; `noEmit:true` | T | ships `src` + `README.md` but **no README.md exists** | none | 5 |
| 4 | `@vitrum/engine` | OK; missing `homepage`/`bugs` | `./src/*.ts` with 3 subpath exports | no build; `noEmit:true` | T | ships `src` + `README.md` but **no README.md exists** | none | 5 |
| 5 | `@vitrum/pt-webgl` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` | README only | **6** (fork file: dep) |
| 6 | `@vitrum/pt-webgpu` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` | README only | 5 |
| 7 | `@vitrum/shared-bvh` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` but **no README.md** | none | 5 |
| 8 | `@vitrum/shared-denoisers` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` but **no README.md** | none | 5 |
| 9 | `@vitrum/shared-samplers` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` but **no README.md** | none | 5 |
| 10 | `@vitrum/three-bindings` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` but **no README.md** | none | 5 |
| 11 | `@vitrum/walkaround-hybrid` | OK; missing `homepage`/`bugs` | `./src/*.ts` | no build; `noEmit:true` | T | ships `src` + `README.md` | README only | 5 |

(Numbers in **Blockers** are the count of distinct flagged issues in the per-package detail sections below; they are *not* a severity score.)

---

## Per-package detail

### 1. root `package.json`

```
name=vitrum, version=0.0.0, private=true
license=MIT, author=jsquire4, type=module
repository=git+https://github.com/jsquire4/vitrum.git
workspaces=[packages/*, examples/*, tools/*]
overrides={ three: 0.171.0 }
```

- Will stay `private:true` forever (workspace root, not for publish). No action.
- Missing `homepage`, `bugs` — nice-to-have, not blocking.
- `release:dry-run` script exists (`npm pack --dry-run --workspaces --if-present`) — good.
- No `engines` field — minor; recommend `"engines": { "node": ">=20" }` once publish path is real (vitest 1.6 + TS 5.5 want modern Node).

### 2. `@vitrum/core`

Identity: name, version `0.0.0`, description, keywords (5), author, license MIT, repository (with `directory: packages/core`), `type: module`. **Strong identity.** Missing `homepage`/`bugs`.

Entrypoints:

```jsonc
"main": "src/index.ts",
"types": "src/index.ts",
"exports": {
  ".":        "./src/index.ts",
  "./scene":  "./src/scene.ts",
  "./engine": "./src/engine.ts",
  "./frame":  "./src/frame.ts"
}
```

Blocker: pointing at `.ts` files is fine for workspace consumers (TS will follow `paths`), but unsafe for npm consumers (they may use a bundler that doesn't resolve `.ts`, or no TS at all).

Build: `tsconfig.json` sets `composite:true` and `outDir: ./dist` (no `noEmit`), so this package *could* emit if a `tsc -b` were run — but there is **no `build` script** that invokes it. `dist/tsconfig.tsbuildinfo` exists from prior incremental compiles but no `.js`/`.d.ts`.

Files field: `["src", "README.md"]` — `README.md` is listed but **does not exist** in `packages/core/`. The root `README.md` does not ship inside the package tarball; npm will warn but not fail.

LICENSE: none in package dir. The MIT text lives at repo root only.

`private: true` — must flip to `false` (or remove) to publish.

`prepublishOnly`: absent. Recommended once a build exists.

Dependencies: clean. `peerDependencies: {}` (empty), `devDependencies: @webgpu/types ^0.1.40 + typescript ^5.5.0`. `@webgpu/types` is referenced via `tsconfig.base.json` `"types": ["@webgpu/types"]` — fine to keep as devDep since `Engine` types reference `GPUDevice`/`GPUTexture`/`GPUTextureFormat` etc. that come from those types; downstream consumers will need their own `@webgpu/types` install to typecheck. Consider promoting to a `peerDependency` (optional) so the typing intent is explicit.

### 3. `@vitrum/dev`

Identity: OK. Missing `homepage`/`bugs`. Description correctly notes it's a devDep-only package.

Entrypoints: `main`/`types` both `src/index.ts`. No `exports` map (single root entry). All React components live under `src/react/` and are re-exported from `index.ts`.

Build: no script. `tsconfig.json` sets `noEmit:true`. **`dist/` does not exist** for this package — `ls` shows no `dist/`. Tests + source ship raw.

Files: `["src", "README.md"]` — `README.md` does not exist.

LICENSE: none in package dir.

Dependencies:

- `dependencies`: `@vitrum/core: file:../core` — workspace `file:` specifier; **must be rewritten** to a version specifier (e.g. `^0.1.0-alpha.1`) before publish.
- `peerDependencies`: `react >=17`, `react-dom >=17`, `three >=0.167.0 <0.190` — all three flagged optional via `peerDependenciesMeta`. The `three` peer is interesting given the package imports nothing from `three` directly (only React/DOM); the optional `three` peer is presumably for future debug overlays. Keep, but document.
- `devDependencies`: full React + happy-dom + vitest stack — appropriate.

`private: true` — flip on publish.

`prepublishOnly`: absent.

### 4. `@vitrum/engine`

The drop-in facade — the most important publishable package since it's what users `import` first.

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints (3 subpaths):

```jsonc
"main": "src/index.ts",
"types": "src/index.ts",
"exports": {
  ".":           "./src/index.ts",
  "./lifecycle": "./src/lifecycle/index.ts",
  "./react":     "./src/react/index.ts"
}
```

The `./lifecycle` and `./react` subpaths are the load-bearing parts of the public API per the README (`attachVitrum` and `<VitrumCanvas>`). Subpath structure exists; just needs to point at built artifacts not `.ts`.

Build: no script. `noEmit:true`. No `dist/`.

Files: `["src", "README.md"]` — `README.md` does not exist (only the repo root has one).

LICENSE: none in pkg dir.

Dependencies (intra-workspace, **all `file:` specifiers — every one must be rewritten**):

- `@vitrum/core: file:../core`
- `@vitrum/three-bindings: file:../three-bindings`
- `@vitrum/walkaround-hybrid: file:../walkaround-hybrid`
- `@vitrum/pt-webgl: file:../pt-webgl` ← transitively depends on the un-publishable `three-gpu-pathtracer` file dep

Peer deps: `three >=0.167.0 <0.190`, `react >=17` (optional via meta), `react-dom >=17` (optional via meta) — three.js is correctly a non-optional peer; React subpath is optional.

`@types/three`: present only as a devDep, not as a peer. Since `src/createEngine.ts` and `sceneAABB.ts` import `THREE` types, downstream TS consumers need `@types/three` installed. **Recommend adding `@types/three` as an optional peer** (or document the install in README).

### 5. `@vitrum/pt-webgl` — **has the worst publish blocker**

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints: `main`/`types` → `src/index.ts`. No subpath `exports`.

Build: no script. `noEmit:true`. No JS/DTS in `dist/`.

Files: `["src", "README.md"]` — README **exists** (`packages/pt-webgl/README.md`). LICENSE absent.

The dependency block has the load-bearing publish blocker:

```jsonc
"dependencies": {
  "@vitrum/core":            "file:../core",
  "@vitrum/shared-samplers": "file:../shared-samplers",
  "@vitrum/three-bindings":  "file:../three-bindings",
  "three-gpu-pathtracer":    "file:../three-gpu-pathtracer",
  "three-mesh-bvh":          ">=0.7.4",
  "xatlas-web":              "^0.1.0"
}
```

- `three-gpu-pathtracer: file:../three-gpu-pathtracer` — now points at the absorbed workspace package. Before public publish, choose a real package name/version for this renderer (for example `@vitrum/three-gpu-pathtracer`) or keep `pt-webgl` private.
- `three-mesh-bvh` declared as both a regular dep AND a peer dep — **inconsistency**. Pick one. Since the import is direct in source (`import { ... } from 'three-mesh-bvh'`) but three.js is a peer dep, the right model is to make `three-mesh-bvh` a peer too (it has a `three` peer of its own; users should pin it once).
- `fixtures/` directory exists at the package root (`packages/pt-webgl/fixtures/hdrAccumGolden.bin`) — currently NOT listed in `files`, so it won't ship. Confirm this is intentional (test-only fixture).
- `src/__tests__/materialsTextureSpectral.test.ts` imports the absorbed renderer subpath through a Vitest alias. Tests don't ship to npm (they're inside `src/__tests__/` which currently does ship — see cross-cutting note below), but package `files` should exclude tests before publish.

Peer deps: `three >=0.167.0 <0.190`, `three-mesh-bvh >=0.7.4`. `three-mesh-bvh` appearing as BOTH a regular dep AND a peer dep is the duplication problem.

`@types/three`: only devDep. Same advice as `@vitrum/engine` — make it an optional peer.

### 6. `@vitrum/pt-webgpu`

Identity: OK. Missing `homepage`/`bugs`. Description correctly flags it as pre-alpha prototype.

Entrypoints: `main`/`types` → `src/index.ts`. No subpath exports.

Build: no script; `noEmit:true`.

Files: `["src", "README.md"]` — README exists. LICENSE absent.

Dependencies:

- `@vitrum/core: file:../core` → rewrite
- `@vitrum/shared-samplers: file:../shared-samplers` → rewrite
- devDeps include `@vitest/browser ^1.6.1` and `playwright ^1.59.1` — appropriate for the GPU test suite.

Tests:

- `test: "vitest run && vitest run --config vitest.gpu.config.ts"` — the default `test` script requires a Playwright-driven WebGPU environment. This is fine for CI but **not what `npm run test --workspaces` should default to in a fresh `npm install`** unless playwright browsers are pre-installed. Consider keeping `test` = `test:node` only, and reserving GPU runs for `test:gpu` (the latter is already a separate script).

`private: true`. Note `RELEASING.md` is silent on whether `pt-webgpu` ships in the first alpha; since it's "internal / not production", probably **stays `private:true` past 0.1.0-alpha.1**. Recommend explicit decision.

### 7. `@vitrum/shared-bvh`

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints: `main`/`types` → `src/index.ts`.

Build: none; `noEmit:true`.

Files: `["src", "README.md"]` — README absent.

LICENSE: absent.

Dependencies:

- `@vitrum/core: file:../core` → rewrite
- `@vitrum/shared-samplers: file:../shared-samplers` → rewrite
- `three-mesh-bvh: ">=0.7.4"` — **also declared as a peer dep**. Same duplication issue as `pt-webgl`. The src does `import { MeshBVH, StaticGeometryGenerator } from 'three-mesh-bvh'` — direct import. Make peer-only.

Peer deps: `three >=0.167.0 <0.190`, `three-mesh-bvh >=0.7.4`.

### 8. `@vitrum/shared-denoisers`

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints: `main`/`types` → `src/index.ts`.

Build: none; `noEmit:true`.

Files: `["src", "README.md"]` — README absent. LICENSE absent.

Dependencies:

- `@vitrum/core: file:../core` → rewrite

Peer deps: `onnxruntime-web ^1.18.0` (optional, gated by `peerDependenciesMeta`). The src uses a dynamic `await importFn('onnxruntime-web')` so the optional peer pattern is correctly implemented (`oidnBridge.ts:283-298`).

Tests script combines node + GPU like `pt-webgpu` — same concern (default `test` requires playwright).

### 9. `@vitrum/shared-samplers`

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints: `main`/`types` → `src/index.ts`.

Build: none; `noEmit:true`.

Files: `["src", "README.md"]` — README absent. LICENSE absent.

Dependencies: `@vitrum/core: file:../core` → rewrite.

No peer deps. No three.js dep. Cleanest package — first candidate for publish smoke test.

### 10. `@vitrum/three-bindings`

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints: `main`/`types` → `src/index.ts`.

Build: none; `noEmit:true`.

Files: `["src", "README.md"]` — README absent. LICENSE absent.

Dependencies:

- `@vitrum/core: file:../core` → rewrite

Peer deps: `three >=0.167.0 <0.190` — **correct.** Three is treated as a true peer here (the right shape).

`@types/three`: only devDep. Consider promoting to optional peer.

### 11. `@vitrum/walkaround-hybrid`

Identity: OK. Missing `homepage`/`bugs`.

Entrypoints: `main`/`types` → `src/index.ts`.

Build: none; `noEmit:true`.

Files: `["src", "README.md"]` — README **exists** (good). LICENSE absent.

Dependencies (all 5 are `file:` — must rewrite):

- `@vitrum/core: file:../core`
- `@vitrum/shared-bvh: file:../shared-bvh`
- `@vitrum/shared-samplers: file:../shared-samplers`
- `@vitrum/shared-denoisers: file:../shared-denoisers`
- `@vitrum/three-bindings: file:../three-bindings`

Peer deps: `three >=0.167.0 <0.190` — correct.

The src uses `three/webgpu` and `three/tsl` subpath imports (`giReceiver.ts`, `cascadeDispatch.ts`) — those subpaths exist in `three@0.167+` so the peer floor is correct.

`@types/three` is devDep only; same advice — consider optional peer.

---

## Cross-cutting issues

### A. The build story is the #1 problem

Every package except `@vitrum/core` declares `"noEmit": true` in its tsconfig. `@vitrum/core` is composite-emit-capable but no script invokes it. There is no top-level orchestrator (`tsc -b` or similar) that would produce JS+DTS for downstream consumers. The `dist/` directories scattered through the workspace contain only incremental-build metadata (`tsconfig.tsbuildinfo`) — phantom evidence of a build that was never finished.

**Required fix before any publish:**

1. Per-package `tsconfig.build.json` (or flip `noEmit` to `false` conditionally) that emits to `./dist/` with `declaration: true`, `declarationMap: true`, `sourceMap: true`.
2. Per-package `build` script (e.g. `"build": "tsc -p tsconfig.build.json"`).
3. Update each `main` → `./dist/index.js`, `types` → `./dist/index.d.ts`, `exports` map likewise.
4. Update each `files` field to `["dist", "README.md", "LICENSE"]` — and drop `src` once dist is the published artifact (or keep `src` if you intentionally want source-shipping for debuggability; some publishers do — that's a design choice).
5. Add `prepublishOnly: "npm run build"` to every publishable package so a stale tarball can't escape.

WGSL transport is via `.wgsl.ts` TypeScript modules (e.g. `packages/shared-denoisers/src/wgsl/atrousKernel.wgsl.ts`), not loose `.wgsl` static files — that's actually **publish-friendly** since TS compilation rolls them into the JS bundle. No asset-copy step needed.

### B. `private: true` on every package

Every package has `private: true` today as the safety belt `RELEASING.md` describes. Mechanically not a code blocker, but every package needs an explicit decision on whether it ships in 0.1.0-alpha.1:

| Package | Ship in 0.1.0-alpha.1? | Notes |
|---|---|---|
| `@vitrum/core` | yes (the contract) | mandatory; everything depends on it |
| `@vitrum/three-bindings` | yes | adapter, no GPU code |
| `@vitrum/shared-samplers` | yes | no three.js dep, pure utility |
| `@vitrum/shared-bvh` | yes | needs `three-mesh-bvh` peer cleanup first |
| `@vitrum/shared-denoisers` | yes | OIDN bridge gated on optional peer |
| `@vitrum/walkaround-hybrid` | yes | the WebGPU GI backend |
| `@vitrum/engine` | yes | the facade — required entry point |
| `@vitrum/dev` | yes (as devDep) | debug overlays |
| `@vitrum/pt-webgl` | **gated** | blocked on renderer package publish/private decision |
| `@vitrum/pt-webgpu` | **probably no** | pre-alpha prototype; keep `private:true` past 0.1.0-alpha.1 |

`RELEASING.md` already calls out the pt-webgl gating. Status quo: keep pt-webgl private for 0.1.0-alpha.1 unless the absorbed renderer package is published under a vitrum-owned package name.

### C. `file:` workspace deps must become version specifiers

Counted across the workspace, there are **11 distinct `file:` specifiers** that must become semver specifiers before publish:

1. `engine` → `core` (file:../core)
2. `engine` → `three-bindings`
3. `engine` → `walkaround-hybrid`
4. `engine` → `pt-webgl`
5. `dev` → `core`
6. `pt-webgl` → `core`
7. `pt-webgl` → `shared-samplers`
8. `pt-webgl` → `three-bindings`
9. `pt-webgl` → `three-gpu-pathtracer` (absorbed renderer workspace package — separate Section D below)
10. `pt-webgpu` → `core`
11. `pt-webgpu` → `shared-samplers`
12. `shared-bvh` → `core`
13. `shared-bvh` → `shared-samplers`
14. `shared-denoisers` → `core`
15. `shared-samplers` → `core`
16. `three-bindings` → `core`
17. `walkaround-hybrid` → `core`
18. `walkaround-hybrid` → `shared-bvh`
19. `walkaround-hybrid` → `shared-samplers`
20. `walkaround-hybrid` → `shared-denoisers`
21. `walkaround-hybrid` → `three-bindings`

(That's 21 total intra-workspace `file:` references when each is counted per-occurrence; mechanical replacement is the right tool — script it.)

The conventional npm-workspace pattern is `"@vitrum/core": "^0.1.0-alpha.1"` with the lockfile resolving locally during dev. (Newer npm supports `"@vitrum/core": "workspace:*"` syntax that rewrites at publish time — viable alternative.)

### D. The `three-gpu-pathtracer` renderer package

Already documented in `RELEASING.md` §"The `three-gpu-pathtracer` renderer package". The sibling checkout blocker is closed; the remaining decision is whether to publish the absorbed renderer under a vitrum-owned scope/name and pin that version, or keep `@vitrum/pt-webgl` private.

Knock-on effect: `@vitrum/engine` depends on `@vitrum/pt-webgl`. The facade either:

- ships with WebGL2 backend dynamically imported and gracefully degrades when missing (preferred — the contract is "Engine picks for you"); or
- ships with WebGL2 path disabled in the published version (worse — breaks `prefer: 'quality'` and the entire `pt-webgl` capability column in the README).

### E. `three-mesh-bvh` declared as BOTH dep and peer

In `@vitrum/pt-webgl` and `@vitrum/shared-bvh`:

```jsonc
"dependencies":     { ..., "three-mesh-bvh": ">=0.7.4" },
"peerDependencies": { ..., "three-mesh-bvh": ">=0.7.4" }
```

This causes consumers to either install two copies (bundler dedup may rescue you, may not) or get a confusing "missing peer dep" warning despite the regular dep being installed. Pick one:

- **Peer-only** (recommended): `three-mesh-bvh` has its own `three` peer, so the consumer should pin one copy at the top level. Remove from `dependencies`, keep in `peerDependencies`.
- Dep-only: drop peer, ship the dep directly — risks version conflicts with the user's own `three-mesh-bvh` install.

### F. `@types/three` should probably be an optional peer

Currently devDep-only on `engine`, `three-bindings`, `walkaround-hybrid`, `shared-bvh`, `pt-webgl`. Since the published `.d.ts` files (once built) will reference THREE types, the consumer must have `@types/three` installed. Either:

- Make it an optional peer (declared intent, no runtime cost): `"peerDependencies": { "@types/three": "..." }` + `peerDependenciesMeta.{ optional: true }`; or
- Document the install in each package README.

The strict TS option is to inline the THREE types into your `.d.ts` via `tsc --declaration --declarationMap` without `--composite`, but that fights how TS treats external types.

### G. `dist/` directories are stale build artifacts that should be `.gitignored` (and are) — but `tsbuildinfo` files are committed?

`.gitignore` lists `dist/` and `*.tsbuildinfo`. The `ls` output shows `dist/tsconfig.tsbuildinfo` files exist in many packages — these are uncommitted local-build leftovers, not in the repo. No git-state issue, but worth a `npm run clean` before any `npm pack --dry-run`. The root has a `clean` workspace pass-through script but no `clean` script in any package currently.

### H. Source tree includes `__tests__/` directories that currently DO ship

Most packages have either `src/__tests__/` (pt-webgl, pt-webgpu, shared-*, three-bindings) or sibling `__tests__/` (engine, dev, walkaround-hybrid, shared-denoisers). The `files: ["src"]` field ships everything inside `src/`, including `src/__tests__/`. The sibling `__tests__/` directories are NOT shipped (good).

To avoid shipping test files to consumers, either:

- Move tests to a sibling `__tests__/` directory (the pattern several packages already use); or
- Add `"!src/**/__tests__"` to `files` (npm 8+ supports negation); or
- Once `dist/` becomes the published artifact and tests are excluded from `tsconfig.build.json`, the test files won't end up in `dist/` — easiest fix.

### I. No CHANGELOG-per-package, only top-level

`CHANGELOG.md` is monorepo-wide. That's a defensible choice for an internal pre-alpha but unconventional for multi-package npm publish — typical pattern is per-package `CHANGELOG.md`. Not a blocker; flag for design decision. If staying monorepo-wide, link each package README to the root `CHANGELOG.md`.

### J. Per-package READMEs missing for most packages

Only `pt-webgl`, `pt-webgpu`, and `walkaround-hybrid` have README files. The other 7 have `"README.md"` in their `files` array but no file exists. When `npm pack` runs, the absent README is silently skipped — but `npmjs.com` displays the package page from the tarball README, so packages without one show a blank landing page.

Mandatory before publish: README per package, at minimum with: install command, 1-line description, 5-line quickstart, link to repo root for full docs.

### K. Per-package LICENSE files missing for ALL packages

No package directory contains a `LICENSE` file. Only the repo root has one. Since `npm pack` ships the package directory's contents (only), the published tarball ships without a LICENSE file unless one is symlinked/copied in.

Fix: copy `LICENSE` from repo root into each package directory, OR add `"./LICENSE": "../../LICENSE"` symlinks. The `package.json` `license: "MIT"` field tells npm the SPDX identifier but doesn't replace the LICENSE text file.

### L. No `engines` field in any `package.json`

Vitest 1.6 + TypeScript 5.5 + WebGPU types want modern Node. Add `"engines": { "node": ">=20" }` to each publishable package's `package.json`. Minor — not a hard publish blocker, but consumers on Node 18 will get an "engine warning" rather than a clear "this won't work".

### M. No `prepublishOnly` script anywhere

`prepublishOnly` runs before `npm publish` and is the standard guard for "don't ship without a fresh build". Add `"prepublishOnly": "npm run build && npm test"` (or similar) per package once a build exists. Without it, a stale tarball is one command away from going public.

### N. `tsconfig.base.json` declares `types: ["@webgpu/types"]` globally

That means any package without `@webgpu/types` in its own devDeps inherits the type lookup. Only `@vitrum/core` declares `@webgpu/types` as a devDep. The other packages need it transitively at consume-time too — this is the same problem as `@types/three`, just less severe (the GPU types are referenced by `Engine` itself, not by every backend).

Cleanest fix: bump `@webgpu/types` from devDep on `core` to a real `peerDependency` (optional) on every package that exposes GPU types through its public API.

---

## Suggested fix order — cheapest wins first

These are ordered by "minimum effort for maximum publish-readiness move". An "S" task takes <30 min, "M" <2 hours, "L" half-day to multi-day.

1. **[S] Add per-package `README.md`** for the 7 packages missing one (`core`, `dev`, `engine`, `shared-bvh`, `shared-denoisers`, `shared-samplers`, `three-bindings`). Stub with install command + 1-paragraph description; link to repo root for everything else.
2. **[S] Copy `LICENSE` into each of the 10 package directories.** One file × 10 dirs.
3. **[S] Add `homepage` and `bugs` to every `package.json`**. Both can point to the GitHub repo.
4. **[S] Add `engines: { node: ">=20" }`** to every publishable package's `package.json`.
5. **[S] Fix `three-mesh-bvh` dep/peer duplication** in `@vitrum/pt-webgl` and `@vitrum/shared-bvh` (remove from `dependencies`, keep as peer).
6. **[M] Decide and document which packages ship in 0.1.0-alpha.1** vs. which stay `private:true` past the first publish (likely `pt-webgpu` stays private; possibly `pt-webgl` until fork-publish lands).
7. **[M] Promote `@types/three` and `@webgpu/types` to optional peer dependencies** on the relevant packages (so consumer install warnings reflect intent).
8. **[L] Set up the real build pipeline** — per-package `tsconfig.build.json`, per-package `build` script, point `main`/`types`/`exports` at `./dist/*.js` and `./dist/*.d.ts`, drop `noEmit:true` from build configs, exclude `__tests__` from the build, add `prepublishOnly: "npm run build"`. Verify with `npm pack --dry-run --workspaces`. This is the single largest piece of work and the actual gate.
9. **[L] Rewrite all 21 `file:` intra-workspace deps to version specifiers** (or `workspace:*`). Lockstep-bump every package from `0.0.0` to `0.1.0-alpha.1`. Update `package-lock.json`.
10. **[L] Resolve the `three-gpu-pathtracer` renderer package for publish**: publish the absorbed package as `@vitrum/three-gpu-pathtracer@0.7.X` (or comparable), rewrite the `file:` specifier in `@vitrum/pt-webgl` to a real version, and verify the `@vitrum/engine` facade can either link `pt-webgl` or dynamically degrade.
11. **[S] Flip `private: false`** on every package that's in scope for 0.1.0-alpha.1. Do this as the last commit before publishing — never earlier.

---

## Estimated effort summary

| Category | Tasks | Estimated effort |
|---|---|---|
| Identity polish (READMEs, LICENSE, homepage/bugs/engines) | 1–4 | **S** total ~2 hours |
| Dep hygiene (three-mesh-bvh dup, @types/three peer, @webgpu/types peer) | 5, 7 | **S–M** ~2–4 hours |
| Publish-scope decisions | 6 | **M** ~1 hour discussion + edits |
| Build pipeline (the big one) | 8 | **L** ~1–2 days |
| `file:` rewrite + version bump | 9 | **L** ~half-day |
| `three-gpu-pathtracer` fork-publish + facade dynamic-load | 10 | **L** ~1–2 days |
| Final publish flip + dry-run + publish | 11 | **S** ~1 hour |

**Total estimated effort: 4–6 days of focused work to go from today's state to a clean 0.1.0-alpha.1 publish.** The shape of that effort is: 1 day of nuisance polish, 2–3 days of real build pipeline work, 1–2 days of fork resolution + facade refactor.

---

## What this audit did NOT do

- Did not modify any `package.json`. Read-only investigation only.
- Did not run `npm pack --dry-run` (would have surfaced npm's own warnings — recommended as the next concrete validation step).
- Did not audit `examples/*` or `tools/*` workspace packages (they ship inside the repo, not to npm — only spot-checked for completeness).
- Did not validate that the existing tests pass — `RELEASING.md` already mandates `npm run typecheck && npm test` as a pre-publish gate; this audit assumes those pass per the project status in `CLAUDE.md`.
- Did not propose specific version numbers or `version` bump plans (`RELEASING.md` already documents `0.1.0-alpha.1` as the first publish; just enumerated what must change before that version is real).

---

## Cross-references

- [`RELEASING.md`](../RELEASING.md) — the canonical publish playbook; already enumerates the `three-gpu-pathtracer` fork blocker and the `private:true` safety belt.
- [`CLAUDE.md`](../CLAUDE.md) — project conventions, includes the "no upstream PRs, no npm publish, no remote pushes without instruction" rule.
- [`CHANGELOG.md`](../CHANGELOG.md) — monorepo changelog; needs a `## [0.1.0-alpha.1] - <date>` heading before first publish per RELEASING.md §"Pre-publish checklist".
- [`README.md`](../README.md) — declares the package status "pre-alpha, private. Not yet on npm" — that status banner is what flips when 0.1.0-alpha.1 ships.
