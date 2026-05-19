# Releasing vitrum

This document describes how to publish vitrum's packages to npm. **Until the user runs the publish command, vitrum stays private.** Every package in this repo currently has `"private": true` as a publish-safety belt — that flag must be removed (or `npm publish` overridden) before any package can be released.

## Pre-publish checklist

1. **All quality gates green.** From the repo root:
   ```sh
   npm run typecheck     # all 12 packages clean
   npm test              # all packages clean (~1209 tests across 11 workspace packages, 3 intentional GPU-only skips)
   npm run test:gpu      # GPU-browser tests — requires `npx playwright install chromium`
   npm run fork-shader-smoke
   npm run shader-compile-ci
   ```

2. **Changelog updated.** Add a `## [0.1.0-alpha.1] - <date>` heading to [CHANGELOG.md](./CHANGELOG.md) summarizing every commit since the last release.

3. **Cornell reference renders match.** Re-run the cornell-box scenes at 64 SPP and visually A/B against `tools/reference-renders/post-sweep-20260512/`. Numerical regression is acceptable only if visually justified (see the testing protocol in `CLAUDE.md`).

4. **Branch is on `main`.** Releases happen from `main`. Feature branches must merge first.

## The `@vitrum` npm scope

`@vitrum/*` is unregistered as of 2026-05-12 (`npm view @vitrum/core` returns 404; `npm search scope:vitrum` returns 0 results). Before the first publish, claim the scope:

```sh
npm login                  # use the GitHub-linked npm account jsquire4 owns
npm org create vitrum      # creates the @vitrum org, costs $0 for public packages
```

If `vitrum` was claimed by someone else between now and publish day, fall back to a different scope (e.g. `@jsquire4-vitrum`) and rewrite every `package.json`'s `name` and `dependencies` accordingly.

## The `three-gpu-pathtracer` fork dep

`@vitrum/pt-webgl` currently depends on `three-gpu-pathtracer` via `file:../../../three-gpu-pathtracer` — a sibling-repo file dep that npm cannot publish. **This must be resolved before pt-webgl can publish.** Options:

1. **Publish the fork to npm** under your own scope (e.g. `@jsquire4/three-gpu-pathtracer`). Update `pt-webgl/package.json` to depend on the published version. Cleanest; what we'd want long-term.
2. **Wait for the fork's patches to land upstream** in `gkjohnson/three-gpu-pathtracer` and depend on the upstream version. Best for ecosystem health; slowest.
3. **Vendor the fork's relevant files into pt-webgl** at publish time. Worst for maintenance; not recommended.

Until one of these happens, `@vitrum/pt-webgl` cannot publish but the rest of the workspace can. The facade `@vitrum/engine` will need to dynamically import the WebGL2 backend and gracefully degrade when it isn't available — or the WebGL2 path stays disabled in the published facade.

## Publishing

After the checklist passes and the scope is claimed:

```sh
# 1. Verify what each package will publish (no actual upload).
npm pack --dry-run --workspaces

# 2. Remove "private": true from each package you intend to publish.
#    Do this by hand — one mistake here ships an unintended package.

# 3. Publish under the alpha tag so it doesn't become the default version
#    for `npm install @vitrum/engine`. Users who want the alpha must opt in
#    via `npm install @vitrum/engine@alpha`.
npm publish --workspaces --tag alpha --access public
```

`--access public` is required because scoped packages default to private (paid feature). All vitrum packages are MIT-licensed and meant to be public.

## Versioning

vitrum stays at `0.0.0` until the first publish. The first published version is `0.1.0-alpha.1`. SemVer applies once the package leaves alpha:

| Bump          | When                                                        |
| ------------- | ----------------------------------------------------------- |
| `0.1.0-alpha.X` | Pre-alpha iteration; any breaking change.                 |
| `0.1.0`       | First non-alpha release. From here on, SemVer applies.     |
| `0.1.X`       | Bugfixes + non-breaking adds.                              |
| `0.2.0`       | Breaking change to a public type or function signature.    |
| `1.0.0`       | API stable. Documented breaking-change policy in effect.   |

## Post-publish

1. Tag the release in git: `git tag v0.1.0-alpha.1 && git push origin v0.1.0-alpha.1`.
2. Cut a GitHub Release pointing at the tag, with the changelog entry as the body.
3. Verify a fresh `npm install @vitrum/engine@alpha` in a scratch directory installs cleanly + types resolve.
4. Update the project README's status badge from "pre-alpha, private" to "pre-alpha, public".

## Rollback

If a published version is broken, **do not delete it from npm**. Instead:

```sh
npm deprecate @vitrum/engine@0.1.0-alpha.1 "broken; use 0.1.0-alpha.2"
```

`npm unpublish` is reserved for last resorts (security incidents). It breaks the lockfile of every consumer who installed during the broken window.
