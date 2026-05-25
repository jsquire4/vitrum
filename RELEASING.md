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

## The `three-gpu-pathtracer` renderer package

`@vitrum/pt-webgl` depends on the absorbed in-repo
`packages/three-gpu-pathtracer` workspace package via
`file:../three-gpu-pathtracer`. The old sibling-repo fork dependency is gone,
so CI and local development no longer need a checkout next to `vitrum/`.

Before publishing `@vitrum/pt-webgl`, decide how to publish this renderer
package:

1. **Publish the absorbed package under a vitrum-owned scope** (for example
   `@vitrum/three-gpu-pathtracer`) and update `pt-webgl/package.json` to use
   that version. Cleanest public-release path.
2. **Keep `pt-webgl` private** while other packages publish.
3. **Bundle/vendor the renderer into `pt-webgl` at publish time.** Avoid unless
   package-level separation becomes more trouble than it is worth.

Until that packaging decision is made, `@vitrum/pt-webgl` remains private even
though it no longer depends on files outside this repository.

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
