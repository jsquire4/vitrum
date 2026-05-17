/**
 * WGSL include-graph composer.
 *
 * Replaces the hand-rolled `COMMON_WGSL + X_WGSL` concatenation pattern that
 * spread across `pipelineCompiler.ts` and the denoiser entries (W1-R6,
 * complexity-sweep 2026-05-17 Theme C). Each WGSL module declares its
 * dependency set as a list of names; `composeWgsl` resolves the transitive
 * closure, deduplicates, and emits each dependency exactly once in dep-first
 * order before appending the root module's own source.
 *
 * Determinism: same `(root, registry)` → byte-identical output. Order is
 *   1. transitive deps, post-order DFS over `requires` arrays in declared
 *      order (so the leaf-most appears first, and ties break by the order
 *      the parent listed them);
 *   2. the root module's `source` appended last.
 * A dependency that is reached via multiple paths is emitted ONCE at its
 * first-completion point — equivalent to the standard "include-once" rule.
 *
 * Bit-identical contract: the composer concatenates raw strings with no
 * inserted whitespace. WGSL treats arbitrary inter-chunk whitespace as
 * identical, so any composed output is functionally equivalent to the
 * original hand-concatenated pattern provided the same set of deps is
 * emitted in the same order. The pre-R6 patterns happened to be:
 *   `COMMON_WGSL + X_WGSL`                           (most passes)
 *   `COMMON_WGSL + SURFACE_TEXTURES_WGSL + DDGI_SAMPLE_WGSL + SHADE_WGSL`
 *   `COMMON_WGSL + DDGI_SAMPLE_WGSL + RIS_GI_WGSL`
 * which the include-graph reproduces verbatim by declaring `requires` in
 * matching dep-first listed order.
 */

export interface WgslModule {
  /** Stable identifier — referenced by other modules' `requires` arrays. */
  readonly name: string;
  /** Raw WGSL source for this module (without its own deps prepended). */
  readonly source: string;
  /** Names of modules this module depends on, in the order they should be
   *  emitted relative to each other when reached via this module. */
  readonly requires: readonly string[];
}

/**
 * Compose a root module's WGSL source by prepending the transitive closure
 * of its declared dependencies. Each dependency is emitted exactly once.
 *
 * @throws if a dependency cycle is detected (re-entering a module that is
 *         already on the DFS stack) or if a required name is not present
 *         in the registry.
 */
export function composeWgsl(
  rootModule: WgslModule,
  registry: ReadonlyMap<string, WgslModule>,
): string {
  const emitted = new Set<string>();
  const onStack = new Set<string>();
  const chunks: string[] = [];

  // Post-order DFS over the dependency graph. We do NOT register the root
  // itself in `emitted` until after its deps are flushed, so a self-reference
  // (which would be a cycle) is caught by the `onStack` check.
  const visit = (mod: WgslModule, path: string[]): void => {
    if (emitted.has(mod.name)) return;
    if (onStack.has(mod.name)) {
      throw new Error(
        `[wgslComposer] Dependency cycle detected: ${[...path, mod.name].join(' -> ')}`,
      );
    }
    onStack.add(mod.name);
    for (const depName of mod.requires) {
      const dep = registry.get(depName);
      if (!dep) {
        throw new Error(
          `[wgslComposer] Module '${mod.name}' requires unknown module '${depName}' ` +
          `(path: ${[...path, mod.name].join(' -> ')})`,
        );
      }
      visit(dep, [...path, mod.name]);
    }
    onStack.delete(mod.name);
    emitted.add(mod.name);
    chunks.push(mod.source);
  };

  // Visit deps first, then append the root's own source. We do this by
  // visiting each dep explicitly (rather than calling visit(rootModule))
  // so the root's source is appended LAST, mirroring the historical
  // `COMMON_WGSL + ROOT_WGSL` pattern.
  onStack.add(rootModule.name);
  for (const depName of rootModule.requires) {
    const dep = registry.get(depName);
    if (!dep) {
      throw new Error(
        `[wgslComposer] Root module '${rootModule.name}' requires unknown module '${depName}'`,
      );
    }
    visit(dep, [rootModule.name]);
  }
  onStack.delete(rootModule.name);
  chunks.push(rootModule.source);

  return chunks.join('');
}
