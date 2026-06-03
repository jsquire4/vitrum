/**
 * Static WGSL cross-module symbol-resolution checker (T9-stepC).
 *
 * The W1-R6 include-graph lets each compute pass declare a `requires` list of
 * WGSL modules; `composeWgsl` flattens that closure into one shader string.
 * T9-stepC narrows each pass's `requires` from the full `common` aggregate
 * (11 focused modules) down to the minimal subset it actually references.
 *
 * There is NO GPU in CI to catch a narrowed-too-far pass at shader-compile
 * time, so this module provides a STATIC safety net.
 *
 * Approach — cross-module symbol resolution (not full name resolution):
 *   1. The "symbol universe" is the set of module-scope identifiers DECLARED
 *      by the focused/helper modules a pass might pull in (struct / fn / const
 *      / alias / module-scope var names). This is the set of names that a pass
 *      can ONLY obtain by including the declaring module.
 *   2. A pass's "available symbols" are the declared module-scope identifiers
 *      in its composed-WGSL closure.
 *   3. A pass's "used library symbols" are the universe symbols whose name
 *      appears as a token in the pass's OWN source.
 *   4. The narrowed module set is sound iff  used ⊆ available.
 *
 * Why this is sound (no false negatives for the failure mode we care about):
 *   - If the pass references a cross-module symbol `foo` but its narrowed
 *     closure does not declare `foo`, the gate fails — exactly the
 *     "narrowed-too-far" case the GPU would reject as an unresolved
 *     identifier.
 *   - Locals and struct-field accesses never enter the universe (they are not
 *     module-scope declarations of the *library* modules), so they cannot
 *     cause false positives — the regex over-collection of local tokens is
 *     intersected away against the universe.
 *
 * Residual conservative edge: if a pass declares a LOCAL variable whose name
 * collides with a library symbol it does NOT otherwise use, the gate would
 * demand the declaring module. That is over-strict (a false positive), never
 * unsound, and is trivially handled by widening `requires` if it ever occurs.
 * In practice the walkaround shaders do not collide local names with the
 * distinctively-named library helpers (`traceSceneFirstHit`, `evalGGX`, …).
 */

/** Strip `//` line comments and block comments from WGSL source. */
function stripComments(src: string): string {
  let out = src.replace(/\/\*[\s\S]*?\*\//g, ' ');
  out = out.replace(/\/\/[^\n]*/g, ' ');
  return out;
}

/**
 * Collect module-scope DECLARED identifiers from a WGSL source.
 *
 * Matches the module-scope declaration forms:
 *   - `struct Name {`
 *   - `fn name(`
 *   - `const NAME` / `const NAME:`
 *   - `alias Name =`
 *   - `var<...> name` / `var name` (module-scope globals incl. bindings)
 *
 * Only MODULE-SCOPE declarations are collected. Function-local `let`/`var`
 * are deliberately excluded:
 *   - `let` is never matched.
 *   - module-scope `var` in these shaders ALWAYS carries an address-space
 *     template (`var<storage,…>` / `var<uniform>`) or is a resource binding
 *     (`@group(..) @binding(..) var name: texture_… / sampler`). Function-
 *     local `var name = …` / `var name: T` has neither, so we match only the
 *     templated form and the binding form. This keeps single-letter locals
 *     (`var i = 0u;`) OUT of the symbol universe, which is essential — a
 *     polluted universe would produce false positives against pass locals.
 */
function collectDeclaredIdents(src: string): Set<string> {
  const clean = stripComments(src);
  const decls = new Set<string>();
  const add = (reSrc: string): void => {
    const g = new RegExp(reSrc, 'g');
    let m: RegExpExecArray | null;
    while ((m = g.exec(clean)) !== null) {
      if (m[1]) decls.add(m[1]);
    }
  };
  add('\\bstruct\\s+([A-Za-z_][A-Za-z0-9_]*)');
  add('\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)');
  add('\\bconst\\s+([A-Za-z_][A-Za-z0-9_]*)');
  add('\\balias\\s+([A-Za-z_][A-Za-z0-9_]*)');
  // Module-scope `var` ONLY: templated address-space form (`var<…> name`)…
  add('\\bvar<[^>]*>\\s+([A-Za-z_][A-Za-z0-9_]*)');
  // …or a resource binding (`@binding(..) var name: …`).
  add('@binding\\([^)]*\\)\\s*var\\s+([A-Za-z_][A-Za-z0-9_]*)');
  return decls;
}

/** Collect every identifier token appearing in a source (de-duplicated). */
function collectTokens(src: string): Set<string> {
  const clean = stripComments(src);
  const out = new Set<string>();
  const re = /[A-Za-z_][A-Za-z0-9_]*/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(clean)) !== null) {
    out.add(m[0]);
  }
  return out;
}

export interface ResolutionInputs {
  /** Raw source of the pass module itself (NOT the composed closure). */
  readonly ownSource: string;
  /** Composed WGSL of the pass's narrowed closure (deps + own source). */
  readonly composed: string;
  /** Union of declared module-scope symbols across the LIBRARY modules a pass
   *  may include (the symbol universe). */
  readonly symbolUniverse: ReadonlySet<string>;
}

export interface ResolutionResult {
  /** Library symbols the pass uses but whose declaration is absent from the
   *  composed closure — the narrowed-too-far failures. Empty = sound. */
  readonly missing: string[];
  /** Library symbols the pass actually references (for diagnostics). */
  readonly used: string[];
}

/**
 * The gate: every library symbol the pass references must be declared in its
 * composed closure.
 */
export function checkCrossModuleResolution(inputs: ResolutionInputs): ResolutionResult {
  const ownTokens = collectTokens(inputs.ownSource);
  const available = collectDeclaredIdents(inputs.composed);
  const used: string[] = [];
  const missing: string[] = [];
  for (const sym of inputs.symbolUniverse) {
    if (!ownTokens.has(sym)) continue;
    used.push(sym);
    if (!available.has(sym)) missing.push(sym);
  }
  used.sort();
  missing.sort();
  return { missing, used };
}

/**
 * Build the symbol universe from a set of library module sources.
 */
export function buildSymbolUniverse(sources: readonly string[]): Set<string> {
  const universe = new Set<string>();
  for (const src of sources) {
    for (const id of collectDeclaredIdents(src)) universe.add(id);
  }
  return universe;
}
