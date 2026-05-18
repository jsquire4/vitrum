/**
 * PassRegistry — collection of {@link Pass} implementations sorted
 * topologically by their declared dependencies.
 *
 * The registry validates: unique IDs, no missing dependencies, no
 * cycles. The sorted order is deterministic — within a single
 * dependency tier, IDs are visited in lexicographic order so the same
 * registration set always yields the same sequence (important for
 * test reproducibility).
 *
 * Premium-library rationale: replaces the position-encoded pass order
 * in {@link WalkaroundGPUPipeline.renderFrame} (~550 LOC method body)
 * with a single data-driven ordering — passes can be added, removed,
 * or reordered without editing the orchestrator (complexity sweep
 * 2026-05-17 Theme B, W1-R5 in the premium-grade refactor plan).
 */

import type { Pass, PassGateOptions } from './Pass.js';

export class PassRegistry {
  private readonly _passes = new Map<string, Pass>();
  private _sortedIds: readonly string[] | null = null;

  /** Register a pass. Throws on duplicate ID. Invalidates the sorted
   *  cache so the next call to {@link sortedPasses} re-sorts. */
  register(pass: Pass): void {
    if (this._passes.has(pass.id)) {
      throw new Error(`PassRegistry: duplicate pass id "${pass.id}"`);
    }
    this._passes.set(pass.id, pass);
    this._sortedIds = null;
  }

  /** Lookup a pass by id. */
  get(id: string): Pass | undefined {
    return this._passes.get(id);
  }

  has(id: string): boolean {
    return this._passes.has(id);
  }

  size(): number {
    return this._passes.size;
  }

  /** All registered IDs in registration order (NOT topo order). */
  ids(): readonly string[] {
    return Array.from(this._passes.keys());
  }

  /** Passes in topological order. Cached until the next register call.
   *  Throws on unknown dependency or cycle. */
  sortedPasses(): readonly Pass[] {
    if (this._sortedIds === null) {
      this._sortedIds = this._topoSort();
    }
    return this._sortedIds.map((id) => this._passes.get(id)!);
  }

  /** Subset of {@link sortedPasses} after applying per-pass
   *  {@link Pass.gates}. Convenience for the per-frame dispatch loop. */
  activePasses(opts: PassGateOptions): readonly Pass[] {
    return this.sortedPasses().filter((p) => p.gates(opts));
  }

  /** Kahn's algorithm with lexicographic tiebreaking. */
  private _topoSort(): readonly string[] {
    const inDegree = new Map<string, number>();
    const dependents = new Map<string, string[]>();

    for (const id of this._passes.keys()) {
      inDegree.set(id, 0);
      dependents.set(id, []);
    }

    for (const [id, pass] of this._passes) {
      for (const dep of pass.dependencies) {
        if (!this._passes.has(dep)) {
          throw new Error(`PassRegistry: pass "${id}" declares unknown dependency "${dep}"`);
        }
        inDegree.set(id, (inDegree.get(id) ?? 0) + 1);
        dependents.get(dep)!.push(id);
      }
    }

    // Sorted queue keeps output deterministic within a dependency tier.
    const queue: string[] = [];
    for (const [id, d] of inDegree) {
      if (d === 0) queue.push(id);
    }
    queue.sort();

    const sorted: string[] = [];
    while (queue.length > 0) {
      const id = queue.shift()!;
      sorted.push(id);
      for (const downstream of dependents.get(id) ?? []) {
        const next = (inDegree.get(downstream) ?? 0) - 1;
        inDegree.set(downstream, next);
        if (next === 0) {
          let i = 0;
          while (i < queue.length && queue[i]! < downstream) i++;
          queue.splice(i, 0, downstream);
        }
      }
    }

    if (sorted.length !== this._passes.size) {
      const remaining = [...this._passes.keys()].filter((id) => !sorted.includes(id));
      throw new Error(`PassRegistry: cycle detected involving passes [${remaining.join(', ')}]`);
    }

    return Object.freeze(sorted);
  }
}
