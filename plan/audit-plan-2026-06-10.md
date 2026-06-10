# Trust Audit Plan — 2026-06-10

> **Mission.** The question this audit answers is not "is the code wrong?" — it is
> **"can a stranger trust this library?"** Two months and enormous effort have gone in;
> the completion estimate has read ~85% for two straight weeks because every audit walked
> the same well-lit rooms (the ledgers, the critical path, greppable patterns). This audit
> is designed around that failure: coverage is **enumerated and accounted**, not impressionistic.
> The deliverable is a completion number with a denominator behind it.

## The three finding classes (every agent reports against all three)

1. **Broken promise** — code exists but is wrong, stubbed, dead-branched, or unwired.
   (The classic audit class. Necessary, not sufficient.)
2. **Unkept promise** — a type, option, JSDoc, README, or capability grade claims
   something the code does not deliver, in full or in part.
3. **Missing promise** — the class no prior audit chased: a capability a reasonable user
   of THIS kind of library (a browser path-tracing/GI engine) would expect and does not
   find — or finds on one backend and inexplicably not another, or finds with no way to
   reach it from the public API. Product wholeness and cohesion, not just correctness.
   Examples of the *shape* (not a checklist): can a host load an ordinary browser Image
   as a texture? recover from device loss? serialize a scene? know why a frame is black?
   migrate between backends without surprises? find any feature from the README alone?

**Trust framing for severity:** a finding's severity = how badly it would burn a user who
trusted the library. Silent wrongness > loud failure > missing nicety.

## Why agents must behave differently in this audit

- **Anti-anchoring:** excavation agents are NOT given the ledgers (items_to_fix, road-to-100)
  and must not seek them out — agents who know where past audits looked re-tread them.
- **Forced coverage:** assignments are explicit file lists; output is a PER-FILE table.
  A skipped file is visible as a missing row. "I covered the package" is not a sentence
  this audit accepts.
- **Evidence or it didn't happen:** every claim carries file:line + a quoted excerpt;
  every "it runs/compiles/passes" claim carries the exact command + pasted output.
  Sub-agent fabrication has been caught in this repo (a "local gate run" that never
  happened); all output is verified by the lead, and a plain **"I could not verify X"**
  is a GOOD outcome — fabricated success is the worst possible outcome.
- **Completeness over cleverness:** finding that something is *absent* counts as much as
  finding that something is broken. The report template has a mandatory
  "what is MISSING here" row per file/area.

## Coverage accounting (the plan-vs-codebase audit the lead performs)

- Source inventory: **469 files** (`plan/audit-darkmap-2026-06-10.txt`).
  - **155 never-cited files** (the dark map, listed in that file) → Track B1 reading
    assignments, every file, every line.
  - **314 previously-cited files** → covered by the 2026-06-10 10-agent audit + the
    v1-closure campaign; re-covered here only via Tracks A/B2-B5 (which cut across all
    files behaviorally, not by reading).
- The **completeness critic** (final step) diffs every B1 agent's per-file table against
  its assignment, samples N claims per agent for lead re-verification, and lists any file
  in the inventory not verifiably opened by anyone. Unaccounted file ⇒ audit not done.

## Track A — Quality of the path we trod (adversarial fix review)

Per-commit skeptic review of the v1-closure campaign (6e90443, 0dbaff5, 44ada1a, 1d31f0b,
06910e2, caab499, d67f0a3). Mandate: REFUTE each fix. Does its test pin behavior or just
strings? Re-derive the math from first principles (BDPT pdf conventions, RC 4π/N estimator,
SPPM radius law, VNDF identity, Preetham polynomials). Did it damage a neighbor? Agents:
2 (commits split by area). No agent audits work from its own round.

## Track B — The excavation (where nobody trod)

- **B1 — Dark-map deep read** (7 agents): the 155 never-cited files, explicit ~22-file
  lists. Per file: (a) role in one sentence, (b) wiring — trace and QUOTE the call sites
  yourself or write "UNWIRED", (c) promises in its comments/types vs code reality,
  (d) stubs/dead branches/TODO-shaped gaps, (e) **what is missing** — the capability this
  file implies but doesn't complete, (f) verdict: TRUSTWORTHY / DEFECTIVE(+evidence) /
  UNWIRED / VESTIGIAL.
- **B2 — Enumerated promise inventory** (1 agent): mechanically list EVERY exported
  symbol, option/extension field, FrameInput/Output field, capability grade, and README
  feature claim across all 12 packages; walk each to kept / partial / broken / unreachable.
  Output: a promise × verdict table + the count that becomes the completion denominator.
- **B3 — Behavioral execution** (2 agents): RUN things — never done in any prior audit.
  deno + lavapipe (the wsl-gpu pattern): boot each backend headless on a minimal core
  Scene; render N frames per major feature flag; assert non-black output, no device/
  validation errors, sane luminance. Execute each examples/ app's logic as far as headless
  allows. Run every tools/ script's smoke path. Paste outputs.
- **B4 — Feature-interaction matrix** (1 agent): pairwise flags (spectral×bdpt,
  restirPtReuse×caustics, checkerboard×denoisers, skinning×TLAS, lite×each, gi-state×RC,
  …) — compose + compile through both shader gates per pair; behavioral smoke where cheap;
  table of pair × verdict.
- **B5 — Coverage ground truth** (1 agent): install @vitest/coverage-v8, produce the
  per-file 0%-coverage map (the untrodden-line map), read the worst uncovered regions in
  production files, report what lives in the dark.
- **B6 — Product-cohesion walk** (1 agent): approach the library as a NEW USER with only
  README + examples + public types. Attempt (conceptually + by tracing the API) the ten
  tasks a real adopter would try. Report every point of friction, inconsistency between
  backends, missing entry point, and "I would expect X here" gap. This is the missing-
  promise audit at product altitude.

## Conduct rules (all agents)

Never run git commands that modify state. Read-only everything outside /tmp. Env: prefix
commands with `export PATH="$HOME/.local/bin:$HOME/.nvm/versions/node/v24.13.0/bin:$PATH"`.
Package-local vitest only if needed; never the root suite. Final message = the structured
table(s), nothing conversational.

## Synthesis (lead)

1. Verify per-agent: sample claims, re-run pasted commands, diff B1 tables vs assignments.
2. Completeness critic pass over the coverage accounting.
3. Merge into: the trust report — findings by class (broken/unkept/missing), the
   enumerated completion number (B2 denominator), and the gap list that becomes the next
   implementation rounds.
