# T1.G — Stray cleanup audit

**Date:** 2026-05-11  
**Branch:** feat/sweep-2026-05-12-followup

---

## G1: commit 0f09b63 vs 9f5e9f1

🟢 **status: merely incremental — no revert, no duplication**

**Diff summary:**  
`git diff 0f09b63 9f5e9f1 -- packages/walkaround-hybrid/src/rc/` produces zero output.
`git log 0f09b63..9f5e9f1` shows exactly one commit: `9f5e9f1` itself.
The `--name-only` output for `9f5e9f1` lists only GTAO/atrous/pipeline files
(`gtao.wgsl.ts`, `indirectCombine.wgsl.ts`, `shade.wgsl.ts`, `atrousVarianceWebGPU.ts`,
`WalkaroundGPUPipeline.ts`, `bindGroup*.ts`, `resourceManager.ts`); none of the RC
files touched by `0f09b63` (`octahedralSolidAngles.ts`, `walkaroundDiffuseLighting.ts`,
`cascadeMerge.wgsl.ts`, `cascadePyramid.ts`, `rcSolidAngles.test.ts`) appear.

**Conclusion:** `9f5e9f1` is a direct child of `0f09b63`. The RC work (Items 21+22)
is carried forward unchanged. The orchestrator commit adds only M9.B (GTAO) and
M9.C (atrous demodulation) on top. No RC content was reverted or duplicated.

---

## G2: RFE-09 reconciliation

🟡 **status: naming collision resolved — two distinct RFE-09 entries**

**Current state (before fixes):**  
- `external_requests/09-runtime-lighting-updates.md`: Status: **Proposed** — proposes
  a future `updatePrimaryLight` / `updateSkyDome` / `updateLights` API on `HybridEngine`.
- `external_requests/IMPLEMENTATION-STATUS.md` §"RFE-09": claims **APPLIED** — but refers
  to the *pt-webgl fork uniform bridge* sprint deliverable (`forkUniformBridge.ts`), which
  is a completely different thing filed under the same number.
- `external_requests/README.md` index: RFE-09 was **missing** from the table entirely.
- CHANGELOG (line 58): correctly describes the uniform bridge under "RFE-09".

**Verdict:** The "RFE-09" in IMPLEMENTATION-STATUS.md is the sprint-internal uniform-bridge
deliverable. The file `09-runtime-lighting-updates.md` is a separately filed external RFE
(Status: Proposed, not implemented). They are genuinely different; neither doc is wrong
about its own subject; but the shared number created apparent contradiction.

**Action taken:**  
1. `external_requests/README.md` — added RFE-09 row: `09-runtime-lighting-updates.md |
   Proposed | Runtime lighting updates without pipeline rebuild`.
2. `external_requests/IMPLEMENTATION-STATUS.md` — added inline HTML comment disambiguating
   the "RFE-09" uniform-bridge section from the `09-runtime-lighting-updates.md` proposal.

No CHANGELOG or `09-runtime-lighting-updates.md` status header changes were needed:
the CHANGELOG correctly describes what was applied, and the file's "Proposed" header
correctly reflects its unimplemented state.

---

## G3: M4.D backtick-template-literal fix

🟢 **status: comment-only addition; no WGSL semantic change; pre-existing confirmed**

**Lines changed (verified by reading `git show d97e806 -- packages/walkaround-hybrid/src/shaders/common.wgsl.ts`):**  
The diff for `common.wgsl.ts` in `d97e806` shows:
- Lines 552–553 (in `bvhIntersectAny`): added two comment lines using single-quoted
  `'stackPtr + 1u < 64u'`. These lines did **not exist** in the pre-M4 state (confirmed
  via `git show 0088a78:…common.wgsl.ts | grep Guard` — no output).
- Line 563: changed actual WGSL guard from `stackPtr < 62u` → `stackPtr + 1u < 64u`
  (this is the real correctness fix, Item 29, not the backtick fix).
- Lines 642–643 / 658: same pattern for `bvhIntersectFirstHit`.
- `grep -E '^\+.*\`|^\-.*\`'` on the diff returns zero lines — **no backticks were
  added or removed** in the diff.

**Interpretation of agent claim:** The commit message describes the backtick fix as a "bonus"
resolved as a side effect. Inspecting the diff shows the Guard comments were simply *added
fresh* with single quotes (no prior comment existed to convert). The template-literal hazard
that the agent referred to was likely a comment they drafted but then immediately wrote with
single quotes. Either way: the current file has no backtick-quoted inline code inside the
JS template literal, and the WGSL logic (`stackPtr + 1u < 64u`) is correct.

**Pre-existing?** Yes. `git show 0088a78:…common.wgsl.ts` at line 534 shows `stackPtr < 62u`
with no Guard comment — the backtick risk would have been latent in any future comment
that used backticks. The guard fix is genuine and pre-existing relative to the sweep.

**Parse check:** `npx tsc --noEmit -p packages/walkaround-hybrid/tsconfig.json` returns
one pre-existing unrelated error (`WalkaroundGPUPipeline.ts:780` argument count); zero
errors touching `common.wgsl.ts`. File parses cleanly as a TypeScript module.

**Conclusion:** Only comment text and WGSL guard arithmetic changed. No WGSL semantics
outside the intended stack-guard fix were affected.
