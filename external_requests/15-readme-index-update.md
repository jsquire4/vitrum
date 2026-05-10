# RFE-15 — Housekeeping: Update external_requests/README.md Index

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED. `external_requests/README.md` now includes RFEs 01–15 and reflects
post-implementation status updates for the currently landed RFEs.

---

## What this request is for

Update `external_requests/README.md` to reflect the actual implementation status of all
RFEs and to add index entries for the 2026-05-09 batch (06–15). The README is the canonical
project status board for outside consumers; stale statuses compound as more RFEs land.

The README itself describes the intended workflow ("Triage and set Status in the table").
This RFE requests that triage.

---

## Affected file

- `external_requests/README.md` — the index table and status values.

---

## Required changes

### Status corrections for existing RFEs

| # | Current Status | Correct Status | Basis |
|---|---|---|---|
| 01 | Proposed | Partial | Sprint 12 spectral accumulator APPLIED (commit `8917492`); full Beer-Lambert blocked on RFE-13 |
| 02 | Proposed | Applied | Sprint 7 volume scattering APPLIED (commit `260c432`); TRANSLUCENT_BIT packing deferred (RFE-11) |
| 03 | Proposed | Proposed | Three-bindings propagation done; no fork plan doc yet (RFE-12) |
| 04 | Proposed | Partial | Sprint 12 helper functions APPLIED; TMM evaluator NOT STARTED (RFE-14); blocked on RFE-13 |
| 05 | Proposed | Proposed | No fork shader work; API-complete in pt-webgl only |
| 06 | (absent) | Applied | Sprint 8 dispersion APPLIED (commit `7ffd15d`); uniform bridge missing (RFE-09) |
| 07 | (absent) | Applied | Sprint 7 volume scattering APPLIED (commit `260c432`) |
| 08 | (absent) | Partial | Sprint 12 APPLIED partially; payload restructure deferred (RFE-13) |

### New index entries (09–15)

| # | Document | Status | Topic |
|---|---|---|---|
| 09 | 09-pt-webgl-material-uniform-bridge.md | Proposed | pt-webgl: drive fork uniforms from vitrum.Material fields |
| 10 | 10-three-bindings-userdata-propagation.md | Closed | three-bindings userData reads — DONE; remaining work in RFE-09 |
| 11 | 11-fork-translucent-bit-materialstexture-packing.md | Proposed | Fork: TRANSLUCENT_BIT packing in MaterialsTexture.js |
| 12 | 12-vitrum-layered-bsdf-fork-patch-plan.md | Proposed | Request fork-patch plan doc for RFE-03 (layered BSDF) |
| 13 | 13-fork-sprint12-ray-payload-restructure.md | Proposed | Fork: vec3 throughput → float wavelength + float throughput |
| 14 | 14-fork-thinfilm-tmm-35layer.md | Proposed | Fork: 35-layer TMM BSDF evaluator (blocked on RFE-13) |
| 15 | 15-readme-index-update.md | Proposed | Housekeeping: update this index |

---

## Acceptance criteria

- [ ] `external_requests/README.md` index table contains rows 01–15.
- [ ] Statuses for 01–08 are corrected per the table above (or updated to reflect any
  implementation work that occurred after 2026-05-09).
- [ ] Status values used conform to the README's own legend: `Proposed` → `Accepted` →
  `Implemented` | `Deferred` | `Rejected`.
- [ ] The "Quick link: plan mapping" section references the new sprint plan docs for
  Sprints 7, 8, and 12 (`plan/sprint-7-pt-fork-patch.md`, `plan/sprint-8-pt-fork-patch.md`,
  `plan/sprint-12-pt-fork-patch.md`) if not already linked.

---

## References

- `external_requests/README.md` — the file to update.
- `IMPLEMENTATION-STATUS.md` — the authoritative source of truth for which fork commits
  applied which patches.
- `SPRINT_12_GAPS.md` — deferred items for Sprint 12.

---

## Out-of-scope notes

- This RFE does NOT change the content of any RFE doc, only the index.
- It does NOT add new RFE rows beyond 15 — the housekeeping is limited to catching up
  with what has already been dropped as of 2026-05-09.
- Status values for 09–15 are "Proposed" at drop time; vitrum's authors set the final
  triage status during their normal triage cycle.

---

## Consumer-side state

The stainedGlass app does not consume the README directly, but its developers use it as
a cross-reference when writing new request docs. An accurate index reduces the risk of
duplicate RFEs and gives the app team a reliable view of what is and isn't actionable on
the vitrum side.
