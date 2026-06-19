# Road to 100 Gap Ledger — 2026-06-11

This file is the compact source-of-truth index for the Road-to-100 campaign.
The detailed ledger remains [road-to-100.md](./road-to-100.md); this file keeps
the named `road-to-100-gap-ledger-2026-06-11.md` artifact present and
machine-checkable so handoffs do not depend on memory or chat summaries.

```json road-to-100-ledger.v1
{
  "schema": "vitrum.road-to-100.gap-ledger.v1",
  "ledgerDate": "2026-06-11",
  "currentAsOf": "2026-06-19",
  "status": "active",
  "canonicalDetail": "plan/road-to-100.md",
  "historicalBugLedger": "items_to_fix.md",
  "sourceCheck": "npm run road-to-100-source-check",
  "proofUmbrella": "npm run proof-check",
  "closedContractCampaigns": [
    "glTF predictable API plumbing, selected-scene compatibility scoping, and texture readiness diagnostics",
    "backend structured warning/error surfaces for known degradation paths",
    "walkaround material atlas, alpha, emitter, and mutation truthfulness tails",
    "pt-webgpu direct-light scoped inverse replay with explicit downgrade diagnostics",
    "pt-webgl2 scalar/material/emitter/environment mutation and warning specificity",
    "engine glTF subpath, progressive handoff, RAF controller playback, and recreate scene retention"
  ],
  "openPromotionBuckets": [
    "GPU material-furnace and reference-render sweeps",
    "real glTF golden asset sweep including real Draco/meshopt samples",
    "GRIS/ReSTIR-GI/PPG/NRC/neural/BDPT quality and radiometric A/B evidence",
    "browser and real-adapter validation outside WSL GPU smoke coverage",
    "explicit support-vs-permanent-unsupported decisions for displacement, spectral, full transparent transport, production neural checkpoint, and NRC default tier"
  ],
  "requiredGreenGates": [
    "npm run typecheck",
    "npm test",
    "npm run shader-gate",
    "npm run proof-check",
    "npm run validate:gpu:smoke after render-changing backend work"
  ]
}
```

## Reading Rules

- Treat source code and tests as the implementation truth.
- Treat [road-to-100.md](./road-to-100.md) as the detailed gap ledger.
- Treat [items_to_fix.md](../items_to_fix.md) as historical audit provenance plus
  any explicitly marked open bug rows.
- Do not promote a row to closed from this file alone. A row is closed only when
  code, tests, and the detailed Road entry agree.

## Current Verdict

The contract/API implementation campaign is in the tail. The remaining Road work
is mostly proof and promotion breadth, plus explicit decisions for rows that are
research-grade or intentionally unsupported. This ledger is not a GA completion
claim.
