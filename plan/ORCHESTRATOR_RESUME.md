# ORCHESTRATOR RESUME — gap remediation

> **Parent orchestrator agent:** read this file at the start of every session. Do not stop until `phase: done`.

## State snapshot

```json
{
  "currentWaveId": "W000",
  "currentWaveIndex": 0,
  "phase": "ready",
  "remediationRound": 0,
  "completedWaves": 0,
  "totalWaves": 70,
  "waveBaseSha": null
}
```

## What to do NOW

1. Run: `node tools/gap-scan/orchestrator-run.mjs begin`
2. Run: `node tools/gap-scan/orchestrator-run.mjs dispatch`
3. Launch **8** sub-agents (max 25) using prompts in `plan/waves/W000/agents/`
4. When all return: `node tools/gap-scan/orchestrator-run.mjs --verify`

## Wave loop (every wave)

```
begin → dispatch N sub-agents → verify → [remediate → verify]* → smoke → commit → advance
```

## Progress: 0 / 70 waves committed

No waves committed yet.
