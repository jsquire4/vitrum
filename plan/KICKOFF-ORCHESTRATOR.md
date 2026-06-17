# Kickoff — gap remediation orchestrator

Paste this into a **new Agent chat** opened in the worktree:

```
cd /home/jsquire4/projects/vitrum-gap-remediation

You are the gap remediation ORCHESTRATOR (not a worker).
Read plan/ORCHESTRATOR_AGENT.md and plan/ORCHESTRATOR_RESUME.md first.

Goal: complete ALL scheduled waves until plan/.gap-orchestrator-state.json shows phase: "done".

Rules:
- Work only in this worktree on branch feat/gap-remediation
- Commit after every wave: node tools/gap-scan/orchestrator-run.mjs --commit --yes
- NEVER git push
- Skip deferred validation tasks (plan/VALIDATION-DEFERRED.md) — code first
- Launch ≤25 Task subagents per wave using plan/waves/Wnnn/agents/agent-NN.md
- YOU run --verify and --smoke; do not trust worker completion claims
- On verify fail: remediate and re-verify
- After each advance: one line only — "Wave N/M complete — next Wnnn"
- Do not stop until phase: done unless verify/smoke fails twice on same task

Begin: orchestrator-run.mjs begin → dispatch → workers → --verify → --smoke → --commit --yes → advance → loop
```

## Resume (new session)

```
cd /home/jsquire4/projects/vitrum-gap-remediation
Resume gap orchestrator. Read plan/ORCHESTRATOR_RESUME.md. Continue until phase: done. Same rules as KICKOFF.
```

## Progress check

```bash
node tools/gap-scan/orchestrator-run.mjs status
git log --oneline -10
```
