# Berghain Contest Handoff

Date: 2026-06-22
Continuity: repo-state handoff. No owner thread was recovered for this pass.

## Current State

- Branch: `main`
- Remote/upstream: `origin/main` at `https://github.com/1kuna/berghain-contest.git`
- Starting state for this handoff pass: clean and aligned with upstream.
- Latest code commit before this handoff: `5d12f17 Add Bayesian Optimization algorithm (algo2.py)`.
- Product shape from README: phase-based optimizer for the Berghain admissions challenge.

## Last Meaningful Work

Recent commits focused on optimizer performance and algorithm variants:

- index-based storage refactor
- performance optimizations
- JSON/numpy serialization fixes
- Bayesian Optimization algorithm in `algo2.py`

## What Is Not Verified In This Pass

No optimizer run was started while writing this handoff. The README warns to keep workers at `1` for rate limits.

## Resume Steps

1. Review `algo1.py`, `algo2.py`, and README command options.
2. If running the optimizer, start with one scenario and one worker.
3. Preserve any generated `berghain_s*_state.json` progress files outside destructive cleanup.
4. Compare `algo1.py` and `algo2.py` results before making `algo2.py` the default.

## Cautions

- Respect rate limits; do not parallelize blindly.
- This handoff is repo-state-only and should be superseded if historical project context is later recovered.
