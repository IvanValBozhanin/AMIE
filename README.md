# AMIE

[![CI](https://img.shields.io/badge/CI-placeholder-lightgrey)](#)

Algorithmic Market Intelligence Engine toolkit.


## Status

v0.1 (in progress)


# AMIE-Lite (v0.1) — Executive Summary (Prototype / In Progress)

**Repo:** IvanValBozhanin/AMIE  
**Stage:** Prototype / In Progress

## TL;DR
- Purpose: deterministic microstructure sandbox that wires synthetic LOB data through engineered features, a two-state Kalman signal, and a transparent policy/backtest stack for explainable experimentation.
- Current state: `scripts/compute_features.py` + `scripts/train_model.py` run end-to-end on seeded synthetic ticks, persist Parquet features, train the Kalman filter, emit predictions/metadata, and unit/integration tests assert determinism, risk guards, and metrics plumbing (`scripts/compute_features.py:50-106`, `scripts/train_model.py:79-200`, `tests/integration/test_model_pipeline.py:30-214`).
- Limitation + next action: everything is still synthetic with a global z-score that can leak future info, so the immediate task is to retrofit look-ahead-safe rolling stats and validate on at least one recorded market day before treating metrics as signal (`src/amie/features/transforms.py:15-22`).

## What It Does (Current Capabilities)
- **Data:** A seeded `SyntheticLOBGenerator` streams regime-switching ticks with reproducible timestamps, spreads, and depth while honoring config defaults (BTC-USD, 1s cadence, regime vol 0.01/0.05, spreads 25/35) (`src/amie/data/sources/synthetic_lob.py:1-182`, `config/data/synthetic.yaml:1-13`). Contracts enforce monotonic time, positive qty, and determinism for golden fixture generation.
- **Features:** `FeatureComputer` derives five microstructure features—log returns, EWMA volatility, global z-score with zero-variance guards, normalized spread, and book imbalance—after sorting by timestamp and keeping warm-up NaNs; tests confirm no NaNs post-window and symmetry/zero-variance behavior (`src/amie/features/transforms.py:57-120`, `tests/unit/test_features.py:31-92`).
- **Model:** The only registered model is a two-state Kalman filter tracking level/trend with adaptive Q/R inflation, uncertainty previews, and auto-fit warmup; unit and integration tests verify convergence, uncertainty behavior under volatility shifts, determinism, and <200 ms latency for 1k ticks (`src/amie/models/kalman.py:1-200`, `tests/unit/test_kalman.py:30-103`, `tests/performance/test_model_latency.py:28-70`, `tests/integration/test_model_pipeline.py:30-192`).
- **Policy:** `SignalPolicy` converts score/uncertainty pairs into ±1 targets gated at `threshold_multiplier * uncertainty` (default 2×) and clamps to `max_position_size` (default 1); tests cover long/flat/capped cases and sensitivity to uncertainty (`src/amie/strategy/policy.py:36-105`, `tests/unit/test_policy.py:31-64`).
- **Backtest:** `BacktestEngine` runs a single-asset, chronological loop that enforces sorted features, matches one signal per row, debits explicit slippage/fee costs via `ExecutionSimulator`, and feeds a `RiskManager` that caps positions and forces flat once drawdown breaches (`src/amie/backtest/engine.py:66-139`, `src/amie/strategy/execution.py:18-88`, `src/amie/strategy/risk.py:15-96`, `config/backtest/default.yaml:1-4`). Unit tests assert equity increases under positive returns, flat behavior under zero signals, fee-only losses when returns are zero, and drawdown caps (`tests/unit/test_backtest.py:76-140`).
- **Metrics:** `BacktestMetrics` exposes Sharpe, Sortino, max drawdown, Calmar, hit rate, and turnover with annualization defaulting to 252; tests cover expected values, NaN handling, and JSON/Parquet persistence for downstream reporting (`src/amie/backtest/metrics.py:31-155`, `tests/unit/test_metrics.py:21-86`).

## Architecture at a Glance
- **Flow:** Synthetic LOB → FeatureComputer → Kalman signals/uncertainty → SignalPolicy + RiskManager + ExecutionSimulator → BacktestEngine → BacktestMetrics/CSV outputs (`scripts/compute_features.py:50-106`, `scripts/train_model.py:79-200`, `src/amie/backtest/engine.py:66-139`, `src/amie/backtest/metrics.py:31-146`).
- **Key Files:** `scripts/compute_features.py` hydrates the pipeline and validates via Pandera before persisting features (`scripts/compute_features.py:78-100`); `scripts/train_model.py` loads Parquet runs, trains/predicts, and writes model/prediction artifacts with RMSE/MAE stats (`scripts/train_model.py:79-200`); `src/amie/models/kalman.py` houses the adaptive filter; `src/amie/strategy/{policy,risk,execution}.py` implement gating, drawdown caps, and deterministic slippage; `src/amie/backtest/engine.py` ties components; `src/amie/backtest/metrics.py` computes risk stats.
- **Determinism:** Each module documents CONTRACT invariants, hydration scripts bind `run_id`/seed, golden artifacts can be regenerated (`scripts/generate_golden_files.py:1-41`), and integration tests confirm identical signals for repeated runs with seed 42 (`tests/integration/test_model_pipeline.py:194-214`).
- **No Look-Ahead:** Features sort and only use past prices for returns while the backtest re-sorts features before looping (`src/amie/features/transforms.py:57-69`, `src/amie/backtest/engine.py:149-179`). Caveat: the global z-score uses full-sample stats (explicit TODO) and should be replaced with rolling stats before live trading (`src/amie/features/transforms.py:15-22`).

## Key Engineering Decisions (and Rationale)
- Vectorized backtest keeps state in simple floats for transparency and easy unit testing instead of an event-driven engine, enabling deterministic assertions on equity, fees, and risk caps (`src/amie/backtest/engine.py:66-139`, `tests/unit/test_backtest.py:76-140`).
- Kalman filter baseline favors explainability and uncertainty estimates over more complex ML; adaptive inflation reacts to innovation spikes without hidden hyperparameters (`src/amie/models/kalman.py:59-198`).
- Policy thresholds scale decisions by model uncertainty to avoid overstating conviction and to keep drawdown logic simple for hiring reviewers (`src/amie/strategy/policy.py:36-105`).
- Trading costs are modeled explicitly (slippage_bps + half-spread + fee_bps) and centralized in configs/ExecutionSimulator so fee-only loss cases are testable (`src/amie/strategy/execution.py:18-88`, `config/backtest/default.yaml:1-4`, `tests/unit/test_backtest.py:110-126`).
- Feature storage and metadata hashing rely on `FeatureStore` to keep schema/partition history reproducible for audits (`src/amie/features/store.py:28-155`, `tests/unit/test_feature_store.py:31-92`).

## Reproducibility: How to Run
```bash
poetry install
poetry run python scripts/compute_features.py run.id=demo-lite data.num_ticks=1000 run.seed=42
# use the emitted run_id (e.g., demo-lite) as training.feature_run_id below
poetry run python scripts/train_model.py run.id=kalman-lite training.feature_run_id=demo-lite
poetry run pytest tests/unit/test_backtest.py tests/integration/test_model_pipeline.py
```
- **Outputs:** Feature runs land in `data/features/<run_id>` with partition metadata (`scripts/compute_features.py:78-100`); training writes `<run_id>_model.pkl`, `<run_id>_predictions.csv`, and `<run_id>_metadata.json` under `results/` with RMSE/MAE/uncertainty stats (`scripts/train_model.py:167-200`).
- **Config constants:** Synthetic feed defaults (seed 42, 1 s cadence, vol 0.01/0.05) live in `config/data/synthetic.yaml:1-13`; model defaults (process_noise 1e-3, observation_noise 1e-2, warmup 100) in `config/model/kalman.yaml:1-4`; backtest costs (initial_capital 100k, slippage 5 bps, fee 1 bps, max position 1) in `config/backtest/default.yaml:1-4`; policy defaults (threshold 2× uncertainty, cap 1) in `src/amie/strategy/policy.py:36-105`; metrics annualization is 252 (`src/amie/backtest/metrics.py:31-146`).

## Results Snapshot (Prototype Data)
- **PnL/Equity:** No canonical equity curves are checked in; behavior is validated via unit tests (positive returns lift equity, zero signals stay flat, zero returns lose only fees) rather than published figures (`tests/unit/test_backtest.py:76-126`). Unknown (code review needed) for real-market performance.
- **Risk/Metrics:** `BacktestMetrics` can emit Sharpe/Sortino/max drawdown/turnover, and training metadata stores RMSE/MAE relative to synthetic returns, but no sample tables or plots are versioned; rerun scripts to regenerate (`src/amie/backtest/metrics.py:31-155`, `scripts/train_model.py:167-200`).
- **Known failure modes:** Trend-following Kalman scores degrade on regime shifts or volatility spikes not captured in the two synthetic regimes (`config/data/synthetic.yaml:1-13`, `tests/unit/test_kalman.py:60-73`); the global z-score includes future returns (`src/amie/features/transforms.py:15-22`); spread spikes widen slippage because execution adds half-spread plus bps cost (`src/amie/strategy/execution.py:39-88`).

## Limitations
- Data is entirely synthetic; there is no real-market ingestion, replay, or sanity checks beyond Pandera schema validation, so distributional drift and microstructure oddities remain untested (`scripts/compute_features.py:78-100`, `src/amie/data/validation.py:14-116`).
- Feature normalization uses whole-sample statistics, which leaks future information and is explicitly called out as a TODO; no shift tests prevent this today (`src/amie/features/transforms.py:15-22`).
- Strategy stack is single-asset with linear costs and no borrow, funding, or inventory penalties; TODOs in the backtest warn that monetary units need confirmation before interpreting PnL (`src/amie/backtest/engine.py:18-23`).
- Model registry contains only the Kalman filter; there is no cross-validation, walk-forward split, or alternative baseline to guard against overfitting or structural misses (`src/amie/models/registry.py:6-32`).
- Tests cover deterministic flows, but there are no “golden seed” CI gates ensuring CLI scripts regenerate the exact same Parquet/CSV that hiring managers might review (`tests/integration/test_model_pipeline.py:30-214`).

## Near-Term Roadmap
- **T+1–2 weeks:** Document CLI contracts for `scripts/compute_features.py` and `scripts/train_model.py`, then add micro-tests for (1) feature warm-up NaNs, (2) explicit no-look-ahead shift assertions, and (3) zero-return fee-only losses directly on the real pipeline outputs instead of stubs (`src/amie/features/transforms.py:57-109`, `tests/unit/test_backtest.py:110-126`). Produce a single notebook that loads a run’s Parquet, plots equity + underwater curve, and embeds the metrics table generated by `BacktestMetrics`.
- **T+2–3 weeks:** Replace the global z-score with rolling statistics plus CI guardrails, add parameter sweeps for `slippage_bps`/`fee_bps`/`threshold_multiplier`, run walk-forward/backwardation experiments, and promote deterministic seeds to smoke tests so Hydra runs fail fast when reproducibility drifts (`src/amie/features/transforms.py:15-22`, `config/backtest/default.yaml:1-4`, `tests/integration/test_model_pipeline.py:30-214`). Layer in simple failure-mode detectors (e.g., clip positions when imbalance or volatility exceed percentiles) before revisiting more complex models.

## Appendix: File Map & Constants
- `src/amie/data/sources/synthetic_lob.py:1-182` — deterministic synthetic LOB with base_price 50 000, regime_duration 250+ and Side enum outputs.
- `scripts/compute_features.py:50-106` — Hydra CLI to generate/validate/store features; defaults seed 42, window_size 20.
- `src/amie/features/transforms.py:57-120` — feature derivations (returns, EWMA vol, z-score, spread, imbalance) and TODO on global stats; assures sorted frames.
- `src/amie/models/kalman.py:1-200` — two-state Kalman with adaptive inflation, warmup defaults (process_noise 1e-3, observation_noise 1e-2, warmup 10/100 via config), and uncertainty preview.
- `src/amie/strategy/{policy.py:36-105,risk.py:15-96,execution.py:18-88}` — policy thresholds (2× uncertainty, cap 1), drawdown cap default 20%, deterministic slippage = half-spread + bps + fee_bps.
- `src/amie/backtest/engine.py:66-139` and `src/amie/backtest/metrics.py:31-155` — sequential backtest with initial_capital 100 000, explicit trade costs, and metrics (Sharpe, Sortino, max DD, Calmar, hit rate, turnover with annualization=252).
- `config/data/synthetic.yaml:1-13`, `config/model/kalman.yaml:1-4`, `config/backtest/default.yaml:1-4`, `config/training/default.yaml:1-4` — centralized defaults for regimes, model hyperparameters, trading costs, and data splits.

Prototype / In Progress — expect rapid iteration.
