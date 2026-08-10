# trend-pullback-bot

## V1.5 risk-managed validation

`develop_v15.py` keeps the V1.4 entry rules frozen and tests whether a
$1,000 cash account can trade one contract without allowing a single premium
or estimated stop loss to dominate the account. The development grid varies
delta, profit target, maximum contract allocation, and estimated stop risk on
2024-2025 data. It selects a candidate only if it has at least six trades,
positive account return, profit factor of at least 1.30, and no more than 25%
account drawdown.

The selected policy is then locked and evaluated once on untouched 2022-2023
data. If no development policy meets the requirements, validation is not run.
All option prices remain synthetic Black-Scholes estimates rather than
historical option quotes, and this workflow cannot submit orders.

## V2.1 ETF share validation

`develop_v21.py` pivots away from options and compares three long-only daily
share strategies on SPY, QQQ, and IWM: trend pullbacks, oversold mean
reversion, and high-volume breakouts. It uses integer shares in a $1,000 cash
account, next-session entries, overnight-gap-aware stops, conservative
slippage and per-share costs, a 2% maximum account risk, and an 80% maximum
cash allocation. The best qualifying family on 2010-2017 is locked before one
evaluation on 2018-2025. The workflow is research-only and cannot place an
order.

## V4.7 two-year research funnel

All new strategy families after V4.6 use `research_protocol_v47.py` before any
long historical run. The first screen is fixed to January 2, 2024 through
December 31, 2025. A strategy advances only with at least 20 completed trades,
positive account return, profit factor of at least 1.25, maximum drawdown no
worse than -15%, and positive P&L in both calendar years.

Passing this fast screen unlocks longer historical robustness testing; it does
not constitute validation. Earlier research consumed every historical
holdout, so only the unchanged 2026-forward paper process can provide fresh
validation. V4.6 continues weekly with its rules and start date unchanged.
