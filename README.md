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
