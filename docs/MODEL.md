# ArbGap Core: the optimization problem

For one battery at one hub on one day, choose when to buy (charge) and when to sell
(discharge) so profit is as high as possible, without breaking the battery's limits.

## Inputs

| Symbol | Meaning | Default |
|---|---|---|
| p_t | price at hour t ($/MWh), t = 0..H-1 | day-ahead for B, real-time for C |
| H | price steps in a day | 24 |
| η (eff) | share of bought energy that gets stored | 0.85 |
| w (wear) | wear cost per MWh sold | $10 |
| c (trade_cost) | cost of one trade (buy + sell) | $0 |
| P (power_mw) | most the battery can buy or sell in one hour | 1 MW |
| E (capacity_mwh) | most energy the battery can hold | 1 MWh |
| k (max_trades) | trades allowed per day | 1 |

## Current model: one trade a day

Decisions: buy hour b, sell hour s, and x = 1 to trade or x = 0 to skip the day.

    maximize   f(b, s, x) = x · [ η · (p_s − w) − p_b − c ]

    subject to
      1. 0 ≤ b < s ≤ H − 1          buy before sell, both inside the day
      2. buy = P · 1 hour = 1 MWh    one hour of charging at full power
      3. 0 ≤ level_t ≤ E, every t    never more than the battery holds, never below empty
                                     (level after the buy = η · 1 MWh = 0.85 ≤ 1)
      4. trades ≤ k = 1
      5. x ∈ {0, 1}                  skip the day (profit 0) if no trade makes money

Constraint 3 is the "sum of purchases ≤ total storage" rule.

The objective is linear in the prices, but b, s and x are whole numbers, so this is a
small integer (combinatorial) problem. With one trade there are only H(H − 1)/2 = 276
possible plans, so the tool checks all of them.

## Where each rule is in the code

| Rule | How the search keeps it | How it is checked |
|---|---|---|
| 1. buy before sell | `best_trade` only tries `sell` in `range(buy + 1, H)` | `check_plan`: `0 <= buy < sell < hours` |
| 2. power | every trade buys for exactly one hour | `check_plan`: at most P MW per hour, no buy and sell in the same hour |
| 3. capacity | one trade stores 0.85 MWh ≤ 1 MWh | `check_plan` replays the battery level hour by hour |
| 4. trades per day | one pair per day | `check_plan`: `len(trades) <= max_trades` |
| 5. skip | `best_profit` starts at 0, so a losing pair is never picked | an empty plan is always valid |

Every backtest replays all A, B and C plans with `check_plan` (battery.py):
368 hub-days × 3 plans = 1,104 plans. A test also feeds it a "cheating" plan that stores
more than the battery holds, and it gets rejected.

## Next version (planned): several trades a day

With k trades, the plan becomes how much to buy (c_t) and sell (d_t) in every hour,
with the battery level s_t tracked hour by hour:

    maximize   Σ_t [ p_t · (d_t − c_t) − w · d_t ]  −  c · trades

    subject to
      1. s_t = s_(t−1) + η · c_t − d_t                  battery level
      2. 0 ≤ s_t ≤ E                                    capacity (purchases ≤ storage)
      3. 0 ≤ c_t ≤ P and 0 ≤ d_t ≤ P                    power limit
      4. s_0 = s_H = 0                                  start and end the day empty
      5. (1 − η) · Σ_t c_t ≤ L                          max energy lost per day
      6. Σ_t p_t · c_t + w · Σ_t d_t + c · trades ≤ M   max money spent per day
      7. trades ≤ k

With k = 1, c_b = 1 MWh and d_s = 0.85 MWh, this gives back the current model.
Brute force can't check every plan here (see the worst-case section), so this version
will use Google OR-Tools, with brute force kept as a test on small cases.