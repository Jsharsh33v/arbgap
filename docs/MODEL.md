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

## Several trades a day (multi.py)

A bigger battery holds up to U charges, where one charge is one 1 MWh buy
(U = E / (η · P), rounded down; a 2 MWh battery holds 2). It makes up to k trades a day.
For every hour t the plan picks one action:

    buy_t  ∈ {0, 1}     buy 1 MWh in hour t
    sell_t ∈ {0, 1}     sell one stored charge (η MWh) in hour t
    level_t             charges stored after hour t

    maximize   Σ_t [ sell_t · η · (p_t − w)  −  buy_t · (p_t + c) ]

    subject to
      1. buy_t + sell_t ≤ 1                       one action per hour (power limit)
      2. level_t = level_(t−1) + buy_t − sell_t   battery level; it starts at 0
      3. 0 ≤ level_t ≤ U                          never below empty, never over capacity
      4. level_(H−1) = 0                          end the day empty
      5. Σ_t buy_t ≤ k                            trades per day

With U = 1 and k = 1 this is the one-trade model above; a test checks that both give the
same profit. Every choice is on or off, so the problem is combinatorial. There is no slope
to follow, which is why gradient descent does not apply.

Not built yet: the two daily limits from the first plan (max energy lost per day and max
money spent per day).

## The four methods and their worst case

| Method | How it works | Always the best plan? | Worst-case work |
|---|---|---|---|
| Brute force `best_plan_brute` | tries every valid hour-by-hour plan | yes | up to 3^H plans. One day has 277 plans for U = 1, k = 1; 21,529 for U = 2, k = 2; 1,941,773,345 for U = 4, k = 8 |
| Dynamic programming `best_plan_dp` | keeps the best profit for each state (charges stored, trades used) after every hour | yes | O(H · U · k) |
| CP-SAT `best_plan_cpsat` | gives the five rules to Google OR-Tools, which searches and proves the best plan | yes, when the status is OPTIMAL | 3H variables; exponential in the worst case, fast in practice |
| Hill climbing `hill_climb` | starts with no trades and keeps making the one change that adds the most profit | no | `add` and `split`: at most k steps, each checks O(H²) hour pairs. `swap` also retries every trade in each step |

Every plan from every method is replayed by `check_plan` (battery.py).

## Hill climbing's neighbor functions

A neighbor is a plan that is one change away from the current plan. `neighbors` picks
which changes are allowed:

- `add`: add one trade.
- `swap`: add one trade, or replace one trade with a better one.
- `split`: add one trade, or sell early and buy back later while the battery holds energy.

Example with prices 10, 100, 20, 110, U = 1, k = 2. `add` and `swap` take the biggest
trade (buy hour 0, sell hour 3, $75.00) and stop, because it fills the battery all day.
`split` then sells at hour 1 and buys back at hour 2. That gives two trades worth $131.50,
which is the best plan.

Results on the real data (`python compare_algorithms.py`): on 736 price-days with U = 4
and k = 8, `add` and `swap` found the best plan on 721 days and `split` on all 736.
Planning 14 Houston days at once, `add` and `swap` earned 53.9% of the best profit and
`split` earned all of it. On 92 days `split` was the slowest method. `split` matched the
exact answer in every case tested, which is evidence, not a proof.