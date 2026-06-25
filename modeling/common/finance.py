"""
FINANCIAL ENGINE (v2) — from "how risky?" to "should we make this loan, and at what price?"

The risk models tell us three things about a loan applicant:
  * PD  = Probability of Default  -> the chance the borrower stops paying.
  * LGD = Loss Given Default      -> if they stop paying, the share of the money we DON'T
                                     get back (e.g. 0.95 means we lose 95 cents on the dollar).
  * EAD = Exposure at Default     -> how much they still owe at the moment they stop paying.

This file turns those risk numbers, plus the loan terms (amount, length, interest rate),
into the three numbers a lender actually makes decisions with:

  #3  LIFETIME ECL  - the dollar loss we expect over the whole life of the loan, in today's
                      money. "ECL" = Expected Credit Loss. This is the number a bank sets
                      aside as a reserve. Use it to know how much a loan is likely to cost us.

  #2  EXPECTED PROFIT (NPV) + RAROC - the dollars we expect to make on the loan after losses
                      and costs, and the return on the safety cushion we set aside. Use it to
                      decide whether the deal is worth doing.

  #1  RISK-BASED PRICE - the interest rate (APR) we'd have to charge to (a) just break even,
                      and (b) earn our target return. Use it to quote a fair price for the risk.

A few everyday-language ideas this code leans on:
  * AMORTIZATION: a normal loan is paid back in equal monthly payments. Early on, most of the
    payment is interest; later, most is paying down the balance. So the amount still owed
    (the EAD) shrinks every month.
  * DISCOUNTING / "today's money": a dollar you'll get in 3 years is worth less than a dollar
    today (you could invest today's dollar). So we shrink future dollars back to today's value
    before adding them up. The shrink rate we use is the loan's own interest rate.
  * CAPITAL: regulators and good sense say a lender must keep a cushion of its own money set
    aside per loan, in case losses come in worse than expected. We hold a small % of the loan.
  * RAROC = Risk-Adjusted Return On Capital = profit divided by that cushion. It's basically
    "return on investment," but the investment is the safety cushion. We compare it to a target.

IMPORTANT SIMPLIFICATIONS (honest about what v2.0 does and does NOT do):
  * WHEN defaults happen: the model gives one lifetime PD (a single chance of ever defaulting).
    We spread that single number across the months using a "timing curve" learned from real
    data (most defaults happen partway through the loan, not on day one). A fuller approach
    would model each month's risk directly; that's a later upgrade.
  * EAD each month = the scheduled balance still owed (from the amortization table). Simple and
    standard, but it ignores that some borrowers prepay or fall behind early.
  * We discount everything at the loan's interest rate, for both losses and profit, to keep one
    consistent rule.
  * PREPAYMENT (paying the loan off early) is NOT modeled yet. Because some borrowers pay off
    early, our interest income is slightly on the optimistic side. This is a known next step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# The default-timing curve is learned from real data by build_default_timing.py and saved here.
TIMING_PATH = REPO_ROOT / "models" / "default_timing.json"


@dataclass
class EconomicAssumptions:
    """The lender's business assumptions. These are knobs you can change, not facts about
    the borrower. Defaults below are reasonable round numbers for a consumer lender."""

    funding_rate: float = 0.04      # What it costs US, per year, to get the money we lend out
                                    # (e.g. 4%). We pay this on whatever is still owed.
    servicing_rate: float = 0.01    # Yearly cost to run/collect the loan (statements, support),
                                    # as a % of the balance (e.g. 1%).
    capital_ratio: float = 0.08     # Safety cushion of our own money per loan (e.g. 8% of the
                                    # loan amount). This is the "investment" RAROC is measured on.
    target_raroc: float = 0.15      # The yearly return we WANT to earn on that cushion (e.g. 15%).
                                    # The recommended price is the rate that hits this.


def load_timing() -> tuple[np.ndarray, int]:
    """Load the "default-timing curve": of all loans that eventually go bad, WHEN in their life
    do they go bad? It's a shape over the loan's life from 0% (just started) to 100% (paid off).

    Built from real Prosper defaults by build_default_timing.py. If that file isn't there yet,
    we fall back to a sensible bump that peaks about a third of the way through the loan's life.
    """
    if TIMING_PATH.exists():
        d = json.loads(TIMING_PATH.read_text())
        return np.asarray(d["density"], dtype=float), int(d["bins"])
    # Fallback shape: a gentle hump peaking ~30% through the loan's life (defaults build up,
    # then taper off). Only used if the data-built curve is missing.
    bins = 20
    x = (np.arange(bins) + 0.5) / bins
    dens = np.exp(-((x - 0.30) ** 2) / (2 * 0.18 ** 2))
    return dens / dens.sum(), bins


def timing_weights(term: int, density: np.ndarray, bins: int) -> np.ndarray:
    """Spread the loan's life across its months and read the timing curve at each month.

    Returns one weight per month, and the weights add up to 1. Think of it as: "IF this loan
    defaults, here's the chance it happens in month 1, month 2, ...". A 60-month loan and a
    12-month loan share the same overall shape, just stretched to fit their length.
    """
    # Where each month sits in the loan's life, as a fraction from 0 to 1 (month 1 of 12 ~ 0.04).
    frac = (np.arange(1, term + 1) - 0.5) / term
    # Look up which "bin" of the timing curve each month falls into.
    idx = np.clip((frac * bins).astype(int), 0, bins - 1)
    w = density[idx]
    s = w.sum()
    # Rescale so the monthly weights add to exactly 1 (a clean probability split over the months).
    return w / s if s > 0 else np.full(term, 1.0 / term)


def amortization(amount: float, annual_rate: float, term: int):
    """Build the month-by-month payment table for a normal fixed-payment loan.

    Returns, for each month:
      * balance_start - how much is still owed at the START of the month (this is the EAD: what
                        we'd be exposed to if the borrower defaulted that month),
      * interest      - the interest portion of that month's payment,
      * principal     - the part of the payment that actually reduces the balance,
    plus the fixed monthly payment amount.
    """
    c = annual_rate / 12.0  # monthly interest rate (the yearly rate split into 12)
    # The standard fixed-payment formula. (If the rate is ~0, payment is just amount / months.)
    payment = amount / term if c <= 1e-9 else amount * c / (1 - (1 + c) ** (-term))
    bal = amount
    start, interest, principal = [], [], []
    for _ in range(int(term)):
        i = bal * c                # interest charged this month on the remaining balance
        p = min(payment - i, bal)  # the rest of the payment pays down the balance (capped so we
        start.append(bal)          #   never pay below zero on the final month)
        interest.append(i)
        principal.append(p)
        bal -= p                   # balance shrinks for next month
    return np.asarray(start), np.asarray(interest), np.asarray(principal), float(payment)


def project(amount, annual_rate, term, pd_life, lgd, assum=None, timing=None, marg_pd=None) -> dict:
    """Run the full month-by-month projection for ONE loan at ONE offered interest rate.

    Inputs: loan amount, the APR we'd charge, the term in months, the model's lifetime PD and
    LGD, and our business assumptions. Output: lifetime ECL (#3), expected profit + RAROC (#2),
    and a few supporting figures. Use this to evaluate a specific loan at a specific price.

    `marg_pd` (optional): a per-month marginal-default-probability vector from the hazard model
    (modeling/survival/term_structure.py) — a borrower-specific term structure. When given, it is
    used directly and `pd_life`/`timing` are ignored. When omitted, we fall back to splitting the
    single lifetime PD across the months with the empirical timing curve (the v2 behavior).
    """
    assum = assum or EconomicAssumptions()
    term = int(term)

    # Payment table: how much is owed each month, and how each payment splits.
    bal_start, interest, _principal, payment = amortization(amount, annual_rate, term)

    # marg_pd[t] = chance this loan defaults specifically in month t.
    if marg_pd is not None:
        # Model term structure (from term_structure.marginal_pd): it already encodes both
        # how-likely and when, so pd_life isn't needed. Defensively conform it to `term` — normally
        # a no-op, since the hazard curve is built at this exact term. Be explicit about the two
        # mismatch cases: truncating drops tail-month default mass; zero-padding assumes no default
        # risk past the supplied curve. Either only bites if a caller passes a wrong-length vector.
        m = np.asarray(marg_pd, dtype=float)
        marg_pd = m[:term] if m.size >= term else np.concatenate([m, np.zeros(term - m.size)])
    else:
        # Empirical fallback: one lifetime PD spread across the months by the timing curve (adds to 1).
        density, bins = timing if timing is not None else load_timing()
        marg_pd = pd_life * timing_weights(term, density, bins)
    # surv_start[t] = chance the loan is still alive (hasn't defaulted) at the start of month t.
    # It starts near 1 and drops as the cumulative chance of having defaulted builds up.
    surv_start = 1.0 - np.concatenate([[0.0], np.cumsum(marg_pd)[:-1]])

    # Discount factor: shrink each future month's dollars back to today's money, using the
    # loan's own monthly rate. Month 1 barely shrinks; far-future months shrink a lot.
    c = annual_rate / 12.0
    disc = 1.0 / (1 + c) ** np.arange(1, term + 1)
    ead_t = bal_start  # what we'd lose exposure to if default happens this month = balance owed

    # #3 LIFETIME ECL: for each month, (chance of default) x (share lost) x (amount owed),
    # shrunk to today's money, then added up over the whole loan.
    ecl = float((marg_pd * lgd * ead_t * disc).sum())

    # Interest we expect to collect: we only earn interest in months the loan is still alive,
    # so weight by survival; then shrink to today's money.
    income = float((surv_start * interest * disc).sum())
    # What it costs us to fund the money still owed, each month it's alive (today's money).
    funding = float((surv_start * (assum.funding_rate / 12.0) * ead_t * disc).sum())
    # What it costs us to service the loan, same idea.
    servicing = float((surv_start * (assum.servicing_rate / 12.0) * ead_t * disc).sum())

    # #2 EXPECTED PROFIT (NPV): what we collect, minus expected losses, minus our costs.
    profit = income - ecl - funding - servicing
    capital = assum.capital_ratio * amount  # the safety cushion we tie up for this loan
    years = term / 12.0
    # RAROC = profit as a yearly % return on that cushion. Divide by years to annualize a
    # multi-year loan (a $150 profit over 5 years is a smaller yearly return than over 1 year).
    raroc = (profit / capital) / years if capital > 0 and years > 0 else float("nan")
    return {
        "lifetime_ecl": ecl,
        "interest_income_pv": income,
        "funding_cost_pv": funding,
        "servicing_cost_pv": servicing,
        "expected_profit": profit,
        "capital": capital,
        "raroc": raroc,
        "monthly_payment": payment,
    }


def _solve_rate(target_profit, amount, term, pd_life, lgd, assum, timing, lo=0.0, hi=0.60,
                marg_pd=None):
    """Find the interest rate that produces a given profit, by guess-and-narrow (bisection).

    Charging more always earns more profit (the borrower's risk PD/LGD/term-structure doesn't
    change with the price we quote — that's why we can solve for price cleanly). So we repeatedly
    try the midpoint of a rate range and shrink the range toward the answer. 60 rounds is plenty
    precise. Searches between 0% and 60% APR; if even 60% isn't enough, it returns ~60%.
    """
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = project(amount, mid, term, pd_life, lgd, assum, timing, marg_pd)["expected_profit"]
        # If this rate makes too little profit, the answer is higher; otherwise it's lower.
        lo, hi = (mid, hi) if p < target_profit else (lo, mid)
    return (lo + hi) / 2.0


def price(amount, term, pd_life, lgd, assum=None, timing=None, marg_pd=None) -> dict:
    """#1 RISK-BASED PRICE: what interest rate should we charge for this risk?

    Returns two rates:
      * breakeven_apr     - the rate where we make exactly zero profit (charge less and we lose
                            money on average). The floor.
      * target_raroc_apr  - the rate that earns our target return on the safety cushion. The
                            rate we'd actually want to quote.
    Use this to turn a risk assessment into a fair price, or to sanity-check a rate someone
    proposed (is it above break-even? does it clear our hurdle?).
    """
    assum = assum or EconomicAssumptions()
    timing = timing if timing is not None else load_timing()
    capital = assum.capital_ratio * amount
    years = term / 12.0
    # Break-even = the rate where expected profit is exactly $0.
    breakeven = _solve_rate(0.0, amount, term, pd_life, lgd, assum, timing, marg_pd=marg_pd)
    # Target = the rate where profit equals (target yearly return) x (cushion) x (years), i.e.
    # where the annual RAROC equals our hurdle.
    target = _solve_rate(assum.target_raroc * capital * years,
                         amount, term, pd_life, lgd, assum, timing, marg_pd=marg_pd)
    return {"breakeven_apr": breakeven, "target_raroc_apr": target,
            "target_raroc": assum.target_raroc}
