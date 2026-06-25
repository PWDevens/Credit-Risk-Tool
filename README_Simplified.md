# Credit Risk Scorecard — The Plain-Language Version

This project is a tool that helps a lender answer one money question: **if we give this
person a loan, how much should we expect to lose?**

This page explains what the tool does and how it works. You don't need any finance or computer
background. Every special word is explained the first time it shows up. (For the technical
version, see [README.md](README.md).)

---

## The problem: will the loan get paid back?

When a lender gives someone a loan, most people pay it back over time. Some don't. When a
borrower stops paying, we say the loan has **defaulted** — the borrower failed to pay back what
they owe. When that happens, the lender loses money.

A lender can't know for sure who will pay and who won't. But they can make a smart estimate from
experience. That's what this tool does: it looks at a borrower and the loan, and it estimates
the risk before the lender hands over any money.

---

## The big idea: Expected Loss

The whole project is built around one piece of math that lenders use every day. In plain words:

> **Expected Loss = (how likely they stop paying) × (how much they still owe when they stop) ×
> (the share of that money we can't get back)**

Those three pieces have names. Here's each one in everyday terms:

- **PD — Probability of Default.** The chance the borrower stops paying, written as a percent.
  A PD of 5% means about 5 out of 100 similar borrowers will default.
- **EAD — Exposure at Default.** The dollar amount the borrower still owes at the moment they
  stop paying. Early in a loan they owe more; near the end they owe less.
- **LGD — Loss Given Default.** Of the money they still owe, the share the lender *can't* get
  back, even after trying to collect. If you recover 40 cents on the dollar, your LGD is 60%.

Multiply the three together and you get the **Expected Loss** — the average dollar loss you'd
expect from a loan like this one.

### A quick example

Say someone wants a **$10,000** loan.

1. The tool estimates a **5%** chance they'll default (PD).
2. If they do default, they'd still owe about **$8,000** at that point (EAD).
3. Of that $8,000, the lender would likely lose **60%** of it (LGD).

Expected Loss = 0.05 × $8,000 × 0.60 = **$240**.

So on average, a loan like this costs the lender about $240 in losses. The lender can now decide:
is the interest they'll earn worth that risk? That single number drives the whole tool.

---

## How does the computer make these guesses?

The estimates above come from a **model**. A model is just a formula the computer builds by
studying thousands of past loans and spotting patterns.

Think of an experienced loan officer who has reviewed 100,000 past loans. Over time, they learn
the warning signs — high debt, a shaky job history, past missed payments. A model does the same
thing, but with math instead of memory. This kind of learning-from-examples is called **machine
learning**. The computer learns the patterns on its own from real data. A person doesn't write
the rules by hand.

This project builds its models two ways and compares them:

- An **AutoML** model — short for "automated machine learning." It's a tool that builds a decent
  model on its own, with little hand-tuning. It's the quick, no-fuss benchmark to beat.
- A **fine-tuned** model — one we adjusted carefully by hand for more control and a clearer
  explanation of *why* it made each decision.

The lender can flip between the two in the app and compare.

---

## How do we know the model is any good?

We score it with a measure called **AUC**. Here's the plain version. Pick two borrowers from the
past: one who defaulted and one who didn't. AUC is how often the model gives the *defaulter* the
higher risk score. A score of 100% means it always gets that right. A score of 50% means it's no
better than a coin flip.

Our models land around **75%**. That's solidly useful, but not magic — and that's an honest
finding, not a letdown. We tested several different models and they all topped out near the same
spot. That tells us the limit comes from the **data itself**: the Prosper loan records only carry
so many clues about who will default. No clever model can squeeze out clues that aren't there.

We also check the model two more ways:

- Against the **AutoML** benchmark above.
- Against **Prosper's own risk grade** — the rating the lending company gave each loan. Beating
  their grade shows our model adds real value over what was already available.

---

## Testing it the honest way

There's an easy way to test a model and an honest way. We use the honest way.

The **easy** way mixes old and new loans together, then hides some to test on. The problem: the
model has already seen loans from the same time period, so it can "recognize the era" instead of
truly predicting.

The **honest** way is to train the model only on **older** loans, then test it on **newer** loans
it has never seen. This copies real life: you build a model today and use it on next year's
applicants. It's a tougher test, and it's the standard that real credit teams trust. We call this
an **out-of-time** test, because the test loans come from a later time than the training loans.

When we ran this honest test, some of the model's apparent skill faded — exactly the kind of
thing the easy test would have hidden. Catching that is the point.

---

## Why the economy matters

People miss payments more often when times are hard and jobs are scarce. So we added the
**unemployment rate** at the time each loan was made — the share of people looking for work but
unable to find it.

This helped, but we found a trap. The unemployment rate also acts like a hidden "date stamp,"
because it rises and falls with the calendar. On the easy test, the model partly used it to
*recognize the year* rather than to understand the economy. That kind of skill doesn't carry over
to new loans.

Our fix was to **smooth** the unemployment number — flatten out the sharp spikes and nudge it
toward its long-run normal. The smoothed version can't be used as a date stamp, so it holds up
better on the honest out-of-time test. The full story is in
[docs/macro-decision.md](docs/macro-decision.md), also written in plain terms.

---

## Turning risk into a decision (the money part)

Knowing the risk is only half the job. A lender still has to decide: lend or not, and at what
price? The tool turns the risk estimate into real money decisions:

- **Lifetime expected loss.** It spreads the risk over the whole life of the loan. It also
  treats money lost later as worth a little less than money lost today — a dollar next year is
  worth less than a dollar now. This is the cushion a lender should set aside to cover losses.
- **Expected profit and "return for the risk."** It estimates what the lender earns after losses,
  compared to how much risk they took on. Lenders call this **RAROC** — risk-adjusted return on
  capital. In plain words: *are you being paid enough for the risk you're taking?*
- **Risk-based pricing.** It suggests an interest rate that fairly covers the risk. Riskier
  borrowers need a higher rate to be worth lending to; safer ones can be offered less.

---

## The app

All of this lives behind a simple app, styled to look like an old **Windows 98** computer screen
(a fun, retro look). A lender types in a borrower and a loan, and the app shows:

- the three risk estimates (PD, LGD, EAD) and the Expected Loss,
- a switch to compare the two models,
- a plain "**Why?**" panel that lists the main reasons the borrower looks risky or safe,
- the money view: the loss cushion, the expected return, and a suggested interest rate.

---

## Where the project stands

The project is built in stages, each one a real skill that credit teams use:

- **Version 1 — done.** The core risk models (PD, LGD, EAD), the Expected Loss math, and the app.
- **Version 2 — done.** The money tools: lifetime loss cushion, expected return, and pricing.
- **Version 3 — in progress.** Making the model trustworthy and honest: testing on future loans,
  carefully checking which extra clues actually help, and adding the smoothed economy data. Still
  to come: a model that estimates *when* a loan is likely to go bad, and a written "model card"
  that documents how it works and its limits.

---

*Want the technical details — the code layout, the exact models, and how to run it? See the main
[README.md](README.md).*
