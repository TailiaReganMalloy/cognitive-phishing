# Study 1B Reproduction and Claim Check

This folder reproduces the SPSS analyses in `Study1B_Syntax.sps` using Python in `study1b.py`, with `Study1B_Data.sav` as input.

## SPSS analyses reproduced

From SPSS syntax:

1. Paired t-test: `humindex` with `aiindex`
2. One-sample t-tests vs `0` for `humindex` and `aiindex`

`study1b.py` reproduces all three tests and reports descriptive statistics, t values, p values, confidence intervals, and Cohen's d.

## Python output (from `study1b.py`)

- **Paired t-test (`humindex - aiindex`)**
  - `N = 101`
  - `t(100) = 22.4270`, `p < .001`
  - Mean difference = `7.4653`
  - `Cohen's d = 2.2316`

- **One-sample t-test (`humindex` vs `0`)**
  - `N = 101`
  - `M = 1.6832`, `SD = 2.4205`
  - `t(100) = 6.9886`, `p < .001`
  - `Cohen's d = 0.6954`

- **One-sample t-test (`aiindex` vs `0`)**
  - `N = 101`
  - `M = -5.7822`, `SD = 2.1615`
  - `t(100) = -26.8842`, `p < .001`
  - `Cohen's d = -2.6751`

## Claim reproducibility verdict

Claim checked:

> "Results from one-sample t-tests showed that participants tended to consider an AI decision-maker as more reliant on reason than feeling, as reflected in an index mean being significantly below zero (M = -5.78, SD = 2.16; t (100) = -26.884, p < .001, Cohen’s d = 2.161). On the other hand, participants tended to view a human decision-maker as more reliant on feeling than reason, for the index mean being significantly above zero (M = 1.68, SD = 2.42; t (100) = 6.989, p < .001, Cohen’s d = 2.42). This difference in the lay beliefs about two decision-makers was significant (paired sample t-test: t (100) = 22.427, p < 0.001; Cohen’s d = 3.345)."

### Reproducible components

- ✅ Means, SDs, t values, and p-value significance statements are reproducible (match to rounding):
  - AI one-sample: `M = -5.78`, `SD = 2.16`, `t(100) = -26.884`, `p < .001`
  - Human one-sample: `M = 1.68`, `SD = 2.42`, `t(100) = 6.989`, `p < .001`
  - Paired: `t(100) = 22.427`, `p < .001`

### Not reproducible exactly

- ❌ The quoted Cohen's d values do not match standard t-test effect-size calculations:
  - AI one-sample quoted `d = 2.161`; reproduced `d = -2.675` (absolute value `2.675`)
  - Human one-sample quoted `d = 2.42`; reproduced `d = 0.695`
  - Paired quoted `d = 3.345`; reproduced `d = 2.232`

## Bottom line

The inferential conclusions in the claim are reproducible, but the reported Cohen's d values are not reproducible from the standard one-sample and paired t-test formulas used in `study1b.py`.
