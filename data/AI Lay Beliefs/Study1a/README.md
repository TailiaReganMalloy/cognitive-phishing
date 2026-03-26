# Study 1A Reproducibility Check (from `study1a.py`)

This README evaluates whether the quoted results are reproducible from the current Python analysis in `study1a.py` using `Study1A_Data.sav` (or `Study1A_Data.csv` fallback).

## Data and tests reproduced

From `study1a.py` output:

- **Human condition (one-sample t-test vs midpoint 5):**
  - `N = 51`
  - `M = 4.3333`
  - `SD = 1.4787`
  - `t(50) = -3.2196`
  - `p = 0.002257`
  - `Cohen's d (one-sample) = -0.4508`

- **AI condition (one-sample t-test vs midpoint 5):**
  - `N = 49`
  - `M = 8.3469`
  - `SD = 0.8552`
  - `t(48) = 27.3968`
  - `p < .001` (printed as `0.000000` due to formatting)
  - `Cohen's d (one-sample) = 3.9138`

Additional between-group test on `dv` by `Condition` (one-way ANOVA, computed from same dataset):

- `F(1, 98) = 273.1418`
- `p = 4.29e-30` (i.e., `< .001`)
- `eta_p^2 = 0.73595` (≈ `.736`, rounds to `.73`)

## Reproducibility verdict for the quoted text

Quoted claim:

> "...AI... above midpoint (M = 8.35, SD = 0.86; t(48) = 27.397, p < .001, Cohen's d = .855). In contrast, Human... below midpoint (M = 4.33, SD = 1.48; t(50) = -3.22, p = .001, Cohen's d = 1.479). ... difference ... significant (F(1, 98) = 273.14, p < .001, ηp2 = .73)."

### What is reproducible

- ✅ **Direction and significance** of both one-sample tests (AI above 5; Human below 5).
- ✅ Means/SDs/t-statistics match after rounding:
  - AI: `M = 8.35`, `SD = 0.86`, `t(48) = 27.397`
  - Human: `M = 4.33`, `SD = 1.48`, `t(50) = -3.22`
- ✅ Between-group ANOVA matches after rounding:
  - `F(1,98) = 273.14`, `p < .001`, `ηp² ≈ .74` (reported `.73` is plausible via rounding conventions).

### What is **not** reproducible exactly

- ❌ **Cohen's d values in the quote are not reproducible** from the one-sample t-tests:
  - Quoted AI `d = .855`; reproducible one-sample `d = 3.9138`
  - Quoted Human `d = 1.479`; reproducible one-sample `d = -0.4508` (absolute value `0.4508`)

- ⚠️ Human p-value is quoted as `.001`, while reproducible value is `p = 0.002257` (rounds to `.002`, still significant).

## Conclusion

The core inferential conclusions are reproducible (AI above midpoint, Human below midpoint, and large between-group difference). However, the quoted Cohen's d values are not consistent with the one-sample effect-size calculation used in `study1a.py`.
