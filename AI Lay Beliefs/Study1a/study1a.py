import pandas as pd
from scipy import stats


DATA_PATH = "./Study1A_Data.sav"
TEST_VALUE = 5.0
ALPHA = 0.05


def load_study1a_data(path: str = DATA_PATH) -> pd.DataFrame:
    try:
        return pd.read_spss(path)
    except ImportError:
        csv_fallback = path.replace(".sav", ".csv")
        print(
            "`pyreadstat` is not installed, so pandas cannot read .sav directly. "
            f"Falling back to {csv_fallback}."
        )
        return pd.read_csv(csv_fallback)


def one_sample_summary(series: pd.Series, test_value: float = TEST_VALUE, alpha: float = ALPHA) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)

    if n < 2:
        raise ValueError("Need at least 2 non-missing observations for a one-sample t-test.")

    mean = clean.mean()
    sd = clean.std(ddof=1)
    se = sd / (n ** 0.5)

    t_stat, p_val = stats.ttest_1samp(clean, popmean=test_value, nan_policy="omit")
    df = n - 1

    # 95% CI for the sample mean (matches SPSS-style CI reporting for one-sample t test output)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se

    # One-sample Cohen's d
    cohen_d = (mean - test_value) / sd if sd > 0 else float("nan")

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "test_value": test_value,
        "mean_diff": mean - test_value,
        "t": t_stat,
        "df": df,
        "p_two_tailed": p_val,
        "ci_mean_95_low": ci_low,
        "ci_mean_95_high": ci_high,
        "cohen_d": cohen_d,
    }


def print_result(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"N: {result['n']}")
    print(f"Mean (dv): {result['mean']:.4f}")
    print(f"SD (dv): {result['sd']:.4f}")
    print(f"Test value: {result['test_value']:.2f}")
    print(f"Mean difference: {result['mean_diff']:.4f}")
    print(f"t({result['df']}): {result['t']:.4f}")
    print(f"Two-tailed p: {result['p_two_tailed']:.6f}")
    print(
        f"95% CI of mean: [{result['ci_mean_95_low']:.4f}, {result['ci_mean_95_high']:.4f}]"
    )
    print(f"Cohen's d (one-sample): {result['cohen_d']:.4f}")


def main() -> None:
    df = load_study1a_data(DATA_PATH)
    cond_raw = df["Condition"]
    cond_str = cond_raw.astype("string").str.strip().str.lower()
    cond_num = pd.to_numeric(cond_raw, errors="coerce")
    dv = pd.to_numeric(df["dv"], errors="coerce")

    # Prefer explicit labels from the dataset (Human / AI).
    # Fallback to the original SPSS numeric coding when present.
    if cond_str.isin(["human", "ai"]).any():
        mask_condition_1 = cond_str == "human"
        mask_condition_not_1 = cond_str != "human"
        label_1 = "Condition = Human (vs 5)"
        label_not_1 = "Condition != Human (AI; vs 5)"
    else:
        mask_condition_1 = cond_num == 1
        mask_condition_not_1 = cond_num != 1
        label_1 = "Condition = 1 (vs 5)"
        label_not_1 = "Condition != 1 (vs 5)"

    # SPSS equivalent:
    # 1) FILTER Condition = 1, then one-sample t-test of dv vs 5
    result_condition_1 = one_sample_summary(dv[mask_condition_1], TEST_VALUE)
    print_result(label_1, result_condition_1)

    # SPSS equivalent:
    # 2) FILTER Condition ~= 1, then one-sample t-test of dv vs 5
    result_condition_not_1 = one_sample_summary(dv[mask_condition_not_1], TEST_VALUE)
    print_result(label_not_1, result_condition_not_1)

    """
    === Condition = Human (vs 5) ===
        N: 51
        Mean (dv): 4.3333
        SD (dv): 1.4787
        Test value: 5.00
        Mean difference: -0.6667
        t(50): -3.2196
        Two-tailed p: 0.002257
        95% CI of mean: [3.9174, 4.7492]
        Cohen's d (one-sample): -0.4508

        === Condition != Human (AI; vs 5) ===
        N: 49
        Mean (dv): 8.3469
        SD (dv): 0.8552
        Test value: 5.00
        Mean difference: 3.3469
        t(48): 27.3968
        Two-tailed p: 0.000000
        95% CI of mean: [8.1013, 8.5926]
        Cohen's d (one-sample): 3.9138
        (venv) tailia.malloy@UNIJR712W3CKF AI Lay Beliefs % 

        "that participants tended to consider an AI decision-maker as relatively reliant on reason, with ratings significantly above the scale midpoint (i.e., 5; M = 8.35, SD = 0.86; t (48) = 27.397, p < .001, Cohen’s d = .855). In contrast, participants tended to view a human decision-maker as relatively reliant on feeling, with ratings significantly below the scale midpoint (M = 4.33, SD = 1.48; t (50) = -3.22, p = .001, Cohen’s d = 1.479). The difference in lay beliefs about two decision-makers was significant (F (1, 98) = 273.14, p < .001, 𝜂𝑝2 = .73)."
    """


if __name__ == "__main__":
    main()
