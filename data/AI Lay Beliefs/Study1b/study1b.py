import pandas as pd
from scipy import stats


DATA_PATH = "./Study1B_Data.sav"
ALPHA = 0.05

CLAIM = (
	"Results from one-sample t-tests showed that participants tended to consider an "
	"AI decision-maker as more reliant on reason than feeling, as reflected in an "
	"index mean being significantly below zero (M = -5.78, SD = 2.16; t (100) = "
	"-26.884, p < .001, Cohen’s d = 2.161). On the other hand, participants tended "
	"to view a human decision-maker as more reliant on feeling than reason, for the "
	"index mean being significantly above zero (M = 1.68, SD = 2.42; t (100) = "
	"6.989, p < .001, Cohen’s d = 2.42). This difference in the lay beliefs about "
	"two decision-makers was significant (paired sample t-test: t (100) = 22.427, "
	"p < 0.001; Cohen’s d = 3.345)."
)


def load_study1b_data(path: str = DATA_PATH) -> pd.DataFrame:
	try:
		return pd.read_spss(path)
	except ImportError:
		csv_fallback = path.replace(".sav", ".csv")
		print(
			"`pyreadstat` is not installed, so pandas cannot read .sav directly. "
			f"Falling back to {csv_fallback}."
		)
		return pd.read_csv(csv_fallback)


def one_sample_summary(series: pd.Series, test_value: float = 0.0, alpha: float = ALPHA) -> dict:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	n = len(clean)
	if n < 2:
		raise ValueError("Need at least 2 non-missing observations for a one-sample t-test.")

	mean = clean.mean()
	sd = clean.std(ddof=1)
	se = sd / (n ** 0.5)
	t_stat, p_val = stats.ttest_1samp(clean, popmean=test_value, nan_policy="omit")
	df = n - 1

	t_crit = stats.t.ppf(1 - alpha / 2, df)
	ci_low = mean - t_crit * se
	ci_high = mean + t_crit * se

	d = (mean - test_value) / sd if sd > 0 else float("nan")

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
		"cohen_d": d,
	}


def paired_summary(x: pd.Series, y: pd.Series, alpha: float = ALPHA) -> dict:
	x_num = pd.to_numeric(x, errors="coerce")
	y_num = pd.to_numeric(y, errors="coerce")
	paired = pd.DataFrame({"x": x_num, "y": y_num}).dropna()

	if len(paired) < 2:
		raise ValueError("Need at least 2 complete paired observations for a paired t-test.")

	diff = paired["x"] - paired["y"]
	n = len(diff)
	df = n - 1

	mean_x = paired["x"].mean()
	sd_x = paired["x"].std(ddof=1)
	mean_y = paired["y"].mean()
	sd_y = paired["y"].std(ddof=1)

	mean_diff = diff.mean()
	sd_diff = diff.std(ddof=1)
	se_diff = sd_diff / (n ** 0.5)

	t_stat, p_val = stats.ttest_rel(paired["x"], paired["y"], nan_policy="omit")

	t_crit = stats.t.ppf(1 - alpha / 2, df)
	ci_low = mean_diff - t_crit * se_diff
	ci_high = mean_diff + t_crit * se_diff

	# SPSS paired t-test effect size with STANDARDIZER(SD):
	# d = mean difference / SD of differences
	d = mean_diff / sd_diff if sd_diff > 0 else float("nan")

	return {
		"n": n,
		"mean_x": mean_x,
		"sd_x": sd_x,
		"mean_y": mean_y,
		"sd_y": sd_y,
		"mean_diff": mean_diff,
		"sd_diff": sd_diff,
		"t": t_stat,
		"df": df,
		"p_two_tailed": p_val,
		"ci_diff_95_low": ci_low,
		"ci_diff_95_high": ci_high,
		"cohen_d": d,
	}


def fmt_p(p: float) -> str:
	return "< .001" if p < 0.001 else f"= {p:.3f}"


def print_one_sample(label: str, result: dict) -> None:
	print(f"\n=== One-sample t-test: {label} vs {result['test_value']:.0f} ===")
	print(f"N: {result['n']}")
	print(f"M: {result['mean']:.4f}")
	print(f"SD: {result['sd']:.4f}")
	print(f"t({result['df']}): {result['t']:.4f}")
	print(f"Two-tailed p: {result['p_two_tailed']:.6f} ({fmt_p(result['p_two_tailed'])})")
	print(f"95% CI of mean: [{result['ci_mean_95_low']:.4f}, {result['ci_mean_95_high']:.4f}]")
	print(f"Cohen's d: {result['cohen_d']:.4f}")


def print_paired(label_x: str, label_y: str, result: dict) -> None:
	print(f"\n=== Paired t-test: {label_x} - {label_y} ===")
	print(f"N (pairs): {result['n']}")
	print(f"{label_x}: M = {result['mean_x']:.4f}, SD = {result['sd_x']:.4f}")
	print(f"{label_y}: M = {result['mean_y']:.4f}, SD = {result['sd_y']:.4f}")
	print(f"Mean difference ({label_x}-{label_y}): {result['mean_diff']:.4f}")
	print(f"SD of differences: {result['sd_diff']:.4f}")
	print(f"t({result['df']}): {result['t']:.4f}")
	print(f"Two-tailed p: {result['p_two_tailed']:.6f} ({fmt_p(result['p_two_tailed'])})")
	print(f"95% CI of mean difference: [{result['ci_diff_95_low']:.4f}, {result['ci_diff_95_high']:.4f}]")
	print(f"Cohen's d (SD of differences): {result['cohen_d']:.4f}")


def print_claim_check(human: dict, ai: dict, paired: dict) -> None:
	print("\n=== Claim reproducibility check ===")
	print(CLAIM)
	print("\nComputed values from this script:")
	print(
		"AI index: "
		f"M={ai['mean']:.2f}, SD={ai['sd']:.2f}, t({ai['df']})={ai['t']:.3f}, "
		f"p {fmt_p(ai['p_two_tailed'])}, d={ai['cohen_d']:.3f}"
	)
	print(
		"Human index: "
		f"M={human['mean']:.2f}, SD={human['sd']:.2f}, t({human['df']})={human['t']:.3f}, "
		f"p {fmt_p(human['p_two_tailed'])}, d={human['cohen_d']:.3f}"
	)
	print(
		"Paired difference (human-ai): "
		f"t({paired['df']})={paired['t']:.3f}, p {fmt_p(paired['p_two_tailed'])}, "
		f"d={paired['cohen_d']:.3f}"
	)


def main() -> None:
	df = load_study1b_data(DATA_PATH)

	humindex = pd.to_numeric(df["humindex"], errors="coerce")
	aiindex = pd.to_numeric(df["aiindex"], errors="coerce")

	# SPSS syntax equivalent:
	# T-TEST PAIRS=humindex WITH aiindex (PAIRED)
	paired = paired_summary(humindex, aiindex)
	print_paired("humindex", "aiindex", paired)

	# SPSS syntax equivalent:
	# T-TEST /TESTVAL=0 /VARIABLES=humindex aiindex
	human_one_sample = one_sample_summary(humindex, test_value=0.0)
	ai_one_sample = one_sample_summary(aiindex, test_value=0.0)

	print_one_sample("humindex", human_one_sample)
	print_one_sample("aiindex", ai_one_sample)

	print_claim_check(human_one_sample, ai_one_sample, paired)


if __name__ == "__main__":
	main()