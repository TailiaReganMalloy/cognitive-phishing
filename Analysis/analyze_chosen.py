from pathlib import Path
import json
import re

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("./data/chosen_emails.csv")
HAM_OUTPUT_PATH = Path("./data/chosen_bias_1_and_2_combinations_ham.png")
PHISHING_OUTPUT_PATH = Path("./data/chosen_bias_1_and_2_combinations_phishing.png")

bias_names = [
	"Authority",
	"Scarcity",
	"Urgency",
	"Anchoring effect",
	"Conformity",
	"Overconfidence",
	"Familiarity",
]


def clean_raw_response(raw_response):
	if not isinstance(raw_response, str) or not raw_response.strip():
		return ""
	cleaned = raw_response.replace('""', '"')
	cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
	cleaned = cleaned.replace("```", "")
	cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
	return cleaned


def extract_bias_scores(raw_response):
	cleaned = clean_raw_response(raw_response)
	scores = {bias: None for bias in bias_names}
	if not cleaned:
		return scores

	start = cleaned.find("{")
	end = cleaned.rfind("}")
	if start != -1 and end != -1 and end > start:
		json_candidate = cleaned[start : end + 1]
		try:
			parsed = json.loads(json_candidate)
			for key, value in parsed.items():
				key_lower = str(key).lower()
				for bias in bias_names:
					if bias.lower() in key_lower:
						try:
							scores[bias] = float(value)
						except (TypeError, ValueError):
							pass
		except json.JSONDecodeError:
			pass

	for bias in bias_names:
		if scores[bias] is None:
			pattern = rf'"?{re.escape(bias)}(?:\s+bias){{0,2}}"?\s*:\s*(-?\d*\.?\d+)'
			regex_match = re.search(pattern, cleaned, flags=re.IGNORECASE)
			if regex_match:
				try:
					scores[bias] = float(regex_match.group(1))
				except ValueError:
					pass

	return scores


def plot_bias_summary_for_type(df, email_type, output_path, color, cmap):
	type_mask = df["Type"].astype(str).str.lower().eq(email_type)
	type_df = df.loc[type_mask].copy()

	bias_scores_df = type_df["LLM Raw Response"].apply(extract_bias_scores).apply(pd.Series)
	bias_binary_df = bias_scores_df.fillna(0).gt(0).astype(int)

	single_bias_counts = bias_binary_df.sum().sort_values(ascending=False)
	pair_bias_counts = bias_binary_df.T.dot(bias_binary_df)

	plt.figure(figsize=(16, 7))

	ax1 = plt.subplot(1, 2, 1)
	sns.barplot(
		x=single_bias_counts.values,
		y=single_bias_counts.index,
		ax=ax1,
		color=color,
	)
	ax1.set_title(f"Single Bias Presence Counts ({email_type.title()} Emails)")
	ax1.set_xlabel(f"{email_type.title()} email count")
	ax1.set_ylabel("Bias type")

	ax2 = plt.subplot(1, 2, 2)
	upper_mask = np.triu(np.ones_like(pair_bias_counts, dtype=bool), k=1)
	sns.heatmap(
		pair_bias_counts,
		mask=upper_mask,
		annot=True,
		fmt="d",
		cmap=cmap,
		cbar_kws={"label": f"{email_type.title()} email count"},
		ax=ax2,
	)
	ax2.set_title(f"Bias Combination Counts (1 or 2 Biases, {email_type.title()} Emails)")
	ax2.set_xlabel("Bias type")
	ax2.set_ylabel("Bias type")

	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()


def main():
	chosen_df = pd.read_csv(DATA_PATH)
	plot_bias_summary_for_type(
		chosen_df,
		email_type="ham",
		output_path=HAM_OUTPUT_PATH,
		color="#3B8B4A",
		cmap="Greens",
	)
	plot_bias_summary_for_type(
		chosen_df,
		email_type="phishing",
		output_path=PHISHING_OUTPUT_PATH,
		color="#D1495B",
		cmap="YlOrRd",
	)

	print(f"Saved ham bias summary to: {HAM_OUTPUT_PATH}")
	print(f"Saved phishing bias summary to: {PHISHING_OUTPUT_PATH}")


if __name__ == "__main__":
	main()