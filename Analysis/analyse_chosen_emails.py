from pathlib import Path
from itertools import combinations
import json
import re

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("./data/Emails_Cognitive.csv")
OUTPUT_PATH = Path("./data/llm_column_counts_barchart.png")
OVERCONFIDENCE_HIST_PATH = Path("./data/llm_overconfidence_histogram.png")
UPDATED_DATA_PATH = Path("./data/Emails_Cognitive_overconfidence_binary.csv")
BIAS_COMBO_VIZ_PATH = Path("./data/llm_bias_1_and_2_combinations.png")
BIAS_COMBO_HAM_VIZ_PATH = Path("./data/llm_bias_1_and_2_combinations_ham_only.png")
ONLY_BIAS_VS_PHISHING_PATH = Path("./data/llm_only_single_bias_vs_phishing.png")
CHOSEN_EMAILS_PATH = Path("./data/chosen_emails.csv")

columns_to_count = [
      "LLM Raw Response",
      "LLM Authority bias",
      "LLM Scarcity bias",
      "LLM Urgency bias",
      "LLM Anchoring effect",
      "LLM Conformity bias",
      "LLM Overconfidence",
      "LLM Familiarity bias",
]

emails_cognitive = pd.read_csv(DATA_PATH)

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


def extract_overconfidence_from_raw_response(raw_response):
      return extract_bias_scores(raw_response)["Overconfidence"]


pair_bias_labels = list(combinations(bias_names, 2))


def build_feature_matrix(binary_df):
      single_matrix = binary_df[bias_names].to_numpy(dtype=int)
      pair_columns = []
      for bias_a, bias_b in pair_bias_labels:
            pair_columns.append((binary_df[bias_a].to_numpy(dtype=int) & binary_df[bias_b].to_numpy(dtype=int)))
      pair_matrix = np.column_stack(pair_columns)
      return np.hstack([single_matrix, pair_matrix]).astype(int)


def balance_score(feature_counts):
      return feature_counts.std() + 0.25 * (feature_counts.max() - feature_counts.min())


def select_balanced_indices(binary_df, type_mask, n_select, seed):
      candidate_indices = binary_df.index[type_mask].to_numpy()
      if len(candidate_indices) < n_select:
            raise ValueError(f"Not enough candidates for selection: required {n_select}, found {len(candidate_indices)}")

      candidate_features = build_feature_matrix(binary_df.loc[candidate_indices])
      rng = np.random.default_rng(seed)
      best_selected_positions = None
      best_score = None

      # Multiple starts improve balance quality while staying fast at this dataset size.
      for _ in range(8):
            available_positions = np.arange(len(candidate_indices))
            selected_positions = []
            current_counts = np.zeros(candidate_features.shape[1], dtype=float)

            for _ in range(n_select):
                  possible_new_counts = current_counts + candidate_features[available_positions]
                  std_values = possible_new_counts.std(axis=1)
                  range_values = possible_new_counts.max(axis=1) - possible_new_counts.min(axis=1)
                  candidate_scores = std_values + 0.25 * range_values
                  candidate_scores = candidate_scores + rng.uniform(0, 1e-8, size=len(candidate_scores))
                  best_local = int(np.argmin(candidate_scores))
                  chosen_position = int(available_positions[best_local])
                  selected_positions.append(chosen_position)
                  current_counts = current_counts + candidate_features[chosen_position]
                  available_positions = np.delete(available_positions, best_local)

            final_score = balance_score(current_counts)
            if best_score is None or final_score < best_score:
                  best_score = final_score
                  best_selected_positions = selected_positions

      selected_index_values = candidate_indices[np.array(best_selected_positions, dtype=int)]
      return selected_index_values, float(best_score)


overconfidence_values = (
      emails_cognitive["LLM Raw Response"]
      .apply(extract_overconfidence_from_raw_response)
      .dropna()
)

bias_scores_df = emails_cognitive["LLM Raw Response"].apply(extract_bias_scores).apply(pd.Series)
bias_binary_df = bias_scores_df.fillna(0).gt(0).astype(int)

emails_cognitive["LLM Overconfidence"] = emails_cognitive["LLM Raw Response"].apply(
      lambda response: 1
      if (value := extract_overconfidence_from_raw_response(response)) is not None and value != 0
      else 0
)

counts = emails_cognitive[columns_to_count].notna().sum()
counts["LLM Overconfidence"] = (emails_cognitive["LLM Overconfidence"] != 0).sum()
counts = counts.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
counts.plot(kind="bar", color="#2E86AB")
plt.title("Column Counts (Overconfidence = Non-Zero Count)")
plt.xlabel("Column")
plt.ylabel("Count")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

emails_cognitive.to_csv(UPDATED_DATA_PATH, index=False)

single_bias_counts = bias_binary_df.sum().sort_values(ascending=False)
pair_bias_counts = bias_binary_df.T.dot(bias_binary_df)

plt.figure(figsize=(16, 7))

ax1 = plt.subplot(1, 2, 1)
sns.barplot(x=single_bias_counts.values, y=single_bias_counts.index, ax=ax1, color="#2E86AB")
ax1.set_title("Single Bias Presence Counts")
ax1.set_xlabel("Email count")
ax1.set_ylabel("Bias type")

ax2 = plt.subplot(1, 2, 2)
mask = np.triu(np.ones_like(pair_bias_counts, dtype=bool), k=1)
sns.heatmap(
      pair_bias_counts,
      mask=mask,
      annot=True,
      fmt="d",
      cmap="YlOrRd",
      cbar_kws={"label": "Email count"},
      ax=ax2,
)
ax2.set_title("Bias Combination Counts (1 or 2 Biases)")
ax2.set_xlabel("Bias type")
ax2.set_ylabel("Bias type")

plt.tight_layout()
plt.savefig(BIAS_COMBO_VIZ_PATH, dpi=300)
plt.close()

ham_mask = emails_cognitive["Type"].astype(str).str.lower().eq("ham")
ham_bias_binary_df = bias_binary_df.loc[ham_mask]
ham_single_bias_counts = ham_bias_binary_df.sum().sort_values(ascending=False)
ham_pair_bias_counts = ham_bias_binary_df.T.dot(ham_bias_binary_df)

plt.figure(figsize=(16, 7))

ax1 = plt.subplot(1, 2, 1)
sns.barplot(x=ham_single_bias_counts.values, y=ham_single_bias_counts.index, ax=ax1, color="#3B8B4A")
ax1.set_title("Single Bias Presence Counts (Ham Emails)")
ax1.set_xlabel("Ham email count")
ax1.set_ylabel("Bias type")

ax2 = plt.subplot(1, 2, 2)
ham_mask_upper = np.triu(np.ones_like(ham_pair_bias_counts, dtype=bool), k=1)
sns.heatmap(
      ham_pair_bias_counts,
      mask=ham_mask_upper,
      annot=True,
      fmt="d",
      cmap="Greens",
      cbar_kws={"label": "Ham email count"},
      ax=ax2,
)
ax2.set_title("Bias Combination Counts (1 or 2 Biases, Ham Emails)")
ax2.set_xlabel("Bias type")
ax2.set_ylabel("Bias type")

plt.tight_layout()
plt.savefig(BIAS_COMBO_HAM_VIZ_PATH, dpi=300)
plt.close()

phishing_mask = emails_cognitive["Type"].astype(str).str.lower().eq("phishing")
only_one_bias_mask = bias_binary_df.sum(axis=1).eq(1)

only_one_bias_counts_all = (
      bias_binary_df.loc[only_one_bias_mask]
      .sum()
      .reindex(bias_names, fill_value=0)
)
only_one_bias_counts_phishing = (
      bias_binary_df.loc[only_one_bias_mask & phishing_mask]
      .sum()
      .reindex(bias_names, fill_value=0)
)

bar_positions = np.arange(len(bias_names))
bar_width = 0.38

plt.figure(figsize=(13, 6))
plt.bar(
      bar_positions - bar_width / 2,
      only_one_bias_counts_all.values,
      width=bar_width,
      color="#1f77b4",
      label="All emails (only this bias)",
)
plt.bar(
      bar_positions + bar_width / 2,
      only_one_bias_counts_phishing.values,
      width=bar_width,
      color="#ff7f0e",
      label="Phishing emails (only this bias)",
)
plt.xticks(bar_positions, bias_names, rotation=25, ha="right")
plt.xlabel("Bias type")
plt.ylabel("Email count")
plt.title("Only-One-Bias Emails: All vs Phishing")
plt.legend()
plt.tight_layout()
plt.savefig(ONLY_BIAS_VS_PHISHING_PATH, dpi=300)
plt.close()

selected_rows = []
selection_scores = {}
for email_type, target_n, seed in [("ham", 100, 42), ("phishing", 100, 84)]:
      type_mask = emails_cognitive["Type"].astype(str).str.lower().eq(email_type)
      selected_indices, score = select_balanced_indices(
            binary_df=bias_binary_df,
            type_mask=type_mask,
            n_select=target_n,
            seed=seed,
      )
      sampled = emails_cognitive.loc[selected_indices].copy()
      sampled["SelectedEmailType"] = email_type
      sampled["SelectionMethod"] = "balanced_single_and_pair_presence"
      selected_rows.append(sampled)
      selection_scores[email_type] = score

if selected_rows:
      chosen_emails = pd.concat(selected_rows, ignore_index=True)
      chosen_emails.to_csv(CHOSEN_EMAILS_PATH, index=False)
else:
      chosen_emails = pd.DataFrame()

plt.figure(figsize=(10, 6))
sns.histplot(overconfidence_values, bins=11, color="#D1495B", edgecolor="white")
plt.title("Histogram of Overconfidence Bias Values (from LLM Raw Response)")
plt.xlabel("Overconfidence bias value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OVERCONFIDENCE_HIST_PATH, dpi=300)
plt.close()

print(f"Saved bar chart to: {OUTPUT_PATH}")
print(f"Saved overconfidence histogram to: {OVERCONFIDENCE_HIST_PATH}")
print(f"Saved updated dataset with binary overconfidence to: {UPDATED_DATA_PATH}")
print(f"Saved 1-bias and 2-bias combination visualization to: {BIAS_COMBO_VIZ_PATH}")
print(f"Saved ham-only 1-bias and 2-bias combination visualization to: {BIAS_COMBO_HAM_VIZ_PATH}")
print(f"Saved only-single-bias comparison (all vs phishing) to: {ONLY_BIAS_VS_PHISHING_PATH}")
print(f"Saved chosen email subset to: {CHOSEN_EMAILS_PATH}")
print(f"Selection balance score (ham): {selection_scores.get('ham'):.4f}")
print(f"Selection balance score (phishing): {selection_scores.get('phishing'):.4f}")

