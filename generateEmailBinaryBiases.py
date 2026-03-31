import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

Emails_Formatted = pd.read_csv("Emails_Formatted.csv")

print(Emails_Formatted.columns)
"""
Index(['Unnamed: 0', 'EmailId', 'BaseEmailID', 'Author', 'Style', 'Type',
       'Sender Style', 'Sender', 'Subject', 'Sender Mismatch',
       'Request Credentials', 'Subject Suspicious', 'Urgent', 'Offer',
       'Link Mismatch', 'Prompt', 'Body', 'LLM Raw Response',
       'LLM Parse Error', 'LLM Authority bias', 'LLM Scarcity bias',
       'LLM Urgency bias', 'LLM Anchoring effect', 'LLM Conformity bias',
       'LLM Overconfidence', 'LLM Familiarity bias', 'SelectedEmailType',
       'SelectionMethod', 'New Email Body', 'New Email Response',
       'New Email Prompt', 'New Email Config', 'New Email Model Name',
       'Target Biases Used', 'Generation Status', 'Generation Error',
       'Generated At UTC', 'combined_target_score'],
      dtype='str')
"""

new_cols = [
    'Scarcity',
    'Urgency', 
    'Anchoring', 
    'Conformity',
    'Overconfidence', 
    'Familiarity',
]

llm_bias_cols = [
    'LLM Authority bias',
    'LLM Scarcity bias',
    'LLM Urgency bias',
    'LLM Anchoring effect',
    'LLM Conformity bias',
    'LLM Overconfidence',
    'LLM Familiarity bias',
]

new_to_llm_map = {
    'Scarcity': 'LLM Scarcity bias',
    'Urgency': 'LLM Urgency bias',
    'Anchoring': 'LLM Anchoring effect',
    'Conformity': 'LLM Conformity bias',
    'Overconfidence': 'LLM Overconfidence',
    'Familiarity': 'LLM Familiarity bias',
}

for col in llm_bias_cols:
    if col not in Emails_Formatted.columns:
        raise KeyError(f"Missing required LLM bias column: {col}")

for col in new_cols:
    Emails_Formatted[col] = 0


def assign_balanced_top_two(df_group):
    if len(df_group) == 0:
        return pd.DataFrame(columns=new_cols, index=df_group.index)

    llm_for_new = [new_to_llm_map[col] for col in new_cols]
    raw_scores = df_group[llm_for_new].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    raw_scores.columns = new_cols

    # Normalize each bias in-group so scales are comparable.
    score_min = raw_scores.min(axis=0)
    score_max = raw_scores.max(axis=0)
    denom = (score_max - score_min).replace(0, 1.0)
    norm_scores = (raw_scores - score_min) / denom

    # Group-specific thresholds: keep each bias eligible for roughly top third of rows.
    # This dampens over-selection of one dominant bias (e.g., Familiarity).
    thresholds = raw_scores.quantile(2.0 / 3.0, axis=0)

    selected = pd.DataFrame(0, index=df_group.index, columns=new_cols, dtype=int)
    bias_counts = {col: 0 for col in new_cols}
    pair_counts = {tuple(sorted(pair)): 0 for pair in combinations(new_cols, 2)}

    # Process high-confidence rows first to preserve strong signal.
    row_strength = norm_scores.apply(lambda r: r.sort_values(ascending=False).head(2).sum(), axis=1)
    row_order = row_strength.sort_values(ascending=False).index.tolist()

    lambda_bias = 0.20
    lambda_pair = 0.55

    for row_idx in row_order:
        row_raw = raw_scores.loc[row_idx]
        row_norm = norm_scores.loc[row_idx]

        eligible = [col for col in new_cols if row_raw[col] >= thresholds[col]]
        candidate_biases = eligible if len(eligible) >= 2 else list(new_cols)
        candidate_pairs = list(combinations(candidate_biases, 2))

        best_pair = None
        best_metric = -1e18
        for b1, b2 in candidate_pairs:
            pair_key = tuple(sorted((b1, b2)))
            score_component = float(row_norm[b1] + row_norm[b2])
            balance_penalty = (lambda_bias * (bias_counts[b1] + bias_counts[b2])) + (lambda_pair * pair_counts[pair_key])
            metric = score_component - balance_penalty
            if metric > best_metric:
                best_metric = metric
                best_pair = (b1, b2)

        b1, b2 = best_pair
        selected.at[row_idx, b1] = 1
        selected.at[row_idx, b2] = 1
        bias_counts[b1] += 1
        bias_counts[b2] += 1
        pair_counts[tuple(sorted((b1, b2)))] += 1

    return selected


for col in new_cols:
    Emails_Formatted[col] = 0

for author in ["Human", "GPT"]:
    for email_type in ["ham", "phishing"]:
        group_mask = (
            Emails_Formatted["Author"].astype(str).str.lower().eq(author.lower())
            & Emails_Formatted["Type"].astype(str).str.lower().eq(email_type.lower())
        )
        group_df = Emails_Formatted.loc[group_mask]
        group_selected = assign_balanced_top_two(group_df)
        Emails_Formatted.loc[group_mask, new_cols] = group_selected[new_cols]

# Fallback for any rows outside the four target groups.
remaining_mask = Emails_Formatted[new_cols].sum(axis=1) == 0
if remaining_mask.any():
    fallback_scores = Emails_Formatted.loc[remaining_mask, [new_to_llm_map[c] for c in new_cols]].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    fallback_scores.columns = new_cols
    top_two = fallback_scores.apply(lambda r: r.sort_values(ascending=False).index[:2].tolist(), axis=1)
    for row_idx, pair in top_two.items():
        Emails_Formatted.at[row_idx, pair[0]] = 1
        Emails_Formatted.at[row_idx, pair[1]] = 1

Emails_Formatted.to_csv("Emails_Formatted.csv", index=False)

print("Added balanced top-2 indicator columns and saved Emails_Formatted.csv")


def compute_pair_matrix(df, bias_columns):
    matrix = pd.DataFrame(np.nan, index=bias_columns, columns=bias_columns, dtype=float)

    for i, col_i in enumerate(bias_columns):
        col_i_vals = pd.to_numeric(df[col_i], errors="coerce").fillna(0).astype(int)
        for j, col_j in enumerate(bias_columns):
            col_j_vals = pd.to_numeric(df[col_j], errors="coerce").fillna(0).astype(int)
            if i == j:
                # Diagonal represents single-bias frequency; hide it for pair-only visualization.
                continue
            matrix.iloc[i, j] = int(((col_i_vals == 1) & (col_j_vals == 1)).sum())

    return matrix


def plot_pair_heatmap(ax, matrix, title):
    values = matrix.values.astype(float)
    masked_values = np.ma.masked_invalid(values)
    im = ax.imshow(values, cmap="YlOrRd")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            ax.text(j, i, int(values[i, j]), ha="center", va="center", color="black", fontsize=8)

    # Render NaN (diagonal) as empty/white cells.
    im.set_array(masked_values)

    return im


authors = ["Human", "GPT"]
types = ["ham", "phishing"]

fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
last_im = None

for r, author in enumerate(authors):
    for c, email_type in enumerate(types):
        ax = axes[r, c]
        subset = Emails_Formatted[
            Emails_Formatted["Author"].astype(str).str.lower().eq(author.lower())
            & Emails_Formatted["Type"].astype(str).str.lower().eq(email_type.lower())
        ]

        matrix = compute_pair_matrix(subset, new_cols)
        label_type = "spam" if email_type == "phishing" else email_type
        title = f"{author} | {label_type} | n={len(subset)}"
        last_im = plot_pair_heatmap(ax, matrix, title)

if last_im is not None:
    fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02, label="Pair Occurrence Count")

output_path = "bias_pair_occurrence_2x2.png"
plt.savefig(output_path, dpi=300)
print(f"Saved 2x2 bias-pair occurrence plot to {output_path}")