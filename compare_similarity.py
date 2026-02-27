import pandas as pd


BIAS_NAMES = [
    "Authority Bias",
    "Survivorship Bias",
    "Pessimism Bias",
    "Zero-Risk Bias",
    "Hyperbolic Discounting",
    "Identifiable Victim Effect",
    "Appeal to Novelty",
    "Urgency Effect",
    "Curiosity",
    "Conformity",
]


def build_human_long_dataframe(source_df):
    rows = []
    bias_names = list(BIAS_NAMES)

    for row_idx, row in source_df.iterrows():
        source_id = row["Unnamed: 0"] if "Unnamed: 0" in source_df.columns else row_idx

        for text_col, language in (("text", "Chinese"), ("text_translated", "English")):
            if text_col not in source_df.columns:
                continue
            text_value = row[text_col]
            if pd.isna(text_value):
                continue

            item = {
                "source_id": source_id,
                "Language": language,
                "Body": text_value,
            }

            for bias in bias_names:
                human_value = row[bias] if bias in source_df.columns else None
                item[f"Human {bias}"] = pd.to_numeric(human_value, errors="coerce")

            rows.append(item)

    return pd.DataFrame(rows)


def standardize_predictions_df(pred_df):
    pred_df = pred_df.copy()
    pred_df.columns = pred_df.columns.str.strip()

    if "source_id" not in pred_df.columns:
        if "Unnamed: 0" in pred_df.columns:
            pred_df["source_id"] = pred_df["Unnamed: 0"]

    if "Body" in pred_df.columns:
        pred_df["Body"] = pred_df["Body"].astype(str)

    return pred_df


def compute_bias_metrics(merged_df):
    rows = []

    for bias in BIAS_NAMES:
        human_col = f"Human {bias}"
        llm_col = f"LLM {bias}"

        if human_col not in merged_df.columns or llm_col not in merged_df.columns:
            continue

        data = merged_df[[human_col, llm_col]].copy()
        data[human_col] = pd.to_numeric(data[human_col], errors="coerce")
        data[llm_col] = pd.to_numeric(data[llm_col], errors="coerce")
        data = data.dropna()

        n = len(data)
        if n == 0:
            rows.append(
                {
                    "Bias": bias,
                    "N": 0,
                    "Pearson": None,
                    "MAE": None,
                    "BinaryAgreement@0.5": None,
                    "BestCutoff": None,
                    "BinaryAgreement@BestCutoff": None,
                }
            )
            continue

        pearson = data[human_col].corr(data[llm_col])
        mae = (data[human_col] - data[llm_col]).abs().mean()
        human_binary = data[human_col] >= 0.5
        agreement = (human_binary == (data[llm_col] >= 0.5)).mean()

        candidate_cutoffs = sorted(set(data[llm_col].tolist() + [0.0, 0.5, 1.0]))
        best_cutoff = 0.5
        best_agreement = -1.0
        best_distance_from_default = float("inf")

        for cutoff in candidate_cutoffs:
            pred_binary = data[llm_col] >= cutoff
            cutoff_agreement = (human_binary == pred_binary).mean()
            distance_from_default = abs(cutoff - 0.5)

            if (
                cutoff_agreement > best_agreement
                or (
                    cutoff_agreement == best_agreement
                    and distance_from_default < best_distance_from_default
                )
                or (
                    cutoff_agreement == best_agreement
                    and distance_from_default == best_distance_from_default
                    and cutoff < best_cutoff
                )
            ):
                best_cutoff = float(cutoff)
                best_agreement = float(cutoff_agreement)
                best_distance_from_default = distance_from_default

        rows.append(
            {
                "Bias": bias,
                "N": int(n),
                "Pearson": float(pearson) if pd.notna(pearson) else None,
                "MAE": float(mae) if pd.notna(mae) else None,
                "BinaryAgreement@0.5": float(agreement) if pd.notna(agreement) else None,
                "BestCutoff": best_cutoff,
                "BinaryAgreement@BestCutoff": best_agreement,
            }
        )

    return pd.DataFrame(rows)


def main():
    source_path = "Cognitive-Bias-Approach/congtive/dataset_excel/datasets_english_and_tag.xlsx"
    pred_path = "data/Email_Cognitive_Approach.csv"

    pred_df = pd.read_csv(pred_path)
    pred_df = standardize_predictions_df(pred_df)

    human_cols_already_present = all(bias in pred_df.columns for bias in BIAS_NAMES)

    if human_cols_already_present:
        merged = pred_df.copy()
        for bias in BIAS_NAMES:
            merged[f"Human {bias}"] = pd.to_numeric(merged[bias], errors="coerce")
    else:
        source_df = pd.read_excel(source_path)
        source_df.columns = source_df.columns.str.strip()
        human_long_df = build_human_long_dataframe(source_df)

        if "source_id" in pred_df.columns and "Language" in pred_df.columns:
            merged = pred_df.merge(
                human_long_df,
                on=["source_id", "Language"],
                how="inner",
                suffixes=("", "_human"),
            )
        else:
            merged = pred_df.merge(
                human_long_df,
                on=["Body", "Language"],
                how="inner",
                suffixes=("", "_human"),
            )

    metrics_df = compute_bias_metrics(merged)

    metrics_output_path = "data/Email_Cognitive_Approach_similarity.csv"
    merged_output_path = "data/Email_Cognitive_Approach_joined_for_similarity.csv"

    metrics_df.to_csv(metrics_output_path, index=False)
    merged.to_csv(merged_output_path, index=False)

    print("Saved metrics:", metrics_output_path)
    print("Saved joined rows:", merged_output_path)
    print(metrics_df)


if __name__ == "__main__":
    main()
