import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BIAS_COLUMNS = [
    "LLM Authority Bias",
    "LLM Survivorship Bias",
    "LLM Pessimism Bias",
    "LLM Zero-Risk Bias",
    "LLM Hyperbolic Discounting",
    "LLM Identifiable Victim Effect",
    "LLM Appeal to Novelty",
    "LLM Urgency Effect",
    "LLM Curiosity",
    "LLM Conformity",
]


def compute_frequency(scores_df, threshold=0.5):
    numeric_scores = scores_df.apply(pd.to_numeric, errors="coerce")
    binary_scores = (numeric_scores >= threshold).astype("float")
    positive_count = binary_scores.sum(skipna=True)
    valid_n = numeric_scores.notna().sum()
    rate = (positive_count / valid_n).fillna(0.0)
    return positive_count, valid_n, rate


def main():
    input_path = "data/Emails_Cognitive.csv"
    output_plot_path = "data/Emails_Cognitive_bias_frequency_human_vs_llm.png"
    output_table_path = "data/Emails_Cognitive_bias_frequency_human_vs_llm.csv"
    threshold = 0.5

    df = pd.read_csv(input_path)

    if "Author" not in df.columns:
        raise ValueError("Expected 'Author' column in data/Emails_Cognitive.csv")

    available_bias_cols = [col for col in BIAS_COLUMNS if col in df.columns]
    if not available_bias_cols:
        raise ValueError("No expected LLM bias columns found in data/Emails_Cognitive.csv")

    human_df = df[df["Author"] == "Human"].copy()
    llm_df = df[df["Author"] != "Human"].copy()

    if human_df.empty or llm_df.empty:
        raise ValueError("Need both Human and non-Human (LLM) rows to build comparison plot")

    human_count, human_valid_n, human_rate = compute_frequency(human_df[available_bias_cols], threshold=threshold)
    llm_count, llm_valid_n, llm_rate = compute_frequency(llm_df[available_bias_cols], threshold=threshold)

    summary_rows = []
    for col in available_bias_cols:
        bias_name = col.replace("LLM ", "")
        summary_rows.append(
            {
                "Bias": bias_name,
                "Human Count>=0.5": int(human_count[col]),
                "Human ValidN": int(human_valid_n[col]),
                "Human FrequencyRate": float(human_rate[col]),
                "LLM Count>=0.5": int(llm_count[col]),
                "LLM ValidN": int(llm_valid_n[col]),
                "LLM FrequencyRate": float(llm_rate[col]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Human FrequencyRate", ascending=False)
    summary_df.to_csv(output_table_path, index=False)

    x = np.arange(len(summary_df))
    width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, summary_df["Human FrequencyRate"], width=width, label="Human")
    plt.bar(x + width / 2, summary_df["LLM FrequencyRate"], width=width, label="LLM")
    plt.ylim(0, 1)
    plt.ylabel("Frequency (Proportion >= 0.5)")
    plt.xlabel("Cognitive Bias")
    plt.title("Cognitive Bias Frequencies: Human vs LLM-Written Emails")
    plt.xticks(x, summary_df["Bias"], rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)

    print(f"Saved frequency comparison table: {output_table_path}")
    print(f"Saved frequency comparison plot: {output_plot_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
