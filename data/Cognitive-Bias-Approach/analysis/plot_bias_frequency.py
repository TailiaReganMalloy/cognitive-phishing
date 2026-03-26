import pandas as pd
import matplotlib.pyplot as plt


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


def main():
    input_path = "data/Emails_Cognitive.csv"
    output_plot_path = "data/Emails_Cognitive_bias_frequency.png"
    output_table_path = "data/Emails_Cognitive_bias_frequency.csv"
    threshold = 0.5

    df = pd.read_csv(input_path)

    available_bias_cols = [col for col in BIAS_COLUMNS if col in df.columns]
    if not available_bias_cols:
        raise ValueError("No expected LLM bias columns were found in data/Emails_Cognitive.csv")

    scores = df[available_bias_cols].apply(pd.to_numeric, errors="coerce")

    binary_df = (scores >= threshold).astype("float")
    frequency_counts = binary_df.sum(skipna=True)
    valid_counts = scores.notna().sum()
    frequency_rates = (frequency_counts / valid_counts).fillna(0.0)

    summary_df = pd.DataFrame(
        {
            "Bias": [col.replace("LLM ", "") for col in available_bias_cols],
            "Count>=0.5": frequency_counts.values.astype(int),
            "ValidN": valid_counts.values.astype(int),
            "FrequencyRate": frequency_rates.values,
        }
    ).sort_values("FrequencyRate", ascending=False)

    summary_df.to_csv(output_table_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["Bias"], summary_df["FrequencyRate"])
    plt.ylim(0, 1)
    plt.ylabel("Frequency (Proportion >= 0.5)")
    plt.xlabel("Cognitive Bias")
    plt.title("Frequency of Cognitive Biases in Emails_Cognitive.csv")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)

    print(f"Saved frequency table: {output_table_path}")
    print(f"Saved bar chart: {output_plot_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
