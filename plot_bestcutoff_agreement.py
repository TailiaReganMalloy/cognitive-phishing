import pandas as pd
import matplotlib.pyplot as plt


def main():
    input_path = "data/Email_Cognitive_Approach_similarity.csv"
    output_plot_path = "data/Email_Cognitive_Approach_binary_agreement_bestcutoff.png"

    df = pd.read_csv(input_path)

    required_cols = ["Bias", "BinaryAgreement@BestCutoff"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {input_path}: {missing_cols}")

    plot_df = df[["Bias", "BinaryAgreement@BestCutoff"]].copy()
    plot_df["BinaryAgreement@BestCutoff"] = pd.to_numeric(
        plot_df["BinaryAgreement@BestCutoff"], errors="coerce"
    )
    plot_df = plot_df.dropna().sort_values("BinaryAgreement@BestCutoff", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["Bias"], plot_df["BinaryAgreement@BestCutoff"])
    plt.ylim(0, 1)
    plt.ylabel("Binary Agreement @ Best Cutoff")
    plt.xlabel("Cognitive Bias")
    plt.title("Binary Agreement @ Best Cutoff by Cognitive Bias")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)

    print(f"Saved plot: {output_plot_path}")
    print(plot_df)


if __name__ == "__main__":
    main()
