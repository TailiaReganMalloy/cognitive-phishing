from utils.utils import *
import pandas as pd 
import os 
import time 


def build_language_expanded_dataframe(df):
    expanded_rows = []
    language_map = {
        "text": "Chinese",
        "text_translated": "English",
    }

    for _, row in df.iterrows():
        for text_col, language in language_map.items():
            if text_col not in row.index:
                continue
            if pd.isna(row[text_col]):
                continue

            row_copy = row.copy()
            row_copy["Language"] = language
            row_copy["Body"] = row[text_col]
            expanded_rows.append(row_copy)

    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def main():
    model = get_model()

    df = pd.read_excel("Cognitive-Bias-Approach/congtive/dataset_excel/datasets_english_and_tag.xlsx")
    df.columns = df.columns.str.strip()
    df = build_language_expanded_dataframe(df)

    output_path = "./data/Email_Cognitive_Approach.csv"

    if os.path.exists(output_path):
        checkpoint_df = pd.read_csv(output_path)
        output_rows = checkpoint_df.to_dict("records")
    else:
        output_rows = []

    total_rows = len(df)
    resume_from = min(len(output_rows), total_rows)

    if resume_from > 0:
        print(f"Resuming from checkpoint: {resume_from}/{total_rows} rows already processed.")

    if len(output_rows) > total_rows:
        output_rows = output_rows[:total_rows]

    start_time = time.perf_counter()
    print_progress(resume_from, total_rows, avg_iter_seconds=0.0)

    for processed_count, (_, email) in enumerate(df.iloc[resume_from:].iterrows(), start=resume_from + 1):
        instruction = get_instruction(email, effects)
        row_data = email.to_dict()
        row_data["LLM Blocked"] = False
        row_data["LLM Block Reason"] = ""

        try:
            response = generate_with_retry(model, instruction)
            raw_response = response.text or ""
            effect_scores, parse_error = parse_effect_scores(raw_response, effects)

            row_data["LLM Raw Response"] = raw_response
            row_data["LLM Parse Error"] = parse_error

            for effect_name, score in effect_scores.items():
                row_data[f"LLM {effect_name}"] = score
        except ValueError as exc:
            error_text = str(exc)
            if "PROHIBITED_CONTENT" in error_text or "block_reason" in error_text or "Response has no candidates" in error_text:
                row_data["LLM Blocked"] = True
                row_data["LLM Block Reason"] = "PROHIBITED_CONTENT"
                row_data["LLM Raw Response"] = ""
                row_data["LLM Parse Error"] = "Skipped: blocked by safety filters (PROHIBITED_CONTENT)."
                for effect_name in effects.keys():
                    row_data[f"LLM {effect_name}"] = None
                print(f"\nSkipped blocked row at processed index {processed_count}.", flush=True)
            else:
                raise

        output_rows.append(row_data)
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_path, index=False)

        elapsed = time.perf_counter() - start_time
        processed_this_run = processed_count - resume_from
        avg_iter = elapsed / processed_this_run if processed_this_run > 0 else 0.0
        print_progress(processed_count, total_rows, avg_iter_seconds=avg_iter)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False)
    print(output_df.head())


if __name__ == "__main__":
    main()