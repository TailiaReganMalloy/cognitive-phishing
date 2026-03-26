from Utilities.utils import *
import pandas as pd 
import os 
import time 

def main():
    model = get_model()

    emails = pd.read_csv("./Data/Emails.csv")

    emails = emails[emails['Style'] == "Plaintext"]
    emails = emails.reset_index(drop=True)

    wide_output_path = "./data/Emails_Cognitive.csv"
    long_output_path = "./data/raw/Emails_with_llm_effect_predictions_long.csv"

    if os.path.exists(wide_output_path):
        checkpoint_wide_df = pd.read_csv(wide_output_path)
        wide_rows = checkpoint_wide_df.to_dict("records")
    else:
        wide_rows = []

    if os.path.exists(long_output_path):
        checkpoint_long_df = pd.read_csv(long_output_path)
        long_rows = checkpoint_long_df.to_dict("records")
    else:
        long_rows = []

    total_emails = len(emails)
    resume_from = min(len(wide_rows), total_emails)

    if resume_from > 0:
        print(f"Resuming from checkpoint: {resume_from}/{total_emails} emails already processed.")

    if len(wide_rows) > total_emails:
        wide_rows = wide_rows[:total_emails]

    start_time = time.perf_counter()
    print_progress(resume_from, total_emails, avg_iter_seconds=0.0)

    for processed_count, (idx, email) in enumerate(emails.iloc[resume_from:].iterrows(), start=resume_from + 1):
        instruction = get_instruction(email, effects)
        response = generate_with_retry(model, instruction)
        raw_response = response.text or ""
        effect_scores, parse_error = parse_effect_scores(raw_response, effects)

        row_data = email.to_dict()
        row_data["LLM Raw Response"] = raw_response
        row_data["LLM Parse Error"] = parse_error

        for effect_name, score in effect_scores.items():
            row_data[f"LLM {effect_name}"] = score
            long_rows.append({
                **email.to_dict(),
                "Effect": effect_name,
                "Effect Score": score,
                "LLM Parse Error": parse_error,
                "LLM Raw Response": raw_response,
            })

        wide_rows.append(row_data)
        emails_with_predictions = pd.DataFrame(wide_rows)
        emails_with_predictions_long = pd.DataFrame(long_rows)
        emails_with_predictions.to_csv(wide_output_path, index=False)
        emails_with_predictions_long.to_csv(long_output_path, index=False)

        elapsed = time.perf_counter() - start_time
        processed_this_run = processed_count - resume_from
        avg_iter = elapsed / processed_this_run if processed_this_run > 0 else 0.0
        print_progress(processed_count, total_emails, avg_iter_seconds=avg_iter)

    emails_with_predictions = pd.DataFrame(wide_rows)
    emails_with_predictions_long = pd.DataFrame(long_rows)

    emails_with_predictions.to_csv(wide_output_path, index=False)
    emails_with_predictions_long.to_csv(long_output_path, index=False)

    print(emails_with_predictions.head())
    print(emails_with_predictions_long.head())

if __name__ == "__main__":
    main()