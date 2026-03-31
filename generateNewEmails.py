import json
import os
from datetime import datetime, timezone

import pandas as pd
from Utilities.CognitivePhishingRAG import CognitivePhishingRAG
from Utilities.Utilities import parse_llm_raw_response, get_max_biases
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Load environment variables from .env file
load_dotenv()

EMAILS_CHOSEN_PATH = "./Data/Emails_Chosen.csv"
EMAILS_PATH = "./Data/Emails.csv"
CHECKPOINT_PATH = "Emails_Gemini.csv"

Emails_Chosen = pd.read_csv(EMAILS_CHOSEN_PATH)
Emails = pd.read_csv(EMAILS_PATH)
definitions_path = "Data/Biases/bias_definitions.json"
project  = os.getenv("GOOGLE_CLOUD_PROJECT")
api_key  = os.getenv("GOOGLE_API_KEY")
location = os.getenv("GOOGLE_API_LOCATION") or "global"
model = os.getenv("GOOGLE_MODEL_NAME") or "gemini-3.1-pro-preview"

generated_columns = [
    "New Email Body",
    "New Email Response",
    "New Email Prompt",
    "New Email Config",
    "New Email Model Name",
    "Target Biases Used",
    "Generation Status",
    "Generation Error",
    "Generated At UTC",
]

for col in generated_columns:
    if col not in Emails_Chosen.columns:
        Emails_Chosen[col] = pd.NA


def _serialize_config(config):
    if hasattr(config, "model_dump"):
        return json.dumps(config.model_dump(mode="json"), ensure_ascii=True)
    return str(config)


if os.path.exists(CHECKPOINT_PATH):
    checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
    if "EmailId" in Emails_Chosen.columns and "EmailId" in checkpoint_df.columns:
        checkpoint_cols = ["EmailId"] + [c for c in generated_columns if c in checkpoint_df.columns]
        Emails_Chosen = Emails_Chosen.merge(
            checkpoint_df[checkpoint_cols],
            on="EmailId",
            how="left",
            suffixes=("", "_checkpoint"),
        )
        for col in generated_columns:
            checkpoint_col = f"{col}_checkpoint"
            if checkpoint_col in Emails_Chosen.columns:
                Emails_Chosen[col] = Emails_Chosen[col].combine_first(Emails_Chosen[checkpoint_col])
                Emails_Chosen.drop(columns=[checkpoint_col], inplace=True)
    elif len(checkpoint_df) == len(Emails_Chosen):
        for col in generated_columns:
            if col in checkpoint_df.columns:
                Emails_Chosen[col] = checkpoint_df[col]
    else:
        print("Checkpoint found but could not align rows safely; starting from source CSV.")

EmailGenerator = CognitivePhishingRAG(
    dataset=Emails_Chosen,
    definitions_path=definitions_path,
    project = project,
    api_key = api_key,
    location=location
)

pending_mask = Emails_Chosen["Generation Status"].fillna("").ne("completed")
pending_indices = Emails_Chosen.index[pending_mask]
print(f"Resuming with {len(pending_indices)} pending emails out of {len(Emails_Chosen)} total.")

for idx in tqdm(pending_indices, total=len(pending_indices), desc="Generating emails"):
    Email = Emails_Chosen.loc[idx]
    base_type = Email['Type']
    formatted_email = Emails[
        (Emails['BaseEmailID'] == Email['BaseEmailID'])
        & (Emails['Author'] == Email['Author'])
        & (Emails['Style'] == 'GPT')
    ]

    if formatted_email.empty:
        Emails_Chosen.at[idx, "Generation Status"] = "skipped_no_base_email"
        Emails_Chosen.at[idx, "Generation Error"] = "No matching base email found."
        Emails_Chosen.to_csv(CHECKPOINT_PATH, index=False)
        continue

    base_body = formatted_email['Body'].iloc[0]
    base_prompt = "Generate HTML code for an email with a header and banner and a logo where appropriate."

    try:
        parsed_response = parse_llm_raw_response(Email["LLM Raw Response"])
        target_biases = get_max_biases(2, parsed_response)

        generation_result = EmailGenerator.generate_email(
            base_type=base_type,
            base_body=base_body,
            base_prompt=base_prompt,
            target_biases=target_biases,
            model_name=model
        )

        # Backward compatibility: support both tuple and plain-text return signatures.
        if isinstance(generation_result, tuple):
            if len(generation_result) == 4:
                response, system_prompt, config, model_name = generation_result
            elif len(generation_result) == 1:
                response = generation_result[0]
                system_prompt = base_prompt
                config = ""
                model_name = model
            else:
                raise ValueError(f"Unexpected generate_email return length: {len(generation_result)}")
        else:
            response = generation_result
            system_prompt = base_prompt
            config = ""
            model_name = model

        response_text = response.text if response is not None else ""
        Emails_Chosen.at[idx, "New Email Body"] = response_text
        Emails_Chosen.at[idx, "New Email Response"] = response_text
        Emails_Chosen.at[idx, "New Email Prompt"] = system_prompt
        Emails_Chosen.at[idx, "New Email Config"] = _serialize_config(config)
        Emails_Chosen.at[idx, "New Email Model Name"] = model_name
        Emails_Chosen.at[idx, "Target Biases Used"] = json.dumps(target_biases, ensure_ascii=True) if target_biases is not None else "{}"
        Emails_Chosen.at[idx, "Generation Status"] = "completed"
        Emails_Chosen.at[idx, "Generation Error"] = ""
        Emails_Chosen.at[idx, "Generated At UTC"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        Emails_Chosen.at[idx, "Generation Status"] = "failed"
        Emails_Chosen.at[idx, "Generation Error"] = str(exc)
    finally:
        # Checkpoint after each email so runs can resume safely.
        Emails_Chosen.to_csv(CHECKPOINT_PATH, index=False)

print(f"Done. Checkpoint saved to {CHECKPOINT_PATH}")