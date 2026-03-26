import os
import ast
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd 

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
except ModuleNotFoundError:
    vertexai = None
    GenerativeModel = None

load_dotenv()

def get_model():
    if vertexai is None or GenerativeModel is None:
        raise ModuleNotFoundError(
            "vertexai is required for get_model(). Install google-cloud-aiplatform to use this path."
        )

    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    model = GenerativeModel(os.getenv("MODEL_NAME", "gemini-2.5-pro")) # gemini-3.1-pro-preview
    return model

def _load_effects_from_json():
    repo_root = Path(__file__).resolve().parent.parent
    definitions_path = repo_root / "Data/Biases/bias_definitions.json"

    with definitions_path.open("r", encoding="utf-8") as f:
        raw_definitions = json.load(f)

    normalized = {}
    for bias_name, info in raw_definitions.items():
        if not isinstance(info, dict):
            continue

        definition = str(info.get("definition", "")).strip()
        phishing_context = str(info.get("phishing_context", "")).strip()
        examples = info.get("examples", [])

        if not isinstance(examples, list):
            examples = [str(examples)]

        cleaned_examples = [str(example).strip() for example in examples if str(example).strip()]

        normalized[str(bias_name).strip()] = {
            "definition": definition,
            "phishing_context": phishing_context,
            "examples": cleaned_examples,
        }

    return normalized


effects = _load_effects_from_json()
biases = list(effects.keys())


def get_instruction(email_row, effects_dict):
    task_instruction = ""
    ordinal_words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }

    for i, (effect_name, effect_info) in enumerate(effects_dict.items(), start=1):
        ordinal = ordinal_words.get(i, f"{i}th")
        if isinstance(effect_info, dict):
            effect_definition = str(effect_info.get("definition", "")).strip()
            phishing_context = str(effect_info.get("phishing_context", "")).strip()
            examples = effect_info.get("examples", [])
            if not isinstance(examples, list):
                examples = [str(examples)]
            examples_text = " | ".join(str(example).strip() for example in examples if str(example).strip())
        else:
            effect_definition = str(effect_info).strip()
            phishing_context = ""
            examples_text = ""

        task_instruction += (
            f"The {ordinal} cognitive bias is the {effect_name} bias with the definition that "
            f"{effect_definition} "
        )
        if phishing_context:
            task_instruction += f"In phishing context: {phishing_context} "
        if examples_text:
            task_instruction += f"Examples: {examples_text} "

    task_instruction += (
        "Reply only with a Python dictionary where each key is the exact effect name and each value "
        "is a float between 0 and 1 indicating the extent to which the input email corresponds to "
        "that effect. Do not include any additional text, explanations, or code fences."
    )
    # [0.2, 0.1, .9 ]

    available_columns = set(email_row.index)

    email_description_parts = []

    for base_col in ["Sender", "Subject", "Type", "Body", "text", "text_translated"]:
        if base_col in available_columns:
            value = email_row[base_col]
            if pd.notna(value):
                email_description_parts.append(f"{base_col}: {value}.")

    flag_templates = {
        "Sender Mismatch": ("The email has a sender mismatch.", "The email does not have a sender mismatch."),
        "Request Credentials": ("The email requests credentials.", "The email does not request credentials."),
        "Subject Suspicious": ("The email has a suspicious subject.", "The email does not have a suspicious subject."),
        "Urgent": ("The email has an urgent tone.", "The email does not have an urgent tone."),
        "Offer": ("The email contains an offer.", "The email does not contain an offer."),
        "Link Mismatch": ("The email has a link mismatch.", "The email does not have a link mismatch."),
    }

    for col_name, (true_text, false_text) in flag_templates.items():
        if col_name in available_columns and pd.notna(email_row[col_name]):
            try:
                is_true = float(email_row[col_name]) == 1.0
            except Exception:
                is_true = str(email_row[col_name]).strip() == "1"
            email_description_parts.append(true_text if is_true else false_text)

    email_description = " ".join(email_description_parts)

    return f"{task_instruction}\n\nEmail to evaluate:\n{email_description}"


def parse_effect_scores(response_text, effects_dict):
    text = (response_text or "").strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("python"):
            text = text[len("python"):].strip()
        elif text.startswith("json"):
            text = text[len("json"):].strip()

    parsed = None
    parse_error = ""

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception as exc:
            parse_error = str(exc)

    scores = {}
    for effect_name in effects_dict:
        value = None
        if isinstance(parsed, dict) and effect_name in parsed:
            try:
                value = float(parsed[effect_name])
                value = max(0.0, min(1.0, value))
            except Exception:
                value = None
        scores[effect_name] = value

    return scores, parse_error


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def print_progress(processed, total, avg_iter_seconds, bar_width=40):
    if total == 0:
        print("No emails to process.")
        return

    progress_ratio = processed / total
    filled_width = int(bar_width * progress_ratio)
    bar = "█" * filled_width + "-" * (bar_width - filled_width)
    remaining = total - processed
    eta_seconds = avg_iter_seconds * remaining
    avg_text = f"{avg_iter_seconds:.2f}s/it"
    eta_text = format_seconds(eta_seconds)
    print(
        f"\rProcessing emails with LLM: |{bar}| {processed}/{total} | {avg_text} | ETA {eta_text}",
        end="",
        flush=True,
    )

    if processed == total:
        print()


def is_resource_exhausted_429(exc):
    message = str(exc)
    return "429" in message and ("ResourceExhausted" in message or "Resource exhausted" in message)


def generate_with_retry(model_obj, prompt):
    wait_seconds = 10
    while True:
        try:
            return model_obj.generate_content(prompt)
        except Exception as exc:
            if is_resource_exhausted_429(exc):
                print(
                    f"\nRate limit hit (429 ResourceExhausted). Retrying in {wait_seconds} seconds...",
                    flush=True,
                )
                time.sleep(wait_seconds)
                wait_seconds += 10
                continue
            raise

def parse_llm_raw_response(raw_text):
    text = str(raw_text).strip()

    # Handle markdown fenced JSON responses from model output.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)

    if not isinstance(parsed, dict):
        raise ValueError("LLM Raw Response is not a dictionary")

    return parsed

def get_max_biases(num_biases, bias_scores):
    if not isinstance(num_biases, int) or num_biases <= 0:
        raise ValueError("num_biases must be a positive integer")
    if not isinstance(bias_scores, dict):
        raise ValueError("bias_scores must be a dictionary")

    numeric_scores = {}
    for bias_name, score in bias_scores.items():
        try:
            numeric_scores[bias_name] = float(score)
        except (TypeError, ValueError):
            continue

    if not numeric_scores or max(numeric_scores.values()) <= 0:
        return None

    top_items = sorted(
        numeric_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[: min(num_biases, len(numeric_scores))]
    return dict(top_items)

__all__ = [
    "get_model",
    "biases",
    "effects",
    "get_instruction",
    "parse_effect_scores",
    "format_seconds",
    "print_progress",
    "is_resource_exhausted_429",
    "generate_with_retry",
    "parse_llm_raw_response",
    "get_max_biases"
]

