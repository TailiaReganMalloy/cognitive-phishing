import os
import ast
import json
import time
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd 

load_dotenv()

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

def get_model():
    model = GenerativeModel(os.getenv("MODEL_NAME", "gemini-2.5-pro"))
    return model

biases = ['Authority Bias', 'Survivorship Bias',
       'Pessimism Bias', 'Zero-Risk Bias', 'Hyperbolic Discounting',
       'Identifiable Victim Effect', 'Appeal to Novelty', 'Urgency Effect',
    'Curiosity', 'Conformity',]

effects = {
    "Authority Bias": "Authority Bias is the tendency to trust and comply with messages from perceived authority figures without sufficient scrutiny. In phishing evaluation, this means users are more likely to trust emails that appear to come from IT departments, executives, banks, or government agencies, making them less likely to question the email's legitimacy.",
    "Survivorship Bias": "Survivorship Bias is the tendency to focus on successful or visible cases while overlooking failures, leading to skewed perception of risk. In phishing evaluation, users who have never been harmed by clicking suspicious links may underestimate the threat, reasoning that past safe outcomes mean future emails are equally safe.",
    "Pessimism Bias": "Pessimism Bias is the tendency to give more psychological weight to negative outcomes or threats than to neutral or positive ones. In phishing evaluation, this can be exploited by attackers who craft emails warning of account suspension, legal action, or security breaches, triggering fear that overrides rational scrutiny.",
    "Zero-Risk Bias": "Zero-Risk Bias is the preference for completely eliminating a small risk over substantially reducing a larger one. In phishing evaluation, users may click a malicious link to 'verify their account' because it feels like it eliminates the risk of losing access, even though it introduces a far greater security risk.",
    "Hyperbolic Discounting": "Hyperbolic Discounting is the tendency to favor immediate rewards over future benefits, even when the future benefit is objectively greater. In phishing evaluation, users may prioritize the immediate convenience of clicking a link to claim a prize or resolve an issue, discounting the longer-term risk of credential theft or malware infection.",
    "Identifiable Victim Effect": "The Identifiable Victim Effect is the tendency to respond more strongly to the plight of a specific, identifiable individual than to abstract or statistical victims. In phishing evaluation, attackers exploit this by crafting emails that reference a named colleague, friend, or family member in distress, making the threat feel personal and urgent enough to bypass critical thinking.",
    "Appeal to Novelty": "Appeal to Novelty is the tendency to value something more simply because it is presented as new, cutting-edge, or exclusive. In phishing evaluation, attackers exploit this by framing links, tools, or opportunities as brand-new releases or limited early access, making users more likely to click before verifying legitimacy.",
    "Urgency Effect": "The Urgency Effect is the cognitive tendency to prioritize time-pressured decisions, often at the expense of careful deliberation. In phishing evaluation, attackers weaponize this by embedding deadlines such as 'your account will be closed in 24 hours,' pushing users to act quickly rather than pause to verify the email's authenticity.",
    "Curiosity": "Curiosity is the tendency to seek missing or intriguing information, even when doing so carries risk. In phishing evaluation, attackers exploit this by using vague but enticing subjects such as 'see who viewed your profile' or 'important confidential file', prompting users to click links or open attachments to satisfy curiosity.",
    "Conformity": "Conformity is the tendency to align one's beliefs or actions with what others are perceived to be doing. In phishing evaluation, this is exploited through social proof cues in emails — such as 'thousands of users have already updated their credentials' — which make compliance feel normal and safe, reducing the user's likelihood of questioning the request.",
}


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

    for i, (effect_name, effect_definition) in enumerate(effects_dict.items(), start=1):
        ordinal = ordinal_words.get(i, f"{i}th")
        task_instruction += (
            f"The {ordinal} cognitive bias is the {effect_name} bias with the definition that "
            f"{effect_definition} "
        )

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
]

