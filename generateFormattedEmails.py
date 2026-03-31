import pandas as pd
from bs4 import BeautifulSoup
import re

EMAILS_GEMINI_PATH = "Emails_Gemini.csv"
EMAILS_FORMATTED_PATH = "Emails_Formatted.csv"

Emails_Gemini = pd.read_csv(EMAILS_GEMINI_PATH)

def body_to_paragraphs(body_text):
    if pd.isna(body_text):
        return []
    lines = [line.strip() for line in str(body_text).splitlines()]
    return [line for line in lines if line]


def append_html_fragment(parent_tag, html_fragment):
    fragment_soup = BeautifulSoup(html_fragment, "html.parser")
    nodes = list(fragment_soup.contents)
    if not nodes:
        return
    for node in nodes:
        parent_tag.append(node)

def replace_visible_text_keep_style(html_text, plaintext_body):
    if pd.isna(html_text) or not str(html_text).strip():
        return html_text

    soup = BeautifulSoup(str(html_text), "html.parser")

    # Prefer well-known content containers. Do not clear the whole body as that can remove layout.
    content_container = (
        soup.select_one(".email-content")
        or soup.select_one(".content")
        or soup.select_one(".main-content")
        or soup.select_one(".email-body")
        or soup.select_one(".message-content")
        or soup.select_one(".body-content")
    )
    if content_container is None:
        # Fallback: choose the most text-heavy non-header/footer container.
        best_candidate = None
        best_score = -1
        for tag in soup.find_all(["div", "section", "article", "main", "td"]):
            attrs_text = " ".join(
                [
                    " ".join(tag.get("class", [])),
                    str(tag.get("id", "")),
                ]
            ).lower()
            if any(blocked in attrs_text for blocked in ["header", "footer", "nav", "menu"]):
                continue

            text_len = len(tag.get_text(" ", strip=True))
            p_count = len(tag.find_all("p"))
            score = text_len + (p_count * 100)
            if score > best_score:
                best_score = score
                best_candidate = tag

        content_container = best_candidate
        if content_container is None:
            return html_text

    paragraphs = body_to_paragraphs(plaintext_body)
    content_container.clear()

    if not paragraphs:
        p = soup.new_tag("p")
        p.string = ""
        content_container.append(p)
    else:
        for line in paragraphs:
            # Preserve inline HTML from human-authored bodies (e.g. anchor tags).
            if re.search(r"<[a-zA-Z][^>]*>", line):
                p = soup.new_tag("p")
                append_html_fragment(p, line)
                content_container.append(p)
            else:
                p = soup.new_tag("p")
                p.string = line
                content_container.append(p)

    return str(soup)


def strip_html_fences_and_prefix(text_value):
    if pd.isna(text_value):
        return text_value

    text = str(text_value)
    opening_fence = re.search(r"```\s*html\b", text, flags=re.IGNORECASE)
    if opening_fence is not None:
        # Drop any model commentary before the first HTML fence token.
        text = text[opening_fence.start():]

    # Remove fence lines entirely, including trailing whitespace/newline.
    text = re.sub(r"(?im)^[ \t]*```\s*html\b[ \t]*\r?\n?", "", text)
    text = re.sub(r"(?im)^[ \t]*```[ \t]*\r?\n?", "", text)

    # If any inline fence token remains, strip it as well.
    text = re.sub(r"```\s*html\b", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # Trim to first actual html tag to avoid a leading spacer where fences were.
    html_start = re.search(r"(?is)\s*(<!doctype\s+html|<html\b)", text)
    if html_start is not None:
        text = text[html_start.start():]

    # Remove leading blank lines left by token cleanup.
    text = re.sub(r"\A(?:[ \t]*\r?\n)+", "", text)
    return text.strip()

if "Author" not in Emails_Gemini.columns or "Body" not in Emails_Gemini.columns or "New Email Body" not in Emails_Gemini.columns:
    raise KeyError("Expected columns Author, Body, and New Email Body were not found.")

human_mask = Emails_Gemini["Author"].astype(str).str.lower().eq("human")
updated_count = 0
unchanged_count = 0

for idx in Emails_Gemini.index[human_mask]:
    template_html = Emails_Gemini.at[idx, "New Email Response"]
    if pd.isna(template_html) or not str(template_html).strip():
        template_html = Emails_Gemini.at[idx, "New Email Body"]

    original_html = Emails_Gemini.at[idx, "New Email Body"]
    plaintext_body = Emails_Gemini.at[idx, "Body"]
    transformed_html = replace_visible_text_keep_style(template_html, plaintext_body)

    if transformed_html != original_html:
        Emails_Gemini.at[idx, "New Email Body"] = transformed_html
        updated_count += 1
    else:
        unchanged_count += 1

Emails_Gemini.to_csv(EMAILS_FORMATTED_PATH, index=False)
print(
    f"Updated {updated_count} human-authored rows in New Email Body, "
    f"left {unchanged_count} unchanged, and saved {EMAILS_FORMATTED_PATH}"
)

# Clean fenced html in New Email Body only.
original_new_body = Emails_Gemini["New Email Body"]
Emails_Gemini["New Email Body"] = Emails_Gemini["New Email Body"].map(strip_html_fences_and_prefix)
cleaned_cells = int((Emails_Gemini["New Email Body"] != original_new_body).fillna(False).sum())

Emails_Gemini.to_csv(EMAILS_FORMATTED_PATH, index=False)
print(
    f"Cleaned markdown html fences in New Email Body and saved "
    f"{EMAILS_FORMATTED_PATH} (changed rows: {cleaned_cells})"
)