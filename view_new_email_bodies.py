import json
import webbrowser
from pathlib import Path

import pandas as pd

CSV_PATH = Path("Emails_Formatted.csv")
HTML_PATH = Path("new_email_viewer.html")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    new_body_column = "New Email Body" if "New Email Body" in df.columns else "Body"
    if new_body_column not in df.columns:
        raise KeyError("Missing required column: New Email Body or Body")
    if "Body" not in df.columns:
        raise KeyError("Missing required column: Body")

    records = []
    for _, row in df.iterrows():
        new_body = "" if pd.isna(row.get(new_body_column)) else str(row.get(new_body_column))
        old_body = "" if pd.isna(row.get("GPT Email Body")) else str(row.get("GPT Email Body"))
        records.append(
            {
                "email_id": "" if pd.isna(row.get("EmailId")) else str(row.get("EmailId")),
                "author": "" if pd.isna(row.get("Author")) else str(row.get("Author")),
                "style": "" if pd.isna(row.get("Style")) else str(row.get("Style")),
                "type": "" if pd.isna(row.get("Type")) else str(row.get("Type")),
                "status": "" if pd.isna(row.get("Generation Status")) else str(row.get("Generation Status")),
                "new_body": new_body,
                "old_body": old_body,
            }
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Email Compare Viewer</title>
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #171a21;
      --text: #e8ebf2;
      --muted: #9aa4b2;
      --accent: #4cc9f0;
      --border: #2a3040;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top right, #1e2536, var(--bg));
      color: var(--text);
    }}
    .wrap {{
      max-width: 1300px;
      margin: 24px auto;
      padding: 0 16px;
      display: grid;
      gap: 12px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    button {{
      border: 1px solid var(--border);
      background: #222838;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
    }}
    button:hover {{
      border-color: var(--accent);
    }}
    input[type=\"number\"] {{
      width: 90px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0e1220;
      color: var(--text);
      padding: 7px 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      min-height: 68vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .card h3 {{
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }}
    iframe {{
      border: 0;
      width: 100%;
      min-height: 65vh;
      background: white;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"toolbar\">
      <button id=\"prevBtn\">← Prev</button>
      <button id=\"nextBtn\">Next →</button>
      <span class=\"meta\" id=\"indexLabel\"></span>
      <label class=\"meta\">Go to index:</label>
      <input id=\"jumpInput\" type=\"number\" min=\"1\" />
      <button id=\"jumpBtn\">Go</button>
      <span class=\"meta\">Use keyboard arrows left/right</span>
    </div>

    <div class=\"toolbar\">
      <span class=\"meta\" id=\"metaLabel\"></span>
    </div>

    <div class=\"grid\">
      <div class=\"card\">
        <h3>New Email (Left)</h3>
        <iframe id=\"newPreview\" sandbox=\"allow-popups allow-popups-to-escape-sandbox\"></iframe>
      </div>
      <div class=\"card\">
        <h3>Old Email (Right)</h3>
        <iframe id=\"oldPreview\" sandbox=\"allow-popups allow-popups-to-escape-sandbox\"></iframe>
      </div>
    </div>
  </div>

  <script>
    const emails = {json.dumps(records)};
    const STORAGE_KEY = 'email_viewer_last_position_v1';
    let idx = 0;

    const newPreview = document.getElementById('newPreview');
    const oldPreview = document.getElementById('oldPreview');
    const indexLabel = document.getElementById('indexLabel');
    const metaLabel = document.getElementById('metaLabel');
    const jumpInput = document.getElementById('jumpInput');

    function clampIndex(value) {{
      if (emails.length === 0) return 0;
      return Math.max(0, Math.min(emails.length - 1, value));
    }}

    function restoreIndex() {{
      if (emails.length === 0) return 0;
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return 0;
        const saved = JSON.parse(raw);

        // Prefer matching by EmailId in case row ordering changed.
        if (saved && saved.email_id) {{
          const found = emails.findIndex((row) => String(row.email_id) === String(saved.email_id));
          if (found >= 0) return found;
        }}

        if (saved && Number.isFinite(saved.idx)) {{
          return clampIndex(saved.idx);
        }}
      }} catch (_err) {{
        // Ignore malformed saved state and fall back to first row.
      }}
      return 0;
    }}

    function saveIndex() {{
      if (emails.length === 0) return;
      const row = emails[idx] || {{}};
      const payload = {{ idx, email_id: row.email_id || '' }};
      try {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      }} catch (_err) {{
        // Ignore storage errors (e.g., private mode restrictions).
      }}
    }}

    function render() {{
      if (emails.length === 0) {{
        indexLabel.textContent = 'No rows found in CSV.';
        metaLabel.textContent = '';
        newPreview.srcdoc = '<html><body><p>No rows found.</p></body></html>';
        oldPreview.srcdoc = '<html><body><p>No rows found.</p></body></html>';
        return;
      }}

      const row = emails[idx];
      const newHtml = row.new_body || '';
      const oldHtml = row.old_body || '';

      newPreview.srcdoc = newHtml;
      oldPreview.srcdoc = oldHtml;

      indexLabel.textContent = `Row ${{idx + 1}} / ${{emails.length}}`;
      metaLabel.textContent =
        `EmailId: ${{row.email_id || 'N/A'}} | Author: ${{row.author || 'N/A'}} | Style: ${{row.style || 'N/A'}} | Type: ${{row.type || 'N/A'}} | Status: ${{row.status || 'N/A'}}`;
      jumpInput.value = idx + 1;
      saveIndex();
    }}

    function next() {{
      idx = clampIndex(idx + 1);
      render();
    }}

    function prev() {{
      idx = clampIndex(idx - 1);
      render();
    }}

    document.getElementById('nextBtn').addEventListener('click', next);
    document.getElementById('prevBtn').addEventListener('click', prev);

    document.getElementById('jumpBtn').addEventListener('click', () => {{
      const value = Number(jumpInput.value);
      if (!Number.isFinite(value)) return;
      idx = clampIndex(value - 1);
      render();
    }});

    document.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowRight') next();
      if (event.key === 'ArrowLeft') prev();
    }});

    idx = restoreIndex();
    render();
  </script>
</body>
</html>
"""

    HTML_PATH.write_text(html, encoding="utf-8")
    webbrowser.open(HTML_PATH.resolve().as_uri())
    print(f"Opened viewer: {HTML_PATH}")


if __name__ == "__main__":
    main()
