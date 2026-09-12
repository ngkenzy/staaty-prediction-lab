STAATY v10 — Gemini Explanation Layer

SECURITY FIRST
--------------
Do NOT put your Gemini key inside app.py, app.js, HTML, GitHub, or this ZIP.

Because a key was pasted into chat, rotate/revoke that key in Google AI Studio
and create a new one before using the integration.

Install:
    cd /Users/nguyen/Documents/Codex/staaty_prediction_lab
    source venv/bin/activate
    pip install -U google-genai

Replace:
    app.py
    templates/index.html
    static/app.js
    static/style.css

Set the NEW key in the shell:
    export GEMINI_API_KEY="YOUR_NEW_KEY"

Optional model override:
    export GEMINI_MODEL="gemini-3.8-flash"

Run:
    python app.py

What v10 does
-------------
- Adds "ASK STAATY AI" beneath Basketball and UFC predictions.
- Sends the existing STAATY prediction output to Gemini.
- Gemini explains the model; it does NOT generate or alter probabilities.
- Adds quick questions:
    Why favored?
    Confidence?
    Upset risks?
    Why this UFC method?
    What could go wrong?

Security / architecture
-----------------------
Browser -> Flask /api/ask -> Gemini API

The Gemini key stays server-side in GEMINI_API_KEY.
It is never sent to app.js or the browser.

Official Google SDK:
    pip install -U google-genai
