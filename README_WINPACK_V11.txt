STAATY v11 — Win Pack

Adds:
1. Upset Radar
2. Historical Replay / Holdout Evidence
3. Zero-cost deterministic explanation engine

Replace:
    app.py
    templates/index.html
    static/app.js
    static/style.css

Also add:
    build_replay_cache.py

Keep:
    data/
    models/
    venv/
    train_basketball.py
    train_ufc.py

Run once:
    python build_replay_cache.py

Then:
    python app.py

Notes:
- The deterministic explainer costs $0 and does not require Gemini.
- It explains existing model output; it never changes probabilities.
- Upset Radar uses the latest saved team states.
- Historical Replay currently shows verified held-out summaries. It does NOT fabricate individual historical-game predictions.
  To show exact per-game replay later, save each final-holdout prediction during training.
