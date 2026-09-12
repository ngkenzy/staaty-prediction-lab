STAATY v9 — Model Proof Screen

Replace these files in your existing `staaty_prediction_lab` project:

    app.py
    templates/index.html
    static/app.js
    static/style.css

Keep everything else, including:
    data/
    models/
    venv/
    train_basketball.py
    train_ufc.py

No retraining is required for this UI update.

Run:
    cd /Users/nguyen/Documents/Codex/staaty_prediction_lab
    source venv/bin/activate
    python app.py

Then open:
    http://127.0.0.1:5000

New tab:
    Model Proof

It reads your existing basketball_meta.json and ufc_meta.json
and displays:
- held-out accuracy
- final test sample size
- basketball home and Elo baselines
- lift over baseline
- UFC margin over the sponsor's 50% threshold
- log loss
- ROC-AUC for basketball where available
- macro-F1 for UFC method/round
- evaluation protocol
- explicit UFC scheduled-round leakage fix

Important:
This update does not change model training or reported metrics.
