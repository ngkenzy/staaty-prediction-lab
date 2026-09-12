STAATY v15 — Edge Meter + Confidence Calibration

Adds:
1. Edge Meter comparing market spread vs STAATY model line.
2. Edge direction: which team the model favors relative to the market.
3. Signal levels:
      <1.5 pts  NO EDGE
      1.5–3     SMALL EDGE
      3–5       MODERATE EDGE
      5+        STRONG EDGE
4. Validation-derived confidence calibration.
5. Prediction cards can say:
      Strong
      Validation hit rate in this band: 71.2% (188 games)

Important methodology:
- Confidence calibration is computed from the 15% VALIDATION set.
- The final 15% test remains untouched.
- The spread Edge Meter compares model-implied line to a user-entered market line.
- Cover probability remains an approximation based on validation residuals.
- Do not present spread output as a betting guarantee.

Install:
Replace:
    train_basketball.py
    app.py
    templates/index.html
    static/app.js
    static/style.css

Keep:
    train_ufc.py
    data/
    venv/

Retrain basketball:
    cd /Users/nguyen/Documents/Codex/staaty_prediction_lab
    source venv/bin/activate
    python train_basketball.py

Then:
    python app.py

After retraining, paste the output into ChatGPT if you want the final
competition numbers checked again.
