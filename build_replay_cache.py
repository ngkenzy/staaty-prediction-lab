
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/"models"
MODELS.mkdir(exist_ok=True)

bbmeta=json.loads((MODELS/"basketball_meta.json").read_text()) if (MODELS/"basketball_meta.json").exists() else {"metrics":{}}
ufcmeta=json.loads((MODELS/"ufc_meta.json").read_text()) if (MODELS/"ufc_meta.json").exists() else {"metrics":{}}

basketball=[]
for league,m in bbmeta.get("metrics",{}).items():
    basketball.append({
        "type":"summary",
        "league":league,
        "accuracy":m.get("accuracy"),
        "test_games":m.get("test_games"),
        "home_baseline_accuracy":m.get("home_baseline_accuracy"),
        "elo_accuracy":m.get("elo_accuracy"),
        "champion":m.get("champion"),
        "note":"Chronological final holdout summary. Individual historical games are not reconstructed here to avoid fabricating per-game predictions."
    })

ufc=[]
for target,m in ufcmeta.get("metrics",{}).items():
    n=m.get("n")
    test_fights=(int(n)-int(int(n)*.85)) if n else None
    ufc.append({
        "type":"summary",
        "target":target,
        "accuracy":m.get("accuracy"),
        "test_fights":test_fights,
        "macro_f1":m.get("macro_f1"),
        "log_loss":m.get("log_loss"),
        "note":"Chronological final holdout summary. Individual historical fights are not reconstructed here to avoid fabricating per-fight predictions."
    })

(MODELS/"basketball_replay.json").write_text(json.dumps(basketball,indent=2))
(MODELS/"ufc_replay.json").write_text(json.dumps(ufc,indent=2))

print("Wrote:")
print(" ", MODELS/"basketball_replay.json")
print(" ", MODELS/"ufc_replay.json")
