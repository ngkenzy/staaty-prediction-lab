
from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def pair_parts(s):
    if pd.isna(s):
        return np.nan, np.nan
    m = re.search(r"(\d+)\s+of\s+(\d+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

def ctrl_seconds(s):
    if pd.isna(s):
        return np.nan
    try:
        mm, ss = str(s).split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return np.nan

def height_inches(s):
    if pd.isna(s) or str(s).strip() == "--":
        return np.nan
    m = re.search(r"(\d+)'\s*(\d+)", str(s))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan

def number_from_text(s):
    if pd.isna(s) or str(s).strip() == "--":
        return np.nan
    m = re.search(r"(\d+)", str(s))
    return float(m.group(1)) if m else np.nan

def split_bout(bout):
    p = str(bout).split(" vs. ")
    return (p[0].strip(), p[1].strip()) if len(p) == 2 else (None, None)

def method_group(x):
    x = str(x)
    if "KO/TKO" in x or "Doctor" in x:
        return "KO_TKO"
    if "Submission" in x:
        return "SUBMISSION"
    if "Decision" in x:
        return "DECISION"
    return "OTHER"

def round_group(x):
    try:
        r = int(x)
        return str(r) if r <= 3 else "4_5"
    except Exception:
        return "OTHER"

def finish_seconds(round_num, time_str):
    try:
        r = int(round_num)
        mm, ss = str(time_str).split(":")
        return (r - 1) * 300 + int(mm) * 60 + int(ss)
    except Exception:
        return np.nan

def clean_weightclass(x):
    return str(x).replace(" Bout", "").strip()

def weight_class_lbs(x):
    x = clean_weightclass(x)
    mapping = {
        "Women's Strawweight": 115,
        "Women's Flyweight": 125,
        "Women's Bantamweight": 135,
        "Flyweight": 125,
        "Bantamweight": 135,
        "Featherweight": 145,
        "Lightweight": 155,
        "Welterweight": 170,
        "Middleweight": 185,
        "Light Heavyweight": 205,
        "Heavyweight": 265,
        "Catch Weight": 180,
        "Open Weight": 205,
    }
    return mapping.get(x, 155)

def multi_metrics(y, p, classes):
    classes = np.asarray(classes)
    pred = classes[p.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, p, labels=classes)),
    }

# -----------------------------
# Load / clean
# -----------------------------
results = pd.read_csv(DATA / "ufc_fight_results.csv", low_memory=False)
stats = pd.read_csv(DATA / "ufc_fight_stats.csv", low_memory=False)
events = pd.read_csv(DATA / "ufc_event_details.csv", low_memory=False)
attrs = pd.read_csv(DATA / "ufc_fighter_tott.csv", low_memory=False)

events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")
results = (
    results.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
    .sort_values(["DATE", "EVENT", "BOUT"])
)

ab = results["BOUT"].apply(split_bout)
results["A"] = [x[0] for x in ab]
results["B"] = [x[1] for x in ab]
results = results[
    results["A"].notna()
    & results["B"].notna()
    & results["DATE"].notna()
].copy()

results["A_RESULT"] = results["OUTCOME"].astype(str).str.split("/").str[0]
results["B_RESULT"] = results["OUTCOME"].astype(str).str.split("/").str[-1]
results = results[
    results["A_RESULT"].isin(["W", "L"])
    & results["B_RESULT"].isin(["W", "L"])
].copy()

results["target_win"] = (results["A_RESULT"] == "W").astype(int)
results["target_method"] = results["METHOD"].map(method_group)
results["target_round"] = results["ROUND"].map(round_group)

# IMPORTANT FIX:
# Scheduled five-round status must come from TIME FORMAT, NOT actual finishing round.
results["scheduled_5_round"] = (
    results["TIME FORMAT"]
    .astype(str)
    .str.contains(r"^5 Rnd", regex=True)
    .astype(int)
)

results["weight_class_lbs"] = results["WEIGHTCLASS"].map(weight_class_lbs)
results["womens_division"] = (
    results["WEIGHTCLASS"].astype(str).str.contains("Women's").astype(int)
)
results["fight_duration_sec"] = [
    finish_seconds(r, t)
    for r, t in zip(results["ROUND"], results["TIME"])
]

# -----------------------------
# Round-stat history
# -----------------------------
for c in ["KD", "SUB.ATT", "REV."]:
    stats[c] = pd.to_numeric(stats[c], errors="coerce")

for c in [
    "SIG.STR.", "TOTAL STR.", "TD",
    "HEAD", "BODY", "LEG",
    "DISTANCE", "CLINCH", "GROUND"
]:
    parts = stats[c].map(pair_parts)
    stats[c + "_landed"] = [x[0] for x in parts]
    stats[c + "_attempted"] = [x[1] for x in parts]

stats["ctrl_sec"] = stats["CTRL"].map(ctrl_seconds)

aggcols = ["KD", "SUB.ATT", "REV.", "ctrl_sec"]
for c in [
    "SIG.STR.", "TOTAL STR.", "TD",
    "HEAD", "BODY", "LEG",
    "DISTANCE", "CLINCH", "GROUND"
]:
    aggcols += [c + "_landed", c + "_attempted"]

fighter_fight = (
    stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[aggcols]
    .sum(min_count=1)
    .merge(results[["EVENT", "BOUT", "DATE"]], on=["EVENT", "BOUT"], how="left")
    .sort_values(["FIGHTER", "DATE"])
)

histcols = []

# Keep career AND recent history. Method/round depend on absolute style,
# not only on A-B differences.
for c in aggcols:
    career = ("career_" + c).replace(".", "_")
    recent = ("recent_" + c).replace(".", "_")

    fighter_fight[career] = fighter_fight.groupby("FIGHTER")[c].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    fighter_fight[recent] = fighter_fight.groupby("FIGHTER")[c].transform(
        lambda x: x.shift(1).ewm(span=6, adjust=False, min_periods=1).mean()
    )
    histcols += [career, recent]

fighter_fight["hist_fights"] = fighter_fight.groupby("FIGHTER").cumcount()
fighter_fight["prev_date"] = fighter_fight.groupby("FIGHTER")["DATE"].shift(1)
fighter_fight["days_since"] = (
    fighter_fight["DATE"] - fighter_fight["prev_date"]
).dt.days.clip(0, 2000)
histcols += ["hist_fights", "days_since"]

# -----------------------------
# Historical outcomes / finish tendencies
# -----------------------------
long_rows = []

for r in results.itertuples(index=False):
    duration = r.fight_duration_sec

    long_rows.append([
        r.EVENT, r.BOUT, r.DATE, r.A,
        1 if r.A_RESULT == "W" else 0,
        r.target_method, r.ROUND, duration
    ])

    long_rows.append([
        r.EVENT, r.BOUT, r.DATE, r.B,
        1 if r.B_RESULT == "W" else 0,
        r.target_method, r.ROUND, duration
    ])

long = pd.DataFrame(
    long_rows,
    columns=[
        "EVENT", "BOUT", "DATE", "FIGHTER",
        "win", "method", "round", "duration"
    ],
).sort_values(["FIGHTER", "DATE"])

definitions = {
    "career_win_rate": long["win"],
    "career_ko_rate": (long["method"] == "KO_TKO").astype(int),
    "career_sub_rate": (long["method"] == "SUBMISSION").astype(int),
    "career_dec_rate": (long["method"] == "DECISION").astype(int),
    "career_r1_finish": (
        (pd.to_numeric(long["round"], errors="coerce") == 1)
        & (long["method"] != "DECISION")
    ).astype(int),
}

for name, values in definitions.items():
    long[name] = values.groupby(long["FIGHTER"]).transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

recent_defs = {
    "recent_win_rate": long["win"],
    "recent_ko_rate": (long["method"] == "KO_TKO").astype(int),
    "recent_sub_rate": (long["method"] == "SUBMISSION").astype(int),
    "recent_dec_rate": (long["method"] == "DECISION").astype(int),
    "recent_r1_finish": (
        (pd.to_numeric(long["round"], errors="coerce") == 1)
        & (long["method"] != "DECISION")
    ).astype(int),
}

for name, values in recent_defs.items():
    long[name] = values.groupby(long["FIGHTER"]).transform(
        lambda x: x.shift(1).ewm(span=6, adjust=False, min_periods=1).mean()
    )

long["win_last3"] = long.groupby("FIGHTER")["win"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
long["win_last5"] = long.groupby("FIGHTER")["win"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
long["avg_duration_sec"] = long.groupby("FIGHTER")["duration"].transform(
    lambda x: x.shift(1).ewm(span=6, adjust=False, min_periods=1).mean()
)

ratecols = [
    "career_win_rate", "career_ko_rate", "career_sub_rate",
    "career_dec_rate", "career_r1_finish",
    "recent_win_rate", "recent_ko_rate", "recent_sub_rate",
    "recent_dec_rate", "recent_r1_finish",
    "win_last3", "win_last5", "avg_duration_sec",
]

fighter_fight = fighter_fight.merge(
    long[["EVENT", "BOUT", "FIGHTER"] + ratecols],
    on=["EVENT", "BOUT", "FIGHTER"],
    how="left",
)
histcols += ratecols

# -----------------------------
# Pre-fight Fighter Elo
# -----------------------------
ratings = {}
elo_rows = []

for r in results.sort_values(["DATE", "EVENT", "BOUT"]).itertuples(index=False):
    ra = ratings.get(r.A, 1500.0)
    rb = ratings.get(r.B, 1500.0)
    pa = 1 / (1 + 10 ** ((rb - ra) / 400.0))

    elo_rows.append([r.EVENT, r.BOUT, ra, rb])

    y = 1 if r.A_RESULT == "W" else 0
    delta = 24 * (y - pa)

    ratings[r.A] = ra + delta
    ratings[r.B] = rb - delta

elo = pd.DataFrame(
    elo_rows,
    columns=["EVENT", "BOUT", "A_elo", "B_elo"]
)

results = results.merge(elo, on=["EVENT", "BOUT"], how="left")

# -----------------------------
# Physical attributes / stance
# -----------------------------
attrs["height_in"] = attrs["HEIGHT"].map(height_inches)
attrs["weight_lb"] = attrs["WEIGHT"].map(number_from_text)
attrs["reach_in"] = attrs["REACH"].map(number_from_text)
attrs["dob"] = pd.to_datetime(attrs["DOB"], errors="coerce")

attr_map = attrs.set_index("FIGHTER")[
    ["height_in", "weight_lb", "reach_in", "dob", "STANCE"]
]

def side_frame(side):
    m = results[["EVENT", "BOUT", "DATE", side]].rename(
        columns={side: "FIGHTER"}
    )

    x = (
        m.merge(
            fighter_fight[
                ["EVENT", "BOUT", "FIGHTER"] + histcols
            ],
            on=["EVENT", "BOUT", "FIGHTER"],
            how="left",
        )
        .join(attr_map, on="FIGHTER")
    )

    x["age_days"] = (x["DATE"] - x["dob"]).dt.days

    cols = histcols + [
        "height_in", "weight_lb", "reach_in", "age_days"
    ]

    out = x[
        ["EVENT", "BOUT", "FIGHTER", "STANCE"] + cols
    ].copy()

    out = out.rename(
        columns={
            "FIGHTER": side,
            "STANCE": side + "_stance",
            **{c: f"{side}_{c}" for c in cols},
        }
    )

    return out, cols

A, statecols = side_frame("A")
B, _ = side_frame("B")

d = (
    results.merge(A, on=["EVENT", "BOUT", "A"], how="left")
    .merge(B, on=["EVENT", "BOUT", "B"], how="left")
)

features = []

# Both difference AND sum features:
# diff -> who is better
# sum  -> what style/finish environment the matchup creates
for c in statecols:
    av = pd.to_numeric(d["A_" + c], errors="coerce")
    bv = pd.to_numeric(d["B_" + c], errors="coerce")

    d["diff_" + c] = av - bv
    d["sum_" + c] = av + bv

    features += ["diff_" + c, "sum_" + c]

d["elo_diff"] = d["A_elo"] - d["B_elo"]
d["elo_sum"] = d["A_elo"] + d["B_elo"]
features += ["elo_diff", "elo_sum"]

# Stance features
for side in ["A", "B"]:
    stance = d[side + "_stance"].fillna("Unknown").astype(str)
    for s in ["Orthodox", "Southpaw", "Switch"]:
        col = f"{side}_{s.lower()}"
        d[col] = (stance == s).astype(int)
        features.append(col)

d["same_stance"] = (
    d["A_stance"].fillna("Unknown")
    == d["B_stance"].fillna("Unknown")
).astype(int)

features.append("same_stance")
features += [
    "weight_class_lbs",
    "womens_division",
    "scheduled_5_round",
]

# -----------------------------
# Chronological train / validate / test
# -----------------------------
d = d.sort_values("DATE").reset_index(drop=True)

X = d[features].replace([np.inf, -np.inf], np.nan)

n = len(d)
cut1 = int(n * 0.70)
cut2 = int(n * 0.85)

medians = X.iloc[:cut1].median(numeric_only=True).fillna(0)
X = X.fillna(medians).fillna(0)

Xtr = X.iloc[:cut1]
Xv = X.iloc[cut1:cut2]
Xt = X.iloc[cut2:]

weights = np.linspace(0, 1, 41)
metrics = {}

# -----------------------------
# Winner
# -----------------------------
yw = d["target_win"].astype(int)
ywtr, ywv, ywt = yw.iloc[:cut1], yw.iloc[cut1:cut2], yw.iloc[cut2:]

wlog = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=3500, C=0.7)),
])

wgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.035,
    max_leaf_nodes=16,
    min_samples_leaf=20,
    l2_regularization=3.0,
    random_state=42,
)

wlog.fit(Xtr, ywtr)
wgb.fit(Xtr, ywtr)

pv1 = wlog.predict_proba(Xv)[:, 1]
pv2 = wgb.predict_proba(Xv)[:, 1]

ww = min(
    weights,
    key=lambda q: log_loss(
        ywv,
        q * pv1 + (1 - q) * pv2,
        labels=[0, 1],
    ),
)

pt = (
    ww * wlog.predict_proba(Xt)[:, 1]
    + (1 - ww) * wgb.predict_proba(Xt)[:, 1]
)

metrics["winner"] = {
    "accuracy": float(accuracy_score(ywt, pt >= 0.5)),
    "log_loss": float(log_loss(ywt, pt, labels=[0, 1])),
    "n": int(n),
}

joblib.dump(
    {
        "type": "binary",
        "logit": wlog,
        "gb": wgb,
        "weight": float(ww),
        "features": features,
        "medians": medians.to_dict(),
    },
    MODELS / "ufc_winner.joblib",
)

# -----------------------------
# Method
# Compare unbalanced vs balanced.
# Select on validation accuracy, tie-break macro-F1.
# -----------------------------
ym = d["target_method"].copy()
ymtr, ymv, ymt = ym.iloc[:cut1], ym.iloc[cut1:cut2], ym.iloc[cut2:]

method_candidates = []

for balanced in [False, True]:
    class_weight = "balanced" if balanced else None

    mlog = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=4000,
                C=0.7,
                class_weight=class_weight,
            ),
        ),
    ])

    mgb = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.03,
        max_leaf_nodes=18,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=42,
    )

    mlog.fit(Xtr, ymtr)
    mgb.fit(Xtr, ymtr)

    classes = mlog.named_steps["model"].classes_

    p1 = mlog.predict_proba(Xv)
    p2 = mgb.predict_proba(Xv)

    mw = min(
        weights,
        key=lambda q: log_loss(
            ymv,
            q * p1 + (1 - q) * p2,
            labels=classes,
        ),
    )

    pv = mw * p1 + (1 - mw) * p2
    vm = multi_metrics(ymv, pv, classes)

    method_candidates.append({
        "balanced": balanced,
        "logit": mlog,
        "gb": mgb,
        "weight": float(mw),
        "classes": classes,
        "validation": vm,
    })

method_champion = max(
    method_candidates,
    key=lambda x: (
        x["validation"]["accuracy"],
        x["validation"]["macro_f1"],
    ),
)

mc = method_champion
mpt = (
    mc["weight"] * mc["logit"].predict_proba(Xt)
    + (1 - mc["weight"]) * mc["gb"].predict_proba(Xt)
)

metrics["method"] = multi_metrics(ymt, mpt, mc["classes"])
metrics["method"]["champion"] = (
    "balanced" if mc["balanced"] else "unbalanced"
)
metrics["method"]["validation_candidates"] = {
    ("balanced" if x["balanced"] else "unbalanced"): x["validation"]
    for x in method_candidates
}
metrics["method"]["n"] = int(n)

joblib.dump(
    {
        "type": "flat",
        "logit": mc["logit"],
        "gb": mc["gb"],
        "weight": mc["weight"],
        "features": features,
        "medians": medians.to_dict(),
        "classes": mc["classes"].tolist(),
    },
    MODELS / "ufc_method.joblib",
)

# -----------------------------
# Round
# Compare balanced vs unbalanced.
# -----------------------------
yr = d["target_round"].copy()
yrtr, yrv, yrt = yr.iloc[:cut1], yr.iloc[cut1:cut2], yr.iloc[cut2:]

round_candidates = []

for balanced in [False, True]:
    class_weight = "balanced" if balanced else None

    rlog = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=4000,
                C=0.7,
                class_weight=class_weight,
            ),
        ),
    ])

    rgb = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.03,
        max_leaf_nodes=18,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=42,
    )

    rlog.fit(Xtr, yrtr)
    rgb.fit(Xtr, yrtr)

    classes = rlog.named_steps["model"].classes_

    p1 = rlog.predict_proba(Xv)
    p2 = rgb.predict_proba(Xv)

    rw = min(
        weights,
        key=lambda q: log_loss(
            yrv,
            q * p1 + (1 - q) * p2,
            labels=classes,
        ),
    )

    pv = rw * p1 + (1 - rw) * p2
    vm = multi_metrics(yrv, pv, classes)

    round_candidates.append({
        "balanced": balanced,
        "logit": rlog,
        "gb": rgb,
        "weight": float(rw),
        "classes": classes,
        "validation": vm,
    })

round_champion = max(
    round_candidates,
    key=lambda x: (
        x["validation"]["accuracy"],
        x["validation"]["macro_f1"],
    ),
)

rc = round_champion
rpt = (
    rc["weight"] * rc["logit"].predict_proba(Xt)
    + (1 - rc["weight"]) * rc["gb"].predict_proba(Xt)
)

metrics["round"] = multi_metrics(yrt, rpt, rc["classes"])
metrics["round"]["champion"] = (
    "balanced" if rc["balanced"] else "unbalanced"
)
metrics["round"]["validation_candidates"] = {
    ("balanced" if x["balanced"] else "unbalanced"): x["validation"]
    for x in round_candidates
}
metrics["round"]["n"] = int(n)

joblib.dump(
    {
        "type": "flat",
        "logit": rc["logit"],
        "gb": rc["gb"],
        "weight": rc["weight"],
        "features": features,
        "medians": medians.to_dict(),
        "classes": rc["classes"].tolist(),
    },
    MODELS / "ufc_round.joblib",
)

# -----------------------------
# Exact untouched holdout predictions for Historical Replay
# -----------------------------
test_rows = d.iloc[cut2:].copy().reset_index(drop=True)
replay_rows = []

method_classes = list(mc["classes"])
round_classes = list(rc["classes"])

for i, row in test_rows.iterrows():
    pa = float(pt[i])
    pred_winner = row["A"] if pa >= .5 else row["B"]
    actual_winner = row["A"] if int(row["target_win"]) == 1 else row["B"]

    method_probs = {str(c): float(mpt[i, j]) for j, c in enumerate(method_classes)}
    round_probs = {str(c): float(rpt[i, j]) for j, c in enumerate(round_classes)}

    pred_method = max(method_probs, key=method_probs.get)
    pred_round = max(round_probs, key=round_probs.get)

    replay_rows.append({
        "sport": "ufc",
        "event": str(row.get("EVENT","")),
        "bout": str(row.get("BOUT","")),
        "date": None if pd.isna(row.get("DATE")) else pd.Timestamp(row.get("DATE")).strftime("%Y-%m-%d"),
        "fighter_a": str(row["A"]),
        "fighter_b": str(row["B"]),
        "fighter_a_win_prob": pa,
        "fighter_b_win_prob": 1.0-pa,
        "predicted_winner": str(pred_winner),
        "actual_winner": str(actual_winner),
        "winner_correct": bool(pred_winner == actual_winner),
        "method_probs": method_probs,
        "predicted_method": str(pred_method),
        "actual_method": str(row["target_method"]),
        "method_correct": bool(pred_method == row["target_method"]),
        "round_probs": round_probs,
        "predicted_round": str(pred_round),
        "actual_round": str(row["target_round"]),
        "round_correct": bool(pred_round == row["target_round"]),
        "scheduled_5_round": int(row["scheduled_5_round"]),
        "weight_class": str(row["WEIGHTCLASS"]),
    })

(MODELS/"ufc_replay.json").write_text(json.dumps(replay_rows,indent=2))
print(f"Saved {len(replay_rows):,} exact UFC holdout predictions to models/ufc_replay.json")

# -----------------------------
# Latest fighter states for app inference
# -----------------------------
latest = (
    fighter_fight.sort_values("DATE")
    .groupby("FIGHTER")
    .tail(1)
    .copy()
    .join(attr_map, on="FIGHTER")
)

latest["age_days"] = (
    pd.Timestamp.today().normalize() - latest["dob"]
).dt.days

latest_state = {}

for _, row in latest.iterrows():
    fighter = str(row["FIGHTER"])
    latest_state[fighter] = {}

    for c in statecols:
        val = row[c] if c in row.index else np.nan
        latest_state[fighter][c] = (
            None if pd.isna(val) else float(val)
        )

meta = {
    "version": "v8_ufc_fixed",
    "metrics": metrics,
    "fighters": sorted(latest_state.keys()),
    "statecols": statecols,
    "latest_state": latest_state,
    "notes": {
        "scheduled_5_round_source": "TIME FORMAT",
        "leakage_fix": True,
    },
}

(MODELS / "ufc_meta.json").write_text(
    json.dumps(meta, indent=2)
)

print(json.dumps(metrics, indent=2))
