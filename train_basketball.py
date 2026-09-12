
from pathlib import Path
import json, math, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

ROLLS = [5, 10]
BASE_ELO = 1500.0
K = 20.0
HOME_ADV = 70.0

RAW_STATS = [
    "team_score","field_goal_pct","three_point_field_goal_pct","free_throw_pct",
    "total_rebounds","offensive_rebounds","defensive_rebounds","assists","steals",
    "blocks","total_turnovers","fast_break_points","points_in_paint","fouls","largest_lead"
]
ADV_STATS = [
    "efg_pct","tov_pct","orb_pct","ft_rate","possessions",
    "off_rating","def_rating","net_rating"
]

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)

def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def mov_multiplier(margin, elo_diff):
    margin = max(abs(float(margin)), 1.0)
    return math.log(margin + 1.0) * (2.2 / (0.001 * abs(float(elo_diff)) + 2.2))

def build_elo(schedule):
    ratings, rows = {}, []
    current_season = None
    s = schedule.sort_values(["game_date","game_id"]).copy()

    for r in s.itertuples(index=False):
        season = str(r.season)
        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = {t: BASE_ELO + 0.75*(v-BASE_ELO) for t,v in ratings.items()}
            current_season = season

        h, a = str(r.home_team_id), str(r.away_team_id)
        rh, ra = ratings.get(h,BASE_ELO), ratings.get(a,BASE_ELO)
        neutral = bool(r.neutral_site) if pd.notna(r.neutral_site) else False
        adv = 0 if neutral else HOME_ADV
        ph = expected(rh+adv, ra)
        rows.append((str(r.game_id), rh, ra, rh-ra, ph))

        if pd.notna(r.home_winner):
            y = int(bool(r.home_winner))
            margin = (float(r.home_score)-float(r.away_score)) if pd.notna(r.home_score) and pd.notna(r.away_score) else 1
            delta = K * mov_multiplier(margin, (rh+adv)-ra) * (y-ph)
            ratings[h], ratings[a] = rh+delta, ra-delta

    return pd.DataFrame(rows, columns=["game_id","home_elo","away_elo","elo_diff","elo_home_prob"])

def final_elo_ratings(schedule):
    ratings = {}
    current_season = None
    s = schedule.sort_values(["game_date","game_id"]).copy()
    for r in s.itertuples(index=False):
        season = str(r.season)
        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = {t: BASE_ELO + 0.75*(v-BASE_ELO) for t,v in ratings.items()}
            current_season = season

        h, a = str(r.home_team_id), str(r.away_team_id)
        rh, ra = ratings.get(h,BASE_ELO), ratings.get(a,BASE_ELO)
        neutral = bool(r.neutral_site) if pd.notna(r.neutral_site) else False
        adv = 0 if neutral else HOME_ADV
        ph = expected(rh+adv, ra)

        if pd.notna(r.home_winner):
            y = int(bool(r.home_winner))
            margin = (float(r.home_score)-float(r.away_score)) if pd.notna(r.home_score) and pd.notna(r.away_score) else 1
            delta = K * mov_multiplier(margin, (rh+adv)-ra) * (y-ph)
            ratings[h], ratings[a] = rh+delta, ra-delta
    return ratings

def enrich_team_box(schedule, team_box):
    tb = team_box.copy()
    tb["game_id"] = tb["game_id"].astype(str)
    tb["team_id"] = tb["team_id"].astype(str)
    tb["opponent_team_id"] = tb["opponent_team_id"].astype(str)

    ssmall = schedule[["game_id","game_date","season","league"]].copy()
    ssmall["game_id"] = ssmall["game_id"].astype(str)
    ssmall["game_date"] = pd.to_datetime(ssmall["game_date"], errors="coerce")
    tb = tb.drop(columns=["league"], errors="ignore").merge(ssmall, on="game_id", how="left")

    numeric_cols = [
        "team_score","field_goals_made","field_goals_attempted",
        "three_point_field_goals_made","three_point_field_goals_attempted",
        "free_throws_made","free_throws_attempted","total_rebounds",
        "offensive_rebounds","defensive_rebounds","assists","steals","blocks",
        "total_turnovers","fast_break_points","points_in_paint","fouls",
        "largest_lead","field_goal_pct","three_point_field_goal_pct",
        "free_throw_pct","opponent_team_score"
    ]
    for c in numeric_cols:
        if c in tb.columns:
            tb[c] = pd.to_numeric(tb[c], errors="coerce")

    opp_cols = [
        "game_id","team_id","field_goals_attempted","free_throws_attempted",
        "offensive_rebounds","defensive_rebounds","total_turnovers"
    ]
    opp = tb[opp_cols].copy().rename(columns={
        "team_id":"opponent_team_id","field_goals_attempted":"opp_fga",
        "free_throws_attempted":"opp_fta","offensive_rebounds":"opp_orb",
        "defensive_rebounds":"opp_drb","total_turnovers":"opp_tov"
    })
    tb = tb.merge(opp, on=["game_id","opponent_team_id"], how="left")

    tb["efg_pct"] = safe_div(tb["field_goals_made"] + .5*tb["three_point_field_goals_made"], tb["field_goals_attempted"])
    tb["possessions"] = tb["field_goals_attempted"] - tb["offensive_rebounds"] + tb["total_turnovers"] + .44*tb["free_throws_attempted"]
    tb["opp_possessions"] = tb["opp_fga"] - tb["opp_orb"] + tb["opp_tov"] + .44*tb["opp_fta"]
    tb["tov_pct"] = safe_div(tb["total_turnovers"], tb["possessions"])
    tb["orb_pct"] = safe_div(tb["offensive_rebounds"], tb["offensive_rebounds"] + tb["opp_drb"])
    tb["ft_rate"] = safe_div(tb["free_throws_attempted"], tb["field_goals_attempted"])
    tb["off_rating"] = 100*safe_div(tb["team_score"], tb["possessions"])
    tb["def_rating"] = 100*safe_div(tb["opponent_team_score"], tb["opp_possessions"])
    tb["net_rating"] = tb["off_rating"] - tb["def_rating"]
    tb["win_num"] = tb["team_winner"].astype(str).str.lower().map({"true":1,"false":0})
    tb["margin"] = tb["team_score"] - tb["opponent_team_score"]

    return tb.sort_values(["league","team_id","game_date","game_id"])

def rolling_team_features(schedule, team_box):
    tb = enrich_team_box(schedule, team_box)
    generated = []

    for c in RAW_STATS + ADV_STATS:
        tb[c] = pd.to_numeric(tb[c], errors="coerce")

    for r in ROLLS:
        for c in RAW_STATS + ADV_STATS:
            name = f"{c}_r{r}"
            tb[name] = tb.groupby(["league","team_id"])[c].transform(
                lambda x: x.shift(1).rolling(r, min_periods=max(3,r//2)).mean()
            )
            generated.append(name)

        for c in ["win_num","margin"]:
            name = f"{'win_pct' if c=='win_num' else 'margin'}_r{r}"
            tb[name] = tb.groupby(["league","team_id"])[c].transform(
                lambda x: x.shift(1).rolling(r, min_periods=max(3,r//2)).mean()
            )
            generated.append(name)

    for c in ["net_rating","margin","win_num","efg_pct","tov_pct","orb_pct"]:
        name = f"{c}_ewm"
        tb[name] = tb.groupby(["league","team_id"])[c].transform(
            lambda x: x.shift(1).ewm(span=8, adjust=False, min_periods=3).mean()
        )
        generated.append(name)

    tb["prev_date"] = tb.groupby(["league","team_id"])["game_date"].shift(1)
    tb["rest_days"] = (pd.to_datetime(tb["game_date"]) - pd.to_datetime(tb["prev_date"])).dt.days.clip(0,14)
    generated.append("rest_days")

    return tb[["game_id","team_id","team_home_away"] + generated], generated

def make_games(schedule, team_box):
    s = schedule.copy()
    for c in ["game_id","home_team_id","away_team_id"]:
        s[c] = s[c].astype(str)
    s["game_date"] = pd.to_datetime(s["game_date"], errors="coerce")

    s = s[
        s["status_detail"].astype(str).str.contains("Final", case=False, na=False)
        & s["home_score"].notna() & s["away_score"].notna()
    ].copy()

    s["target"] = s["home_winner"].astype(str).str.lower().map({"true":1,"false":0})
    s["target_margin"] = pd.to_numeric(s["home_score"],errors="coerce") - pd.to_numeric(s["away_score"],errors="coerce")
    s = s[s["target"].notna()].copy()

    elo = build_elo(s)
    rf, generated = rolling_team_features(s, team_box)

    home = rf[rf["team_home_away"].astype(str).str.lower()=="home"].drop(columns=["team_home_away"]).add_prefix("home_")
    away = rf[rf["team_home_away"].astype(str).str.lower()=="away"].drop(columns=["team_home_away"]).add_prefix("away_")
    home = home.rename(columns={"home_game_id":"game_id"})
    away = away.rename(columns={"away_game_id":"game_id"})

    g = s.merge(elo,on="game_id",how="left").merge(home,on="game_id",how="left").merge(away,on="game_id",how="left")

    advanced = ["elo_diff","elo_home_prob"]
    simple = ["elo_diff","elo_home_prob"]

    for c in generated:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in g.columns and ac in g.columns:
            f = f"diff_{c}"
            g[f] = pd.to_numeric(g[hc],errors="coerce") - pd.to_numeric(g[ac],errors="coerce")
            advanced.append(f)
            if any(k in c for k in ["win_pct_r10","margin_r10","field_goal_pct_r10","three_point_field_goal_pct_r10","free_throw_pct_r10","total_rebounds_r10","assists_r10","total_turnovers_r10","rest_days"]):
                simple.append(f)

    g["rank_diff"] = pd.to_numeric(g["away_curated_rank"],errors="coerce") - pd.to_numeric(g["home_curated_rank"],errors="coerce")
    g["home_ranked"] = pd.to_numeric(g["home_curated_rank"],errors="coerce").notna().astype(int)
    g["away_ranked"] = pd.to_numeric(g["away_curated_rank"],errors="coerce").notna().astype(int)
    g["rank_diff"] = g["rank_diff"].fillna(0)
    g["neutral"] = g["neutral_site"].astype(str).str.lower().map({"true":1,"false":0}).fillna(0)
    g["conference_game"] = g["conference_competition"].astype(str).str.lower().map({"true":1,"false":0}).fillna(0)

    advanced += ["rank_diff","home_ranked","away_ranked","neutral","conference_game"]
    simple += ["neutral"]

    return g, list(dict.fromkeys(simple)), list(dict.fromkeys(advanced))

def score(y, p):
    pred = (p>=.5).astype(int)
    out = {
        "accuracy":float(accuracy_score(y,pred)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "brier":float(brier_score_loss(y,p))
    }
    try: out["roc_auc"] = float(roc_auc_score(y,p))
    except: out["roc_auc"] = None
    return out

def fit_candidate(Xtr, ytr, Xv, yv, mode):
    if mode == "simple":
        logit = Pipeline([("scale",StandardScaler()),("m",LogisticRegression(max_iter=2500,C=.7))])
        gb = HistGradientBoostingClassifier(max_iter=180,learning_rate=.05,max_leaf_nodes=10,min_samples_leaf=35,l2_regularization=2.0,random_state=42)
    else:
        logit = Pipeline([("scale",StandardScaler()),("m",LogisticRegression(max_iter=3000,C=.5))])
        gb = HistGradientBoostingClassifier(max_iter=280,learning_rate=.035,max_leaf_nodes=16,min_samples_leaf=20,l2_regularization=2.5,random_state=42)

    logit.fit(Xtr,ytr)
    gb.fit(Xtr,ytr)
    p1, p2 = logit.predict_proba(Xv)[:,1], gb.predict_proba(Xv)[:,1]

    weights = np.linspace(0,1,41)
    bestw = min(weights, key=lambda w: log_loss(yv, w*p1+(1-w)*p2, labels=[0,1]))
    p = bestw*p1 + (1-bestw)*p2
    return {"logit":logit,"gb":gb,"weight":float(bestw),"val":score(yv,p)}



def calibration_bands(y, p):
    """
    Build confidence bands from VALIDATION predictions only.
    This preserves the final test set as untouched evaluation.
    """
    y=np.asarray(y).astype(int)
    p=np.asarray(p,dtype=float)
    favored_prob=np.maximum(p,1-p)
    pred=(p>=.5).astype(int)
    correct=(pred==y).astype(int)

    specs=[
        ("Toss-up",.50,.55),
        ("Lean",.55,.65),
        ("Strong",.65,.75),
        ("High confidence",.75,1.000001),
    ]

    out=[]
    for label,lo,hi in specs:
        mask=(favored_prob>=lo)&(favored_prob<hi)
        n=int(mask.sum())
        out.append({
            "label":label,
            "min_prob":float(lo),
            "max_prob":float(min(hi,1.0)),
            "n":n,
            "hit_rate":None if n==0 else float(correct[mask].mean()),
            "avg_confidence":None if n==0 else float(favored_prob[mask].mean()),
        })
    return out

def regression_metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
    }

def fit_spread_candidate(Xtr, ytr, Xv, yv, mode):
    # Interpretable linear baseline + nonlinear tree model.
    alpha = 12.0 if mode == "simple" else 20.0
    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha))
    ])

    if mode == "simple":
        gb = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=.04, max_leaf_nodes=10,
            min_samples_leaf=35, l2_regularization=3.0, random_state=42
        )
    else:
        gb = HistGradientBoostingRegressor(
            max_iter=320, learning_rate=.03, max_leaf_nodes=16,
            min_samples_leaf=25, l2_regularization=4.0, random_state=42
        )

    ridge.fit(Xtr, ytr)
    gb.fit(Xtr, ytr)
    p1 = ridge.predict(Xv)
    p2 = gb.predict(Xv)

    weights = np.linspace(0,1,41)
    bestw = min(
        weights,
        key=lambda w: mean_absolute_error(yv, w*p1 + (1-w)*p2)
    )
    pred = bestw*p1 + (1-bestw)*p2

    return {
        "ridge": ridge,
        "gb": gb,
        "weight": float(bestw),
        "validation": regression_metrics(yv, pred),
        "validation_pred": pred,
    }

def current_team_states(schedule, team_box):
    tb = enrich_team_box(schedule, team_box)
    states = {}

    for (league,team_id), g in tb.groupby(["league","team_id"]):
        g = g.sort_values("game_date").copy()
        state = {}
        for r in ROLLS:
            tail = g.tail(r)
            for c in RAW_STATS + ADV_STATS:
                val = pd.to_numeric(tail[c],errors="coerce").mean()
                state[f"{c}_r{r}"] = None if pd.isna(val) else float(val)
            for c,name in [("win_num","win_pct"),("margin","margin")]:
                val = pd.to_numeric(tail[c],errors="coerce").mean()
                state[f"{name}_r{r}"] = None if pd.isna(val) else float(val)

        for c in ["net_rating","margin","win_num","efg_pct","tov_pct","orb_pct"]:
            e = pd.to_numeric(g[c],errors="coerce").ewm(span=8,adjust=False,min_periods=3).mean()
            val = e.iloc[-1] if len(e) else np.nan
            state[f"{c}_ewm"] = None if pd.isna(val) else float(val)

        state["rest_days"] = 0.0
        states.setdefault(str(league),{})[str(team_id)] = state

    return states

def main():
    schedule = pd.read_csv(DATA/"schedule.csv",low_memory=False)
    team_box = pd.read_csv(DATA/"team_box.csv",low_memory=False)
    teams = pd.read_csv(DATA/"teams.csv",low_memory=False)

    games, simple_features, advanced_features = make_games(schedule,team_box)
    all_metrics = {}
    spread_metrics = {}
    replay_rows = []

    for league, gl in games.groupby("league"):
        gl = gl.sort_values("game_date").copy()
        y = gl["target"].astype(int)
        n = len(gl)
        c1, c2 = int(n*.70), int(n*.85)

        candidate_results = {}
        candidate_models = {}

        for mode, features in [("simple",simple_features),("advanced",advanced_features)]:
            X = gl[features].replace([np.inf,-np.inf],np.nan)
            Xtr_raw, Xv_raw, Xt_raw = X.iloc[:c1], X.iloc[c1:c2], X.iloc[c2:]
            ytr, yv, yt = y.iloc[:c1], y.iloc[c1:c2], y.iloc[c2:]

            med = Xtr_raw.median(numeric_only=True).fillna(0)
            Xtr = Xtr_raw.fillna(med).fillna(0)
            Xv = Xv_raw.fillna(med).fillna(0)
            Xt = Xt_raw.fillna(med).fillna(0)

            pack = fit_candidate(Xtr,ytr,Xv,yv,mode)
            pack["features"] = features
            pack["medians"] = med.to_dict()

            ptest = pack["weight"]*pack["logit"].predict_proba(Xt)[:,1] + (1-pack["weight"])*pack["gb"].predict_proba(Xt)[:,1]
            candidate_results[mode] = {
                "validation":pack["val"],
                "test_preview":score(yt,ptest)
            }
            candidate_models[mode] = (pack, Xt, yt)

        # Champion selection uses VALIDATION LOG LOSS only.
        champion = min(candidate_results.keys(), key=lambda m:candidate_results[m]["validation"]["log_loss"])
        pack, Xt, yt = candidate_models[champion]
        ptest = pack["weight"]*pack["logit"].predict_proba(Xt)[:,1] + (1-pack["weight"])*pack["gb"].predict_proba(Xt)[:,1]
        final = score(yt,ptest)

        # Confidence calibration is estimated from VALIDATION only,
        # never from the untouched final holdout.
        Xv_cal = gl.iloc[c1:c2][pack["features"]].replace([np.inf,-np.inf],np.nan)
        cal_med = pd.Series(pack["medians"])
        Xv_cal = Xv_cal.fillna(cal_med).fillna(0)
        pval = (
            pack["weight"]*pack["logit"].predict_proba(Xv_cal)[:,1]
            +(1-pack["weight"])*pack["gb"].predict_proba(Xv_cal)[:,1]
        )
        confidence_calibration = calibration_bands(
            gl.iloc[c1:c2]["target"].astype(int),
            pval
        )

        # Save every untouched final-holdout prediction for exact historical replay.
        test_gl = gl.iloc[c2:].copy()
        for idx, (_, row) in enumerate(test_gl.iterrows()):
            ph = float(ptest[idx])
            actual_home_win = int(row["target"])
            replay_rows.append({
                "sport": "basketball",
                "league": str(league),
                "game_id": str(row.get("game_id","")),
                "date": None if pd.isna(row.get("game_date")) else pd.Timestamp(row.get("game_date")).strftime("%Y-%m-%d"),
                "home_team_id": str(row.get("home_team_id","")),
                "away_team_id": str(row.get("away_team_id","")),
                "home_score": None if pd.isna(row.get("home_score")) else float(row.get("home_score")),
                "away_score": None if pd.isna(row.get("away_score")) else float(row.get("away_score")),
                "home_win_prob": ph,
                "away_win_prob": 1.0-ph,
                "predicted_home_win": int(ph >= 0.5),
                "actual_home_win": actual_home_win,
                "correct": bool((ph >= 0.5) == bool(actual_home_win)),
                "champion": champion,
                "neutral": bool(row.get("neutral",0)),
            })

        # Baselines on final holdout.
        home_acc = float(test_gl["target"].mean())
        elo_acc = float(((test_gl["elo_home_prob"].fillna(.5)>=.5).astype(int) == test_gl["target"].astype(int)).mean())

        final.update({
            "champion":champion,
            "home_baseline_accuracy":home_acc,
            "elo_accuracy":elo_acc,
            "validation_log_loss":candidate_results[champion]["validation"]["log_loss"],
            "n_games":int(n),
            "test_games":int(len(yt)),
            "n_features":len(pack["features"]),
            "logit_weight":pack["weight"],
            "gb_weight":1-pack["weight"],
            "candidate_validation":{k:v["validation"] for k,v in candidate_results.items()},
            "confidence_calibration":confidence_calibration
        })
        all_metrics[str(league)] = final

        joblib.dump(pack, MODELS/f"basketball_{league}.joblib")

        # -------------------------------------------------
        # BONUS MODEL: predicted scoring margin / spread
        # -------------------------------------------------
        spread_candidates = {}
        spread_models = {}

        ymargin = pd.to_numeric(gl["target_margin"], errors="coerce")

        for mode, features in [("simple",simple_features),("advanced",advanced_features)]:
            Xs = gl[features].replace([np.inf,-np.inf],np.nan)

            Xtr_raw = Xs.iloc[:c1].copy()
            Xv_raw = Xs.iloc[c1:c2].copy()
            Xt_raw = Xs.iloc[c2:].copy()

            ytr_m = ymargin.iloc[:c1]
            yv_m = ymargin.iloc[c1:c2]
            yt_m = ymargin.iloc[c2:]

            smed = Xtr_raw.median(numeric_only=True).fillna(0)
            Xtr_s = Xtr_raw.fillna(smed).fillna(0)
            Xv_s = Xv_raw.fillna(smed).fillna(0)
            Xt_s = Xt_raw.fillna(smed).fillna(0)

            spack = fit_spread_candidate(Xtr_s, ytr_m, Xv_s, yv_m, mode)
            spack["features"] = features
            spack["medians"] = smed.to_dict()

            ptest_margin = (
                spack["weight"] * spack["ridge"].predict(Xt_s)
                + (1-spack["weight"]) * spack["gb"].predict(Xt_s)
            )

            # Residual sigma comes from validation only, never the final test.
            val_resid = yv_m.to_numpy() - spack["validation_pred"]
            sigma = float(np.std(val_resid, ddof=1))
            spack["residual_sigma"] = sigma

            spread_candidates[mode] = spack["validation"]
            spread_models[mode] = (spack, ptest_margin, yt_m)

        spread_champion = min(
            spread_candidates.keys(),
            key=lambda m: spread_candidates[m]["mae"]
        )
        spack, ptest_margin, yt_m = spread_models[spread_champion]
        sm = regression_metrics(yt_m, ptest_margin)
        sm.update({
            "champion": spread_champion,
            "validation_mae": spread_candidates[spread_champion]["mae"],
            "test_games": int(len(yt_m)),
            "residual_sigma": float(spack["residual_sigma"]),
            "n_features": int(len(spack["features"])),
            "ridge_weight": float(spack["weight"]),
            "gb_weight": float(1-spack["weight"]),
        })
        spread_metrics[str(league)] = sm

        joblib.dump(
            spack,
            MODELS/f"basketball_spread_{league}.joblib"
        )

        # Attach exact spread predictions to the already-created holdout replay rows.
        league_replays = [r for r in replay_rows if r["league"] == str(league)]
        if len(league_replays) == len(ptest_margin):
            for rr, margin_pred in zip(league_replays, ptest_margin):
                rr["predicted_margin"] = float(margin_pred)
                rr["actual_margin"] = float(rr["home_score"] - rr["away_score"])
                rr["margin_error"] = float(abs(rr["actual_margin"] - rr["predicted_margin"]))

    team_names = {str(r.team_id):r.display_name for r in teams.itertuples(index=False)}
    schedule["game_date"] = pd.to_datetime(schedule["game_date"],errors="coerce")
    states = current_team_states(schedule,team_box)

    team_lists, elo_by_league = {}, {}
    for league in schedule["league"].dropna().unique():
        ls = schedule[schedule["league"]==league].copy()
        tids = set(ls["home_team_id"].astype(str)) | set(ls["away_team_id"].astype(str))
        team_lists[str(league)] = sorted([{"id":t,"name":team_names.get(t,t)} for t in tids], key=lambda x:x["name"])
        elo_by_league[str(league)] = {str(k):float(v) for k,v in final_elo_ratings(ls).items()}

    meta = {
        "version":"v7_champion",
        "metrics":all_metrics,
        "spread_metrics":spread_metrics,
        "teams":team_lists,
        "states":states,
        "elo":elo_by_league,
        "home_adv":HOME_ADV
    }
    (MODELS/"basketball_meta.json").write_text(json.dumps(meta,indent=2))

    # Add human-readable team names after training.
    for r in replay_rows:
        r["home_team"] = team_names.get(r["home_team_id"], r["home_team_id"])
        r["away_team"] = team_names.get(r["away_team_id"], r["away_team_id"])
        r["predicted_winner"] = r["home_team"] if r["predicted_home_win"] else r["away_team"]
        r["actual_winner"] = r["home_team"] if r["actual_home_win"] else r["away_team"]
        r["predicted_win_prob"] = r["home_win_prob"] if r["predicted_home_win"] else r["away_win_prob"]

    (MODELS/"basketball_replay.json").write_text(json.dumps(replay_rows,indent=2))
    print(f"Saved {len(replay_rows):,} exact basketball holdout predictions to models/basketball_replay.json")
    print(json.dumps({"winner_models":all_metrics,"spread_models":spread_metrics},indent=2))

if __name__=="__main__":
    main()
