
from pathlib import Path
import json
import os
import math
import numpy as np
import pandas as pd
import joblib
from google import genai
from google.genai import types
from flask import Flask, render_template, request, jsonify

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/"models"
app=Flask(__name__)

def load_json(name,default):
    p=MODELS/name
    return json.loads(p.read_text()) if p.exists() else default

BBMETA=load_json("basketball_meta.json",{"metrics":{},"spread_metrics":{},"teams":{},"states":{},"elo":{},"home_adv":70})
UFCMETA=load_json("ufc_meta.json",{"metrics":{},"fighters":[],"statecols":[],"latest_state":{}})


def confidence_label(p):
    q=max(float(p),1-float(p))
    if q < .55: return "Toss-up"
    if q < .65: return "Lean"
    if q < .75: return "Strong"
    return "High confidence"


def confidence_calibration(metric, p):
    q=max(float(p),1-float(p))
    for band in metric.get("confidence_calibration",[]) or []:
        lo=float(band.get("min_prob",0))
        hi=float(band.get("max_prob",1))
        if q>=lo and (q<hi or hi>=1.0):
            return band
    return None

def edge_label(edge_abs):
    edge_abs=abs(float(edge_abs))
    if edge_abs < 1.5:
        return "NO EDGE"
    if edge_abs < 3.0:
        return "SMALL EDGE"
    if edge_abs < 5.0:
        return "MODERATE EDGE"
    return "STRONG EDGE"

def basketball_predict_core(league, home, away, neutral=False):
    states=BBMETA.get("states",{}).get(league,{})
    if home not in states or away not in states:
        raise ValueError("Team state unavailable.")

    pack=joblib.load(MODELS/f"basketball_{league}.joblib")
    hs,aws=states[home],states[away]
    helo=float(BBMETA.get("elo",{}).get(league,{}).get(home,1500))
    aelo=float(BBMETA.get("elo",{}).get(league,{}).get(away,1500))
    adv=0 if neutral else float(BBMETA.get("home_adv",70))
    elo_prob=lambda ra,rb:1/(1+10**((rb-ra)/400))

    feats={
        "elo_diff":helo-aelo,"elo_home_prob":elo_prob(helo+adv,aelo),
        "neutral":1 if neutral else 0,"conference_game":0,
        "rank_diff":0,"home_ranked":0,"away_ranked":0,"diff_rest_days":0
    }

    for f in pack["features"]:
        if f.startswith("diff_") and f!="diff_rest_days":
            key=f[5:]
            hv,av=hs.get(key),aws.get(key)
            if hv is not None and av is not None:
                feats[f]=float(hv)-float(av)

    med=pack.get("medians",{})
    row=[]
    for f in pack["features"]:
        v=feats.get(f,med.get(f,0))
        if v is None or (isinstance(v,float) and np.isnan(v)):
            v=med.get(f,0)
        row.append(float(v))
    X=pd.DataFrame([row],columns=pack["features"])

    p1=pack["logit"].predict_proba(X)[:,1]
    p2=pack["gb"].predict_proba(X)[:,1]
    ph=float(pack["weight"]*p1[0]+(1-pack["weight"])*p2[0])

    lookup={x["id"]:x["name"] for x in BBMETA.get("teams",{}).get(league,[])}
    hn,an=lookup.get(home,home),lookup.get(away,away)
    m=BBMETA.get("metrics",{}).get(league,{})

    drivers=[
        {"name":"Elo edge","value":round(feats.get("elo_diff",0),1)},
        {"name":"Last-10 win-rate edge","value":round(feats.get("diff_win_pct_r10",0)*100,1)},
        {"name":"Last-10 net-rating edge","value":round(feats.get("diff_net_rating_r10",0),1)},
        {"name":"Effective FG% edge","value":round(feats.get("diff_efg_pct_r10",0)*100,1)}
    ]

    spread = None
    spread_path = MODELS/f"basketball_spread_{league}.joblib"
    if spread_path.exists():
        sp=joblib.load(spread_path)
        smed=sp.get("medians",{})
        srow=[]
        for f in sp["features"]:
            v=feats.get(f,smed.get(f,0))
            if v is None or (isinstance(v,float) and np.isnan(v)):
                v=smed.get(f,0)
            srow.append(float(v))
        SX=pd.DataFrame([srow],columns=sp["features"])
        predicted_margin=float(
            sp["weight"]*sp["ridge"].predict(SX)[0]
            +(1-sp["weight"])*sp["gb"].predict(SX)[0]
        )
        spread={
            "predicted_margin":predicted_margin,
            "model_home_line":-predicted_margin,
            "residual_sigma":float(sp.get("residual_sigma",12.0)),
            "metrics":BBMETA.get("spread_metrics",{}).get(league,{})
        }

    return {
        "winner":hn if ph>=.5 else an,
        "home":{"id":home,"name":hn,"prob":ph},
        "away":{"id":away,"name":an,"prob":1-ph},
        "upset_risk":min(ph,1-ph),
        "confidence":confidence_label(ph),
        "confidence_calibration":confidence_calibration(m,ph),
        "drivers":drivers,
        "metrics":m,
        "spread":spread,
        "neutral":neutral
    }

def deterministic_explanation(context):
    sport=context.get("sport")
    p=context.get("prediction",{})

    if sport=="basketball":
        winner=p.get("winner","The predicted team")
        hp=(p.get("home") or {}).get("prob",.5)
        ap=(p.get("away") or {}).get("prob",.5)
        winp=max(hp,ap)
        conf=p.get("confidence",confidence_label(winp))
        drivers=p.get("drivers",[])
        useful=[d for d in drivers if abs(float(d.get("value",0)))>0.01]
        useful=sorted(useful,key=lambda x:abs(float(x.get("value",0))),reverse=True)[:3]
        parts=[f"STAATY favors {winner} at {winp*100:.1f}% ({conf})."]
        if useful:
            txt=", ".join(f"{d['name']} {d['value']:+g}" for d in useful)
            parts.append("The largest model signals are "+txt+".")
        risk=(p.get("upset_risk") or min(winp,1-winp))*100
        parts.append(f"The upset side still carries about {risk:.1f}% probability, so this is not a certainty.")
        return " ".join(parts)

    if sport=="ufc":
        winner=p.get("winner","The predicted fighter")
        pa=float(p.get("a_prob",.5)); pb=float(p.get("b_prob",.5))
        winp=max(pa,pb)
        method=p.get("method",{})
        rnd=p.get("round",{})
        m=max(method,key=method.get) if method else None
        r=max(rnd,key=rnd.get) if rnd else None
        parts=[f"STAATY favors {winner} at {winp*100:.1f}% ({confidence_label(winp)})."]
        if m:
            parts.append(f"The most likely method is {m.replace('_','/')} at {method[m]*100:.1f}%.")
        if r:
            parts.append(f"The most likely ending round bucket is {r.replace('_','-')} at {rnd[r]*100:.1f}%.")
        parts.append("These probabilities come from the statistical models; the explanation does not change them.")
        return " ".join(parts)

    return "Run a prediction first so STAATY has model evidence to explain."

def final_holdout_count(n):
    try:
        n=int(n)
        return n-int(n*0.85)
    except Exception:
        return None

@app.route("/")
def home():
    return render_template("index.html",bb=BBMETA,ufc=UFCMETA)


@app.route("/api/ask", methods=["POST"])
def ask_staaty():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    context = payload.get("context", {})

    if not question:
        return jsonify({"error": "Ask a question about the prediction."}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY is not set in the server environment."
        }), 500

    # Keep prediction authority in STAATY's statistical models.
    # Gemini receives only the already-computed output and explains it.
    system_instruction = """
You are STAATY Analyst, the explanation layer for a sports prediction system.

Rules:
1. Never invent, alter, recalculate, or override the supplied prediction probabilities.
2. Treat the supplied STAATY model output as the source of truth for this answer.
3. Clearly distinguish model evidence from interpretation.
4. Never claim certainty or guarantee an outcome.
5. Do not imply correlation is causation.
6. If the supplied context cannot answer the question, say that the available model data is insufficient.
7. When useful, cite the exact probability, model accuracy, baseline, or feature value supplied in the context.
8. Keep answers concise and understandable to a sports fan.
9. Do not provide betting guarantees, "locks," or instructions to wager.
"""

    prompt = (
        "STAATY MODEL CONTEXT:\n"
        + json.dumps(context, indent=2)
        + "\n\nUSER QUESTION:\n"
        + question
    )

    try:
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=500,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "Gemini returned no text for this question."
        return jsonify({"answer": answer, "model": model})
    except Exception as e:
        return jsonify({
            "error": "Gemini request failed.",
            "detail": str(e)
        }), 500


@app.route("/api/replay")
def replay():
    sport=request.args.get("sport","basketball")
    limit=max(1,min(int(request.args.get("limit","30")),100))
    only_correct=request.args.get("correct","all")
    league=request.args.get("league","")

    p=MODELS/f"{sport}_replay.json"
    if not p.exists():
        return jsonify({"error":f"No {sport} replay data yet. Re-run the v12 trainer."}),404

    data=json.loads(p.read_text())

    if sport=="basketball" and league:
        data=[x for x in data if x.get("league")==league]

    if only_correct in ("yes","no"):
        want=(only_correct=="yes")
        key="correct" if sport=="basketball" else "winner_correct"
        data=[x for x in data if bool(x.get(key))==want]

    # Most recent first for a better judge demo.
    data=sorted(data,key=lambda x:(x.get("date") or ""),reverse=True)

    summary={
        "sport":sport,
        "total":len(data),
        "rows":data[:limit]
    }
    return jsonify(summary)

@app.route("/api/proof")
def proof():
    basketball=[]
    for league,m in BBMETA.get("metrics",{}).items():
        basketball.append({
            "league":league,
            "accuracy":m.get("accuracy"),
            "log_loss":m.get("log_loss"),
            "brier":m.get("brier"),
            "roc_auc":m.get("roc_auc"),
            "test_games":m.get("test_games"),
            "home_baseline_accuracy":m.get("home_baseline_accuracy"),
            "elo_accuracy":m.get("elo_accuracy"),
            "champion":m.get("champion"),
            "n_features":m.get("n_features"),
        })

    ufc=[]
    for target,m in UFCMETA.get("metrics",{}).items():
        ufc.append({
            "target":target,
            "accuracy":m.get("accuracy"),
            "macro_f1":m.get("macro_f1"),
            "log_loss":m.get("log_loss"),
            "test_fights":final_holdout_count(m.get("n")),
            "champion":m.get("champion"),
            "n":m.get("n"),
        })

    return jsonify({
        "basketball":basketball,
        "ufc":ufc,
        "protocol":{
            "split":"70% train · 15% validation · 15% final holdout",
            "chronological":True,
            "shifted_history":True,
            "ufc_5round_source":"TIME FORMAT",
            "basketball_selection":"Champion selected on validation performance",
            "ufc_selection":"Candidate selected on validation performance",
            "bounty_threshold":0.50
        }
    })

@app.route("/api/basketball")
def basketball():
    league=request.args.get("league","nba")
    home=request.args.get("home","")
    away=request.args.get("away","")
    neutral=request.args.get("neutral","0")=="1"

    if home==away:
        return jsonify({"error":"Choose two different teams."}),400
    try:
        return jsonify(basketball_predict_core(league,home,away,neutral))
    except Exception as e:
        return jsonify({"error":str(e)}),400

@app.route("/api/upset-radar")
def upset_radar():
    league=request.args.get("league","nba")
    teams=BBMETA.get("teams",{}).get(league,[])
    if len(teams)<2:
        return jsonify({"error":"No teams available."}),400

    # Rank all pairings as hypothetical neutral-ish upcoming matchups using home/away context.
    # Limit to the most current 40 teams in metadata to keep response quick for NCAA.
    sample=teams[:40] if len(teams)>40 else teams
    candidates=[]
    for i,h in enumerate(sample):
        for a in sample[i+1:]:
            try:
                pred=basketball_predict_core(league,h["id"],a["id"],False)
                fav=pred["home"] if pred["home"]["prob"]>=.5 else pred["away"]
                dog=pred["away"] if pred["home"]["prob"]>=.5 else pred["home"]
                candidates.append({
                    "favorite":fav["name"],
                    "underdog":dog["name"],
                    "favorite_prob":max(pred["home"]["prob"],pred["away"]["prob"]),
                    "upset_prob":min(pred["home"]["prob"],pred["away"]["prob"]),
                    "confidence":pred["confidence"]
                })
            except:
                pass

    candidates=sorted(candidates,key=lambda x:x["upset_prob"],reverse=True)[:8]
    return jsonify({"league":league,"matchups":candidates})

@app.route("/api/explain", methods=["POST"])
def explain():
    payload=request.get_json(silent=True) or {}
    context=payload.get("context",{})
    return jsonify({"answer":deterministic_explanation(context),"source":"STAATY deterministic explainer"})


@app.route("/api/cover")
def cover():
    league=request.args.get("league","nba")
    home=request.args.get("home","")
    away=request.args.get("away","")
    neutral=request.args.get("neutral","0")=="1"
    try:
        home_spread=float(request.args.get("home_spread","0"))
    except:
        return jsonify({"error":"Enter a numeric home-team spread, such as -5.5 or +3.0."}),400

    try:
        pred=basketball_predict_core(league,home,away,neutral)
    except Exception as e:
        return jsonify({"error":str(e)}),400

    s=pred.get("spread")
    if not s:
        return jsonify({"error":"Spread model unavailable. Re-run python train_basketball.py."}),404

    margin=float(s["predicted_margin"])
    sigma=max(float(s["residual_sigma"]),1e-6)

    # Home covers when actual home margin + home spread > 0.
    z=(margin+home_spread)/sigma
    home_cover=0.5*(1.0+math.erf(z/math.sqrt(2.0)))

    model_home_line=-margin
    edge_points=home_spread-model_home_line
    edge_side=pred["home"]["name"] if edge_points>0 else pred["away"]["name"]
    edge_abs=abs(edge_points)

    return jsonify({
        "home_team":pred["home"]["name"],
        "away_team":pred["away"]["name"],
        "home_spread":home_spread,
        "predicted_margin":margin,
        "home_cover_prob":home_cover,
        "away_cover_prob":1-home_cover,
        "model_home_line":model_home_line,
        "residual_sigma":sigma,
        "edge_points":edge_abs,
        "edge_side":edge_side,
        "edge_signal":edge_label(edge_abs),
        "edge_signed_home":edge_points
    })

@app.route("/api/ufc")
def ufc():
    a=request.args.get("a","")
    b=request.args.get("b","")
    wc=request.args.get("weight_class","Lightweight")
    rounds=int(request.args.get("scheduled_rounds","3"))

    if a==b or a not in UFCMETA.get("latest_state",{}) or b not in UFCMETA.get("latest_state",{}):
        return jsonify({"error":"Choose two different fighters."}),400

    sa,sb=UFCMETA["latest_state"][a],UFCMETA["latest_state"][b]
    feats={}
    for c in UFCMETA["statecols"]:
        va=0 if sa.get(c) is None else sa.get(c)
        vb=0 if sb.get(c) is None else sb.get(c)
        feats["diff_"+c]=va-vb
        feats["sum_"+c]=va+vb

    wm={"Women's Strawweight":115,"Women's Flyweight":125,"Women's Bantamweight":135,"Flyweight":125,"Bantamweight":135,"Featherweight":145,"Lightweight":155,"Welterweight":170,"Middleweight":185,"Light Heavyweight":205,"Heavyweight":265}
    feats["weight_class_lbs"]=wm.get(wc,155)
    feats["womens_division"]=1 if "Women's" in wc else 0
    feats["scheduled_5_round"]=1 if rounds==5 else 0

    # Current app cannot infer stance because stance is not persisted in v8 metadata.
    # Any stance indicator expected by the model falls back to its train median below.
    def frame(pack):
        med=pack.get("medians",{})
        row=[]
        for f in pack["features"]:
            v=feats.get(f,med.get(f,0))
            if v is None or (isinstance(v,float) and np.isnan(v)):
                v=med.get(f,0)
            row.append(float(v))
        return pd.DataFrame([row],columns=pack["features"])

    wp=joblib.load(MODELS/"ufc_winner.joblib")
    X=frame(wp)
    p=wp["weight"]*wp["logit"].predict_proba(X)[:,1]+(1-wp["weight"])*wp["gb"].predict_proba(X)[:,1]
    pA=float(p[0])

    mp=joblib.load(MODELS/"ufc_method.joblib")
    X=frame(mp)
    pp=mp["weight"]*mp["logit"].predict_proba(X)+(1-mp["weight"])*mp["gb"].predict_proba(X)
    method={c:float(v) for c,v in zip(mp["classes"],pp[0])}

    rp=joblib.load(MODELS/"ufc_round.joblib")
    X=frame(rp)
    pp=rp["weight"]*rp["logit"].predict_proba(X)+(1-rp["weight"])*rp["gb"].predict_proba(X)
    rnd={c:float(v) for c,v in zip(rp["classes"],pp[0])}

    joint=[]
    for m,pm in method.items():
        if m=="DECISION": continue
        for r,pr in rnd.items():
            joint.append({"label":f"{m.replace('_','/')} · Round {r.replace('_','-')}","prob":pm*pr})
    joint=sorted(joint,key=lambda x:x["prob"],reverse=True)[:5]

    return jsonify({
        "winner":a if pA>=.5 else b,
        "a_prob":pA,"b_prob":1-pA,
        "method":method,"round":rnd,"joint":joint,
        "metrics":UFCMETA.get("metrics",{})
    })

if __name__=="__main__":
    app.run(debug=True,port=5000)
