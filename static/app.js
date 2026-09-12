
document.querySelectorAll(".tab").forEach(b=>b.addEventListener("click",()=>{
 document.querySelectorAll(".tab,.panel").forEach(x=>x.classList.remove("active"));
 b.classList.add("active");
 document.getElementById(b.dataset.tab).classList.add("active");
 if(b.dataset.tab==="proof") loadProof();
}));

const pct=x=>x==null?"—":(x*100).toFixed(1)+"%";
const num=x=>x==null?"—":Number(x).toFixed(3);
const prettyLeague=x=>({"nba":"NBA","wnba":"WNBA","ncaa_mbb":"NCAA Men","ncaa_wbb":"NCAA Women"}[x]||x);
function barRows(obj){return Object.entries(obj).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<div class="row"><span>${k.replaceAll("_","/")}</span><span>${pct(v)}</span></div><div class="bar"><i style="width:${v*100}%"></i></div>`).join("")}

function initBasketball(){
 const t=window.BASKETBALL_TEAMS||{},l=document.getElementById("bbLeague");if(!l)return;
 Object.keys(t).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=prettyLeague(k);l.appendChild(o)});
 l.addEventListener("change",fillTeams);fillTeams();
}
function fillTeams(){
 const league=document.getElementById("bbLeague").value;
 const list=(window.BASKETBALL_TEAMS||{})[league]||[];
 const h=document.getElementById("bbHome"),a=document.getElementById("bbAway");
 h.innerHTML="";a.innerHTML="";
 list.forEach(t=>{
   const x=document.createElement("option");x.value=t.id;x.textContent=t.name;h.appendChild(x);
   const y=document.createElement("option");y.value=t.id;y.textContent=t.name;a.appendChild(y);
 });
 if(list.length){
   h.selectedIndex=0;
   document.getElementById("bbHomeSearch").value=list[0].name;
 }
 if(list.length>1){
   a.selectedIndex=1;
   document.getElementById("bbAwaySearch").value=list[1].name;
 }else if(list.length){
   a.selectedIndex=0;
   document.getElementById("bbAwaySearch").value=list[0].name;
 }
 closeAllPickers();
}


function normalizePickText(s){
  return (s||"").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"");
}

function basketballTeamList(){
  const league=document.getElementById("bbLeague")?.value;
  return (window.BASKETBALL_TEAMS||{})[league]||[];
}

function renderPicker(menu, matches, chooseFn){
  menu.innerHTML="";
  if(!matches.length){
    menu.innerHTML='<div class="picker-empty">No match</div>';
    menu.classList.remove("hidden");
    return;
  }
  matches.slice(0,12).forEach(item=>{
    const b=document.createElement("button");
    b.type="button";
    b.className="picker-option";
    b.textContent=item.name;
    b.onclick=()=>chooseFn(item);
    menu.appendChild(b);
  });
  menu.classList.remove("hidden");
}

function filterTeamPicker(side){
  const input=document.getElementById(side==="home"?"bbHomeSearch":"bbAwaySearch");
  const menu=document.getElementById(side==="home"?"bbHomeMatches":"bbAwayMatches");
  const q=normalizePickText(input.value);
  const list=basketballTeamList();
  const matches=list.filter(t=>normalizePickText(t.name).includes(q));
  renderPicker(menu,matches,item=>chooseTeam(side,item));
}

function chooseTeam(side,item){
  const input=document.getElementById(side==="home"?"bbHomeSearch":"bbAwaySearch");
  const select=document.getElementById(side==="home"?"bbHome":"bbAway");
  input.value=item.name;
  select.value=item.id;
  closeAllPickers();
}

function syncTypedTeam(side){
  const input=document.getElementById(side==="home"?"bbHomeSearch":"bbAwaySearch");
  const list=basketballTeamList();
  const q=normalizePickText(input.value).trim();

  let match=list.find(t=>normalizePickText(t.name)===q);
  if(!match){
    const partial=list.filter(t=>normalizePickText(t.name).includes(q));
    if(partial.length===1) match=partial[0];
  }
  if(match){
    chooseTeam(side,match);
    return true;
  }
  return false;
}

function fighterItems(){
  return (window.UFC_FIGHTERS||[]).map(x=>({name:x,id:x}));
}

function filterFighterPicker(side){
  const input=document.getElementById(side==="a"?"fighterASearch":"fighterBSearch");
  const menu=document.getElementById(side==="a"?"fighterAMatches":"fighterBMatches");
  const q=normalizePickText(input.value);
  const matches=fighterItems().filter(t=>normalizePickText(t.name).includes(q));
  renderPicker(menu,matches,item=>chooseFighter(side,item));
}

function chooseFighter(side,item){
  const input=document.getElementById(side==="a"?"fighterASearch":"fighterBSearch");
  const select=document.getElementById(side==="a"?"fighterA":"fighterB");
  input.value=item.name;
  select.value=item.id;
  closeAllPickers();
}

function syncTypedFighter(side){
  const input=document.getElementById(side==="a"?"fighterASearch":"fighterBSearch");
  const list=fighterItems();
  const q=normalizePickText(input.value).trim();

  let match=list.find(t=>normalizePickText(t.name)===q);
  if(!match){
    const partial=list.filter(t=>normalizePickText(t.name).includes(q));
    if(partial.length===1) match=partial[0];
  }
  if(match){
    chooseFighter(side,match);
    return true;
  }
  return false;
}

function closeAllPickers(){
  document.querySelectorAll(".picker-menu").forEach(x=>x.classList.add("hidden"));
}

function initFighterSearch(){
  const list=window.UFC_FIGHTERS||[];
  if(!list.length)return;
  const a=document.getElementById("fighterA");
  const b=document.getElementById("fighterB");
  if(a?.options.length){
    document.getElementById("fighterASearch").value=a.options[a.selectedIndex].text;
  }
  if(b?.options.length){
    document.getElementById("fighterBSearch").value=b.options[b.selectedIndex].text;
  }
}

document.addEventListener("click",e=>{
  if(!e.target.closest(".search-picker")) closeAllPickers();
});


async function predictBasketball(){
 const el=document.getElementById("bbResult");
 if(!syncTypedTeam("home") || !syncTypedTeam("away")){
   el.classList.remove("hidden");
   el.innerHTML='<div class="notice">Choose a valid team from the search results.</div>';
   return;
 }
 const league=bbLeague.value,home=bbHome.value,away=bbAway.value,neutral=bbNeutral.checked?"1":"0";
 const r=await fetch(`/api/basketball?league=${encodeURIComponent(league)}&home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}`);
 const d=await r.json();el.classList.remove("hidden");
 if(!r.ok){el.innerHTML=`<div class="notice">${d.error}</div>`;return}
 window.LAST_BASKETBALL_PREDICTION=d;
 document.getElementById("bbAsk").classList.remove("hidden");
 const wp=d.winner===d.home.name?d.home.prob:d.away.prob,m=d.metrics||{};
 const s=d.spread;
 const cal=d.confidence_calibration;
 const calibrationText=cal&&cal.hit_rate!=null
   ? `Validation hit rate in this band: ${pct(cal.hit_rate)} (${cal.n} games)`
   : "Not enough validation games in this confidence band";
 const marginText=s?(s.predicted_margin>=0?`${d.home.name} by ${s.predicted_margin.toFixed(1)}`:`${d.away.name} by ${Math.abs(s.predicted_margin).toFixed(1)}`):"Spread model unavailable";
 el.innerHTML=`<div class="result-card"><h3>Predicted winner</h3><div class="winner">${d.winner}</div><div class="prob">${pct(wp)}</div><div class="model-pill">${d.confidence}</div><div class="calibration-note">${calibrationText}</div></div>
 <div class="result-card"><h3>Matchup probability</h3><div class="row"><span>${d.home.name}</span><span>${pct(d.home.prob)}</span></div><div class="bar"><i style="width:${d.home.prob*100}%"></i></div><div class="row"><span>${d.away.name}</span><span>${pct(d.away.prob)}</span></div><div class="bar"><i style="width:${d.away.prob*100}%"></i></div></div>
 <div class="result-card"><h3>Projected margin</h3><div class="winner spread-margin">${marginText}</div>${s?`<div class="row"><span>Model line (home)</span><span>${s.model_home_line>0?"+":""}${s.model_home_line.toFixed(1)}</span></div><div class="row"><span>Holdout MAE</span><span>${s.metrics?.mae?s.metrics.mae.toFixed(1)+" pts":"—"}</span></div>`:""}</div>
 <div class="result-card"><h3>Why?</h3>${d.drivers.map(x=>`<div class="row"><span>${x.name}</span><span>${x.value>0?"+":""}${x.value}</span></div>`).join("")}</div>
 <div id="coverCard" class="result-card"><h3>Spread cover</h3><div class="notice small-note">Enter a home spread above to compare the model with a market line.</div></div>
 <div class="result-card"><h3>Model proof</h3><div class="row"><span>Champion</span><span>${m.champion||"—"}</span></div><div class="row"><span>Held-out accuracy</span><span>${pct(m.accuracy)}</span></div><div class="row"><span>Home baseline</span><span>${pct(m.home_baseline_accuracy)}</span></div><div class="row"><span>Elo baseline</span><span>${pct(m.elo_accuracy)}</span></div></div>`;

 const market=document.getElementById("bbMarketSpread").value;
 if(market!=="" && s){
   const cr=await fetch(`/api/cover?league=${encodeURIComponent(league)}&home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}&home_spread=${encodeURIComponent(market)}`);
   const cd=await cr.json();
   const cc=document.getElementById("coverCard");
   if(cr.ok){
     const edgeClass=cd.edge_signal==="STRONG EDGE"?"edge-strong":cd.edge_signal==="MODERATE EDGE"?"edge-moderate":cd.edge_signal==="SMALL EDGE"?"edge-small":"edge-none";
     cc.innerHTML=`<h3>EDGE METER</h3>
       <div class="edge-side">${cd.edge_signal==="NO EDGE"?"No meaningful model edge":cd.edge_side}</div>
       <div class="edge-points">${cd.edge_points.toFixed(1)} pts</div>
       <div class="edge-signal ${edgeClass}">${cd.edge_signal}</div>
       <div class="row"><span>Market home line</span><span>${Number(cd.home_spread)>0?"+":""}${Number(cd.home_spread).toFixed(1)}</span></div>
       <div class="row"><span>STAATY home line</span><span>${cd.model_home_line>0?"+":""}${cd.model_home_line.toFixed(1)}</span></div>
       <div class="row"><span>${cd.home_team} cover estimate</span><span>${pct(cd.home_cover_prob)}</span></div>
       <div class="row"><span>${cd.away_team} cover estimate</span><span>${pct(cd.away_cover_prob)}</span></div>
       <small>Cover estimate uses the validation residual distribution. Spread models have material historical error; treat this as decision support, not certainty.</small>`;
   }else{
     cc.innerHTML=`<div class="notice">${cd.error}</div>`;
   }
 }
}
async function predictUFC(){
 const el=document.getElementById("ufcResult");
 if(!syncTypedFighter("a") || !syncTypedFighter("b")){
   el.classList.remove("hidden");
   el.innerHTML='<div class="notice">Choose a valid fighter from the search results.</div>';
   return;
 }
 const a=fighterA.value,b=fighterB.value,wc=weightClass.value,rounds=scheduledRounds.value;
 const r=await fetch(`/api/ufc?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&weight_class=${encodeURIComponent(wc)}&scheduled_rounds=${rounds}`);
 const d=await r.json();el.classList.remove("hidden");
 if(!r.ok){el.innerHTML=`<div class="notice">${d.error}</div>`;return}
 window.LAST_UFC_PREDICTION=d;
 window.LAST_UFC_PREDICTION.matchup={fighter_a:a,fighter_b:b,weight_class:wc,scheduled_rounds:Number(rounds)};
 document.getElementById("ufcAsk").classList.remove("hidden");
 const wp=d.winner===a?d.a_prob:d.b_prob;
 el.innerHTML=`<div class="result-card"><h3>Predicted winner</h3><div class="winner">${d.winner}</div><div class="prob">${pct(wp)}</div></div>
 <div class="result-card"><h3>Method</h3>${barRows(d.method)}</div>
 <div class="result-card"><h3>Round</h3>${barRows(d.round)}</div>
 <div class="result-card wide"><h3>Top finish scenarios</h3>${d.joint.map(x=>`<div class="row"><span>${x.label}</span><span>${pct(x.prob)}</span></div>`).join("")}</div>`;
}

let proofLoaded=false;
async function loadProof(){
 if(proofLoaded)return;
 const r=await fetch("/api/proof");
 const d=await r.json();

 const bb=document.getElementById("basketballProof");
 bb.innerHTML=d.basketball.map(x=>{
   const uplift=x.home_baseline_accuracy==null?null:x.accuracy-x.home_baseline_accuracy;
   return `<article class="proof-card">
     <div class="proof-head"><span>${prettyLeague(x.league)}</span><strong>${pct(x.accuracy)}</strong></div>
     <div class="threshold-row"><span>Bounty threshold</span><span class="${x.accuracy>.5?"pass":"fail"}">${x.accuracy>.5?"PASS":"FAIL"}</span></div>
     <div class="row"><span>Final test games</span><span>${x.test_games??"—"}</span></div>
     <div class="row"><span>Home baseline</span><span>${pct(x.home_baseline_accuracy)}</span></div>
     <div class="row"><span>Elo baseline</span><span>${pct(x.elo_accuracy)}</span></div>
     <div class="row"><span>Lift vs home</span><span>${uplift==null?"—":(uplift*100>=0?"+":"")+(uplift*100).toFixed(1)+" pts"}</span></div>
     <div class="row"><span>Log loss</span><span>${num(x.log_loss)}</span></div>
     <div class="row"><span>ROC-AUC</span><span>${num(x.roc_auc)}</span></div>
     <div class="model-pill">${x.champion||"model"} · ${x.n_features||"—"} features</div>
   </article>`;
 }).join("");

 const uf=document.getElementById("ufcProof");
 uf.innerHTML=d.ufc.map(x=>`<article class="proof-card">
   <div class="proof-head"><span>UFC ${x.target.toUpperCase()}</span><strong>${pct(x.accuracy)}</strong></div>
   <div class="threshold-row"><span>Bounty threshold</span><span class="${x.accuracy>.5?"pass":"fail"}">${x.accuracy>.5?"PASS":"FAIL"}</span></div>
   <div class="row"><span>Final test fights</span><span>${x.test_fights??"—"}</span></div>
   <div class="row"><span>Log loss</span><span>${num(x.log_loss)}</span></div>
   <div class="row"><span>Macro-F1</span><span>${x.macro_f1==null?"—":num(x.macro_f1)}</span></div>
   <div class="row"><span>Margin over 50%</span><span>${((x.accuracy-.5)*100>=0?"+":"")+((x.accuracy-.5)*100).toFixed(1)} pts</span></div>
   <div class="model-pill">${x.champion||"ensemble"} champion</div>
 </article>`).join("");

 proofLoaded=true;
}

document.addEventListener("DOMContentLoaded",()=>{initBasketball();initFighterSearch();});


async function askGemini(sport){
  const isBB=sport==="basketball";
  const input=document.getElementById(isBB?"bbQuestion":"ufcQuestion");
  const answer=document.getElementById(isBB?"bbAnswer":"ufcAnswer");
  const context=isBB?window.LAST_BASKETBALL_PREDICTION:window.LAST_UFC_PREDICTION;

  if(!context){
    answer.classList.remove("hidden");
    answer.textContent="Run a prediction first.";
    return;
  }

  const question=input.value.trim();
  answer.classList.remove("hidden");
  answer.innerHTML='<span class="thinking">STAATY is reading the model evidence…</span>';

  try{
    const r=await fetch("/api/explain",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question,context:{sport,prediction:context}})
    });
    const d=await r.json();
    answer.textContent=d.answer || d.error || "No explanation available.";
  }catch(e){
    answer.textContent="Could not generate explanation: "+e.message;
  }
}

function quickAsk(sport,question){
  const input=document.getElementById(sport==="basketball"?"bbQuestion":"ufcQuestion");
  input.value=question;
  askGemini(sport);
}

document.addEventListener("keydown",e=>{
  if(e.key!=="Enter")return;
  if(e.target.id==="bbQuestion")askGemini("basketball");
  if(e.target.id==="ufcQuestion")askGemini("ufc");
});


async function loadRadar(){
  const league=document.getElementById("radarLeague").value;
  const el=document.getElementById("radarResults");
  el.innerHTML='<div class="notice">Scanning matchup space…</div>';
  const r=await fetch(`/api/upset-radar?league=${encodeURIComponent(league)}`);
  const d=await r.json();
  if(!r.ok){el.innerHTML=`<div class="notice">${d.error}</div>`;return}
  el.innerHTML=d.matchups.map((x,i)=>`
    <article class="radar-card">
      <div class="rank">#${i+1}</div>
      <div class="radar-title">${x.underdog} vs ${x.favorite}</div>
      <div class="radar-prob">${pct(x.upset_prob)} upset chance</div>
      <div class="row"><span>Favorite</span><span>${x.favorite}</span></div>
      <div class="row"><span>Favorite win</span><span>${pct(x.favorite_prob)}</span></div>
      <div class="model-pill">${x.confidence}</div>
    </article>`).join("");
}




function toggleReplayLeague(){
  const sport=document.getElementById("replaySport").value;
  document.getElementById("replayLeagueWrap").style.display=sport==="basketball"?"flex":"none";
}

function replayProb(x){return (Number(x)*100).toFixed(1)+"%";}

async function loadExactReplay(){
  const sport=document.getElementById("replaySport").value;
  const correct=document.getElementById("replayCorrect").value;
  const league=document.getElementById("replayLeague").value;
  const el=document.getElementById("replayResults");
  const summary=document.getElementById("replaySummary");

  el.innerHTML='<div class="notice">Loading exact held-out predictions…</div>';
  const q=new URLSearchParams({sport,correct,limit:"30"});
  if(sport==="basketball" && league) q.set("league",league);

  const r=await fetch("/api/replay?"+q.toString());
  const d=await r.json();

  if(!r.ok){
    el.innerHTML=`<div class="notice">${d.error}</div>`;
    return;
  }

  summary.classList.remove("hidden");
  summary.textContent=`Showing ${d.rows.length} of ${d.total} matching untouched holdout predictions.`;

  if(!d.rows.length){
    el.innerHTML='<div class="notice">No replay rows match these filters.</div>';
    return;
  }

  if(sport==="basketball"){
    el.innerHTML=d.rows.map(x=>{
      const cls=x.correct?"correct-card":"miss-card";
      return `<article class="replay-event ${cls}">
        <div class="replay-top">
          <div><span class="replay-date">${x.date||""}</span><strong>${prettyLeague(x.league)}</strong></div>
          <span class="verdict">${x.correct?"✓ CORRECT":"✕ MISS"}</span>
        </div>
        <h3>${x.away_team} @ ${x.home_team}</h3>
        <div class="replay-prediction">
          <div><small>STAATY predicted</small><b>${x.predicted_winner}</b><span>${replayProb(x.predicted_win_prob)}</span></div>
          <div><small>Actual winner</small><b>${x.actual_winner}</b><span>${x.away_score} – ${x.home_score}</span></div>
        </div>
        <div class="replay-foot">Model: ${x.champion} · Game ID ${x.game_id}</div>
      </article>`;
    }).join("");
  } else {
    el.innerHTML=d.rows.map(x=>{
      const cls=x.winner_correct?"correct-card":"miss-card";
      const methodProb=x.method_probs?.[x.predicted_method];
      const roundProb=x.round_probs?.[x.predicted_round];
      return `<article class="replay-event ${cls}">
        <div class="replay-top">
          <div><span class="replay-date">${x.date||""}</span><strong>${x.event}</strong></div>
          <span class="verdict">${x.winner_correct?"✓ WINNER RIGHT":"✕ WINNER MISS"}</span>
        </div>
        <h3>${x.fighter_a} vs ${x.fighter_b}</h3>
        <div class="replay-prediction three">
          <div><small>Winner prediction</small><b>${x.predicted_winner}</b><span>${replayProb(Math.max(x.fighter_a_win_prob,x.fighter_b_win_prob))}</span><em>Actual: ${x.actual_winner}</em></div>
          <div><small>Method prediction</small><b>${x.predicted_method.replaceAll("_","/")}</b><span>${methodProb==null?"—":replayProb(methodProb)}</span><em>Actual: ${x.actual_method.replaceAll("_","/")} ${x.method_correct?"✓":"✕"}</em></div>
          <div><small>Round prediction</small><b>Round ${x.predicted_round.replaceAll("_","-")}</b><span>${roundProb==null?"—":replayProb(roundProb)}</span><em>Actual: ${x.actual_round.replaceAll("_","-")} ${x.round_correct?"✓":"✕"}</em></div>
        </div>
        <div class="replay-foot">${x.weight_class} · ${x.scheduled_5_round?"5-round scheduled fight":"3-round scheduled fight"}</div>
      </article>`;
    }).join("");
  }
}

document.addEventListener("DOMContentLoaded",toggleReplayLeague);
