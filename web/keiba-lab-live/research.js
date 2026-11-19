const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=p=>Number.isFinite(p)?(p*100).toFixed(3)+'%':'未取得';
const NAMES={H_PL:'H由来のPL参考',raw_market_PL:'市場由来のPL参考',ABCD_PL:'ABCD由来のPL参考',direct_official_place:'複勝直接モデル',shared_order_probability:'共通着順モデル',H:'H（保存開発版）',raw_market:'無補正市場',H_style:'H_style（不採用・保存参照）',H_pace:'H_pace（不採用・保存参照）'};
const WAGERS={win:'単勝',place:'複勝',wide:'ワイド',umaren:'馬連'};
export function renderMetrics(comparison){
 if(!comparison)return '<p>学習済みモデルの共通集合比較は未取得です。</p>';
 const num=x=>Number.isFinite(x)?x.toFixed(6):'未算出';
 return `<p class="notice">${e(comparison.development_period.join(' ～ '))}。券種ごとに同じレース・同じ対象集合で比較。馬単位／ペア単位の二値指標です。券種間のAUCの大小を優劣として比較しません。収益は未精算です。</p>${Object.entries(comparison.metrics).map(([wager,models])=>`<h3>${e(WAGERS[wager])} · 共通${comparison.coverage[wager].common_official_races}レース</h3><div class="scroll"><table><thead><tr><th>モデル</th><th>二値Log Loss ↓</th><th>Brier ↓</th><th>AUC ↑</th><th>最上位対象の的中率</th></tr></thead><tbody>${Object.entries(models).map(([name,m])=>`<tr><td>${e(NAMES[name]||name)}</td><td>${num(m.binary_log_loss)}</td><td>${num(m.brier_score)}</td><td>${num(m.roc_auc)}</td><td>${pct(m.top_probability_target_hit_rate_percent/100)}</td></tr>`).join('')}</tbody></table></div>`).join('')}<p class="muted">追加探索や自動採用は行いません。元のHは2025年の保存開発モデルの基準であり、Stage 7の途中成績ではありません。</p>`;
}
export function ranked(values){return Object.keys(values).sort((a,b)=>values[b]-values[a]||a.localeCompare(b));}
export function officialHit(outcome,target){return outcome?.target_status==='confirmed'?outcome.official_targets.includes(target):null;}
export function viewRows(race,wager){
 const models=wager==='win'?race.win_models:Object.fromEntries(Object.entries(race.wager_models).filter(([,m])=>m.probabilities[wager]));
 const first=Object.values(models)[0];const values=wager==='win'?first.probabilities:first.probabilities[wager];
 return {models,targets:ranked(values),rows:Object.keys(values).map(target=>({target,
  values:Object.fromEntries(Object.entries(models).map(([name,m])=>{let probabilities=wager==='win'?m.probabilities:m.probabilities[wager],order=ranked(probabilities);return [name,{probability:probabilities[target],rank:order.indexOf(target)+1}];})),
  official_hit:wager==='win'?race.finishes[target]===1:officialHit(race.official[wager],target),
  rank_reference_hit:wager==='win'?race.finishes[target]===1:!!first.rank_reference_labels[wager][target],
  payout:null}))};
}
export function renderResearch(race,wager,{results=false,limit=20}={}){
 if(!race||!WAGERS[wager])return '<p>研究データは未取得です。</p>';
 const {models,targets,rows}=viewRows(race,wager),names=Object.keys(models),count=limit==='all'?targets.length:Number(limit);
 const selected=targets.slice(0,count).map(t=>rows.find(r=>r.target===t));
 const official=race.official[wager],label=wager==='win'?'保存済み単勝確率':'PL参考値と、利用可能な学習済み研究モデルの比較';
 const modelLabel=m=>wager==='win'?'保存済み単勝確率':m.probability_status==='trained_direct_official_place'?'公式複勝対象の直接推定・独立較正':m.probability_status==='trained_order_PL_reference'?'学習済み着順分布・券種別較正は未実施':'未較正・PLモデルの参考確率';
 const targetText=t=>t.split('-').map(h=>`${e(h)}${race.horse_names[h]?' '+e(race.horse_names[h]):''}`).join(' − ');
 const cards=`<div class="picks research-picks">${names.map(n=>{let p=wager==='win'?models[n].probabilities:models[n].probabilities[wager],top=ranked(p)[0];return `<div class="pick"><small>${e(NAMES[n]||n)}</small><strong>${targetText(top)}</strong><span>${pct(p[top])} · ${e(modelLabel(models[n]))}</span></div>`;}).join('')}</div>`;
 const status=wager==='win'?'保存確率・順位を変更せず表示':official?.target_status==='confirmed'?'公式HRの通常的中対象を確認済み · 金額は未精算':'公式的中対象は未確認 · 順位参考のみ';
 return `<div class="race-heading"><h2>${e(race.venue)} ${race.race_no}R · ${e(WAGERS[wager])}</h2><span class="tag">${e(race.date)} · ${race.roster.length}頭 · 同一集合</span></div><p class="notice">${e(label)}。${wager==='place'?`同時点O1の複勝着払キー：${race.paid_positions}着払い。`:''}${e(status)}。</p>${cards}<p class="muted">全${targets.length}対象のうち${selected.length}対象を表示。最初の系列の順位で並べています。</p><div class="scroll"><table id="research-table"><thead><tr><th>${wager==='wide'||wager==='umaren'?'馬番ペア（順不同）':'馬番・馬名'}</th>${names.map(n=>`<th>${e(NAMES[n]||n)}<small>順位 / ${wager==='win'?'勝率':models[n].probability_status==='trained_direct_official_place'?'直接推定確率':'PL参考確率'}</small></th>`).join('')}${results?'<th>順位参考の的中</th><th>公式的中 / 着順</th><th>払戻 / 100円</th>':''}</tr></thead><tbody>${selected.map(row=>`<tr data-target="${e(row.target)}"><td>${targetText(row.target)}</td>${names.map(n=>`<td data-model="${e(n)}" data-rank="${row.values[n].rank}" data-probability="${row.values[n].probability}" class="${row.values[n].rank===1?'top':''}">${row.values[n].rank}位 / <span title="${row.values[n].probability}">${pct(row.values[n].probability)}</span></td>`).join('')}${results?`<td>${row.rank_reference_hit?'参考的中':'参考不的中'}</td><td>${wager==='win'?e(race.finishes[row.target])+'着':row.official_hit===null?'未確認':row.official_hit?'的中':'不的中'}</td><td>未精算</td>`:''}</tr>`).join('')}</tbody></table></div><details class="provenance"><summary>出所・モデル・研究区分</summary><p>開催後の保存開発評価再演 / 発走前の真正な生成時刻：未確認。学習済みモデルは2024年以前の訓練・較正による2025年の再利用開発評価です。PL基準と直接推定を混同しません。</p>${names.map(n=>`<p>${e(NAMES[n]||n)} · モデルID ${e(models[n].model_id)} · T=${e(models[n].temperature??'元の保存値')}<br>${e(modelLabel(models[n]))}<br>元予測SHA-256：${e(models[n].source_sha256)}</p>`).join('')}<p>公式HR：${e(official?.revision_basis||'金額対象外')}。金額を表示オッズや着順から再計算しません。</p></details>`;
}
if(typeof document!=='undefined'){
 const $=q=>document.querySelector(q);let data=null,wager='win',busy=false,lastId=null;
 function render(){if(!data?.races?.length){$('#research-detail').innerHTML='<p>保存研究データは未準備です。現在の発走前予想は上のリンクで閲覧できます。</p>';return;}
  let race=data.races.find(r=>r.race_id===$('#research-race').value)||data.races[0];
  $('#research-detail').innerHTML=renderResearch(race,wager,{results:$('#show-research-results').checked,limit:$('#research-limit').value});
  $('#research-metrics').innerHTML=renderMetrics(data.development_comparison);
 }
 async function refresh(){if(busy)return;busy=true;try{let response=await fetch('/api/research',{cache:'no-store',credentials:'same-origin'}),body=await response.json();if(!response.ok||body.schema_version!=='keiba-lab.multiwager.view.v1')throw Error();data=body;
  if(lastId!==body.artifact_id){let selected=$('#research-race').value;$('#research-race').innerHTML=body.races.map(r=>`<option value="${e(r.race_id)}" ${r.race_id===selected?'selected':''}>${e(r.date)} · ${e(r.venue)} ${r.race_no}R · ${r.roster.length}頭</option>`).join('');lastId=body.artifact_id;}
  $('#coverage').innerHTML=[['研究レース',body.races.length],['出走馬',body.races.reduce((s,r)=>s+r.roster.length,0)],['券種',4],['学習済み追加モデル',body.included_saved_research_fits||0]].map(([name,value])=>`<article><strong>${value}</strong><span>${name}</span></article>`).join('');render();$('#status').textContent='● ローカル保存版を自動確認 · '+new Date().toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo'});
 }catch{$('#status').textContent='研究データの読込みを保留しています。既存の発走前記録器は独立稼働中です。';}finally{busy=false;}}
 document.querySelectorAll('[data-wager]').forEach(b=>b.addEventListener('click',()=>{wager=b.dataset.wager;document.querySelectorAll('[data-wager]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-selected',String(x===b));});render();}));
 ['#research-race','#research-limit','#show-research-results'].forEach(q=>$(q).addEventListener('change',render));setInterval(refresh,15000);refresh();
}
