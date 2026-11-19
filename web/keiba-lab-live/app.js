export const D39=['old_D39','new_D39','control_D39'];
export const FOUR=[...D39,'H'];
export const labels={old_D39:'旧D39',new_D39:'新D39',control_D39:'D39対照型',H:'主力H'};
export const resolveDay=(choice,instant=new Date())=>choice==='today'?instant.toLocaleDateString('sv-SE',{timeZone:'Asia/Tokyo'}):choice;
const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>Number.isFinite(v)?(v*100).toFixed(2)+'%':'—';
const yen=v=>Number.isFinite(v)?v.toLocaleString('ja-JP')+'円':'—';
const clock=v=>v?new Date(v).toLocaleTimeString('ja-JP',{timeZone:'Asia/Tokyo',hour12:false}):'未取得';
const reasons={awaiting_T10:'T−10の保存待ち',forecast_not_saved:'保存予想なし',forecast_missing:'予想欠損',fourth_sidecar_pending:'対照型の記録待ち',control_prediction_missing:'対照型欠測',result_roster_missing_or_duplicate:'結果の全頭照合待ち',full_order_flash_waiting_for_confirmed_hr:'全馬速報・払戻確定待ち',result_status_identity_or_cancellation_unconfirmed:'結果区分・馬ID・取消の確認待ち',hr_se_winner_mismatch:'着順と払戻の対応を保留',result_source_unavailable:'DB結果の取得待ち',needs_authentication:'DB認証が利用できません',dead_heat_or_unconfirmed_winner:'同着などの特殊結果は保留',artifact_invalid:'保存記録の整合性を確認できません',schedule_or_roster_changed_after_registration:'発走予定・出走馬変更のため参考表示',forecast_vetoed:'予想の採用条件を満たさず照合対象外'};
const reason=v=>reasons[v]||v||'待機中';
Object.assign(reasons,{odds_invalid:'単勝オッズの不整合により欠測',ledger_tampered:'保存記録の変更を検知したため保留',forecast_missing_or_vetoed:'欠測・無効のため照合対象外',result_unavailable:'公式結果待ち',result_rank_unconfirmed:'公式着順の確認待ち',unestablished_or_no_confirmed_winner:'不成立・勝馬未確定のため保留',special_pending:'特殊払戻の確認待ち',refund_pending:'返還を含む公式払戻の確認待ち',result_unconfirmed:'公式払戻の確定待ち',result_duplicate:'払戻訂正・複数版の確認待ち'});
reasons.full_order_not_received='勝馬・単勝払戻は確認済み。2着以降の全着順は未取得です';
const horse=(r,h)=>r.horse_names?.[h]||'馬名未取得';
export function picks(r){return Object.fromEntries(FOUR.map(n=>[n,r.models[n]?.rows?.find(x=>x.rank===1)?.horse_no??null]));}
export function agreement(r){let selected=picks(r),p=(r.forecast_kind==='current_reference'?D39:FOUR).map(n=>selected[n]);return p.some(x=>x===null)?'pending':new Set(p).size===1?'same':'different';}
export function summarize(races){
  const shared=races.filter(r=>D39.every(n=>r.models[n]?.status==='saved')),common=shared.filter(r=>r.result_status==='confirmed');
  return {common:common.length,financial_common:shared.length,models:D39.map(name=>{
    let items=common.map(r=>r.models[name]),tickets=shared.map(r=>r.models[name].paper_ticket).filter(Boolean),settled=tickets.filter(t=>t.status==='settled'),pending=tickets.filter(t=>t.status==='pending');
    let investment=tickets.reduce((s,t)=>s+t.stake_yen,0),paid=settled.reduce((s,t)=>s+t.payout_yen,0),settledInvestment=settled.reduce((s,t)=>s+t.stake_yen,0);
    return {name,races:items.length,hits:items.filter(x=>x.top_hit).length,top3:items.filter(x=>typeof x.top_finish==='number'&&x.top_finish<=3).length,
      bets:tickets.filter(t=>t.stake_yen>0).length,investment,pending:pending.length,payout:pending.length?null:paid,profit:pending.length?null:paid-investment,roi:pending.length||!investment?null:paid/investment,
      settledInvestment,settledPayout:paid};
  })};
}
function stats(items){return items.map(([title,value])=>`<article><strong>${e(value)}</strong><span>${e(title)}</span></article>`).join('');}
export const finishLabel=v=>typeof v==='number'?`${v}着`:v||'未確定';
export function watchPick(model){
  if(!model)return '';
  const names={win:'単勝的中',loss:'不的中',pending:'結果待ち',missing:'予測欠測',held:'保留・判定対象外',refunded:'返還'};
  let text=names[model.selection_state]||'確認待ち',t=model.ticket;
  if(model.top_finish!==null)text+=' · '+finishLabel(model.top_finish);
  if(t?.status==='settled'||t?.status==='refunded')text+=' · 払戻 '+yen(t.payout_yen);
  else if(model.selection_state==='win'||model.selection_state==='loss')text+=' · 払戻確認待ち';
  return `<div class="watch-pick watch-${e(model.selection_state)}">${e(text)}</div>`;
}
function topCards(r,names){return `<div class="picks ${names.length===3?'three':''}">${names.map(n=>{let model=r.models[n],top=model?.rows?.find(x=>x.rank===1),watched=r.watch?.models[n];return `<div class="pick ${n==='H'?'market':''} ${watched?.selection_state==='win'?'winner':''}"><small>${labels[n]}</small><strong>${top?'馬番 '+e(top.horse_no):'予想なし'}</strong><span>${top?e(horse(r,top.horse_no))+' · '+pct(top.probability):e(reason(model?.reason||r.reason))}</span>${watchPick(watched)}</div>`;}).join('')}</div>`;}
function predictedTable(r){
  let horses=r.roster||[];
  let names=r.forecast_kind==='current_reference'?D39:FOUR;
  if(!Object.values(r.models).some(m=>m.status==='saved'))return '';
  return `<div class="scroll"><table><thead><tr><th>馬番 / 馬名</th>${r.watch?'<th>公式着順</th>':''}${names.map(n=>`<th>${labels[n]}<small>順位 / 勝率</small></th>`).join('')}</tr></thead><tbody>${horses.map(h=>`<tr data-horse="${e(h)}" class="finish-${e(r.watch?.finishes[h])}"><td>${e(h)} · ${e(horse(r,h))}</td>${r.watch?`<td>${e(finishLabel(r.watch.finishes[h]))}</td>`:''}${names.map(n=>{let m=r.models[n],x=m?.rows?.find(x=>x.horse_no===h);return `<td class="${x?.rank===1?'top':''}" data-model="${n}" ${x?`data-probability="${x.probability}" data-rank="${x.rank}"`:''}>${x?`${x.rank}位 <span title="${x.probability}">${pct(x.probability)}</span>`:'欠測'}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function raceHeading(r){return `<span>${e(r.venue||'競馬場未取得')} <b>${r.race_no}R</b></span><span class="muted">${e(r.date)} · ${clock(r.scheduled_start)}発走予定</span>`;}
export function watchOutcome(r){
  let w=r.watch;if(!w)return '';
  const fullTitle=w.result_basis==='official_SE7'?'公式確定着順確認済み':'公式全着順速報・単勝払戻確認済み';
  const title={confirmed:fullTitle,winner_confirmed:'公式勝馬・単勝払戻確認済み',provisional:'全着順速報・払戻確認待ち',pending:'公式結果待ち'}[w.result_status]||'確認待ち';
  return `<div class="official-result"><b>${title}</b>${w.podium.length?`<div class="podium">${w.podium.map(p=>`<span class="place-${p.finish}"><b>${p.finish}着</b> ${e(p.horse_no)} ${e(horse(r,p.horse_no))}</span>`).join('')}</div>`:''}${w.reason?`<p class="muted">${e(reason(w.reason))}</p>`:''}${w.payout_status!=='confirmed'&&w.result_status==='confirmed'?`<p class="muted">${e(reason(w.payout_reason||'result_unavailable'))}。未精算として表示します。</p>`:''}<small>結果照合 ${clock(w.checked_at)}</small></div>`;
}
function predictionRace(r){let a=agreement(r),ref=r.forecast_kind==='current_reference',names=ref?D39:FOUR;return `<details class="race" data-race="${e(r.race_id)}"><summary><div class="race-heading">${raceHeading(r)}<span class="tag ${a==='same'?'good':''}">${a==='same'?(ref?'D39 3モデル本命一致':'4モデル本命一致'):a==='different'?'本命が異なる':'予想待ち・欠損'}</span></div>${topCards(r,names)}${watchOutcome(r)}${r.reason?`<p class="muted">${e(reason(r.reason))}</p>`:''}</summary><div class="race-body">${predictedTable(r)}<p class="provenance">${ref?'参考予想・出馬表取得 '+clock(r.source_observed_at):'入力元 '+e(r.odds_source||'未取得')+' / 判断時点 '+clock(r.input_cutoff)+' / オッズ公表 '+clock(r.odds_published_at)+' / PC初回観測 '+clock(r.odds_first_seen_at)} / 保存予測 ${clock(r.predicted_at)}<br>${names.map(n=>`${labels[n]}：${e(r.models[n]?.model_id||'未保存')} · T=${e(r.models[n]?.temperature??'—')} · 生成 ${clock(r.models[n]?.predicted_at)}`).join('<br>')}<br>保存元SHA-256：${e(r.source_prediction_sha256||'未保存')}</p></div></details>`;}

export function watchSummary(races){
  return Object.fromEntries(FOUR.map(name=>{
    let items=races.map(r=>r.watch.models[name]),eligible=items.filter(m=>m.eligible),decided=eligible.filter(m=>typeof m.top_hit==='boolean');
    let settled=eligible.map(m=>m.ticket).filter(t=>['settled','refunded'].includes(t.status));
    let investment=settled.reduce((s,t)=>s+t.stake_yen,0),paid=settled.reduce((s,t)=>s+t.payout_yen,0),hits=decided.filter(m=>m.top_hit).length;
    return [name,{target_races:races.length,prediction_saved:items.filter(m=>m.prediction_saved).length,
      eligible_races:eligible.length,missing:items.filter(m=>!m.prediction_saved).length,held:items.filter(m=>m.selection_state==='held').length,
      result_confirmed_races:decided.length,hits,hit_rate:decided.length?hits/decided.length:null,
      refunded:eligible.filter(m=>m.selection_state==='refunded').length,bet_count:eligible.length,committed_investment_yen:100*eligible.length,
      settled_count:settled.length,unsettled_count:eligible.filter(m=>m.ticket.status==='pending').length,
      investment_yen:investment,payout_yen:paid,net_profit_yen:paid-investment,return_rate:investment?paid/investment:null}];
  }));
}
export function watchBoard(races){
  let scores=watchSummary(races);
  return `<h2>4モデル対抗スコアボード</h2><p class="muted">表示中の${races.length}レース。本命1位を単勝100円で仮想購入（EV選別なし）。的中率の分母は有効予想の勝敗確認済み件数で、返還は含めません。収支・回収率は精算済み分のみです。</p><div class="scroll"><table id="watch-summary"><thead><tr><th>モデル</th><th>対象</th><th>予測保存</th><th>本命的中 / 確認済み</th><th>的中率</th><th>仮想投資額<small>全件 / 精算済み</small></th><th>公式払戻<small>精算済み</small></th><th>仮想収支<small>精算済み</small></th><th>回収率<small>精算済み</small></th><th>精算済み / 未精算</th><th>返還</th><th>欠測 / 保留</th></tr></thead><tbody>${FOUR.map(n=>{let m=scores[n];return `<tr data-watch-model="${n}"><td>${labels[n]}</td><td>${m.target_races}</td><td>${m.prediction_saved}</td><td>${m.hits} / ${m.result_confirmed_races}</td><td>${pct(m.hit_rate)}</td><td>${yen(m.committed_investment_yen)}<small>精算済み ${yen(m.investment_yen)}</small></td><td>${m.settled_count?yen(m.payout_yen):'—'}</td><td class="${m.net_profit_yen>0?'positive':m.net_profit_yen<0?'negative':''}">${m.settled_count?yen(m.net_profit_yen):'—'}</td><td>${pct(m.return_rate)}</td><td>${m.settled_count} / ${m.unsettled_count}</td><td>${m.refunded}</td><td>${m.missing} / ${m.held}</td></tr>`;}).join('')}</tbody></table></div><p class="muted">未精算の払戻を0円の損失として足しません。欠測・保留の仮想投資は0円。取消返還は投資100円・返還100円として区別します。開催中の暫定値は公式訂正に応じて更新され、保存予想は変更しません。</p>`;
}
function resultRace(r){
  if(r.result_status!=='confirmed')return `<article class="panel"><div class="race-heading">${raceHeading(r)}</div><p>${e(reason(r.reason))}</p></article>`;
  let horses=Object.keys(r.finishes).sort((a,b)=>a.localeCompare(b));
  return `<details class="race" data-result-race="${e(r.race_id)}"><summary><div class="race-heading">${raceHeading(r)}<span class="tag">${r.payout_status==='confirmed'?'着順・払戻確認済み':'着順確認・払戻待ち'}</span></div>${topCards(r,D39)}<div class="result-picks">${D39.map(n=>`${labels[n]}：${r.models[n]?.status==='saved'?e(r.models[n].top_finish)+'着'+(r.models[n].top_hit?' / 的中':''):'欠測'}`).join('　')}</div></summary><div class="race-body"><div class="scroll"><table><thead><tr><th>馬番</th><th>着順</th>${D39.map(n=>`<th>${labels[n]}<small>順位 / 勝率</small></th>`).join('')}<th>単勝払戻 / 100円</th></tr></thead><tbody>${horses.map(h=>`<tr class="finish-${r.finishes[h]}" data-horse="${e(h)}"><td>${e(h)}</td><td>${e(r.finishes[h])}${typeof r.finishes[h]==='number'?'着':''}</td>${D39.map(n=>{let row=r.models[n]?.rows?.find(x=>x.horse_no===h);return `<td class="${row?.rank===1?'top':''}">${row?`${row.rank}位 / ${pct(row.probability)}`:'欠測'}</td>`;}).join('')}<td>${yen(r.models[D39.find(n=>r.models[n]?.rows)]?.rows?.find(x=>x.horse_no===h)?.win_payout_100)}</td></tr>`).join('')}</tbody></table></div><p class="provenance">結果の根拠：${e(r.result_basis)}。単勝精算：${e(r.payout_status)}。予想値は保存後に変更していません。<br>${D39.map(n=>`${labels[n]}：${e(r.models[n]?.model_id||'未取得')} · 予測保存 ${clock(r.models[n]?.predicted_at)}`).join('<br>')}<br>元予想SHA-256：${e(r.source_prediction_sha256||'未取得')}</p></div></details>`;
}
function resultSummary(data){let s=summarize(data.races);return `<h2>D39共通 ${s.common}レースの探索集計</h2><p class="muted">的中率は3モデル共通・着順確認済み。収支は3モデル共通の発走済み保存購入 ${s.financial_common}レース（未精算含む）。Hは対象外です。正式な優越判定はしません。</p><div class="scroll"><table id="d39-summary"><thead><tr><th>モデル</th><th>本命1着 / 対象</th><th>本命1着率</th><th>3着以内率</th><th>仮想購入</th><th>投資</th><th>払戻</th><th>純利益</th><th>回収率</th><th>未精算</th></tr></thead><tbody>${s.models.map(m=>`<tr data-model="${m.name}"><td>${labels[m.name]}</td><td>${m.hits} / ${m.races}</td><td>${m.races?pct(m.hits/m.races):'—'}</td><td>${m.races?pct(m.top3/m.races):'—'}</td><td>${m.bets}件</td><td>${yen(m.investment)}</td><td>${yen(m.payout)}</td><td>${yen(m.profit)}</td><td>${pct(m.roi)}</td><td>${m.pending}件</td></tr>`).join('')}</tbody></table></div><p class="muted">払戻が未確認の購入を損失0円とせず、総払戻・収益を保留します。購入0件の回収率は算出不能です。</p>`;}
if(typeof document!=='undefined'){
  const $=q=>document.querySelector(q);let data=null,busy=false,stopped=false,phase='official';
  const mode=document.body.dataset.mode;
  const selectedDay=()=>resolveDay($('#day').value);
  function updateOptions(select,values,label){
    if(JSON.stringify(Array.from(select.options).slice(1).map(o=>o.value))===JSON.stringify(values.map(String)))return;
    const current=select.value;
    select.innerHTML='<option value="">すべて</option>'+values.map(v=>`<option value="${e(v)}" ${String(v)===current?'selected':''}>${e(label(v))}</option>`).join('');
  }
  function render(){
    if(!data)return;
    let opened=new Set([...document.querySelectorAll('details[open]')].map(x=>x.dataset.race||x.dataset.resultRace)),venue=$('#venue').value;
    const venues=[...new Set(data.races.map(r=>r.venue).filter(Boolean))].sort();updateOptions($('#venue'),venues,v=>v);
    let list=data.races.filter(r=>!venue||r.venue===venue);
    if(mode==='predictions'){
      let filter=$('#agreement').value;if(filter)list=list.filter(r=>agreement(r)===filter);
      const raceNo=$('#race-no')?.value||'';
      if($('#race-no'))updateOptions($('#race-no'),[...new Set(data.races.map(r=>r.race_no))].sort((a,b)=>a-b),n=>`${n}R`);
      if(raceNo)list=list.filter(r=>String(r.race_no)===raceNo);
      const hit=$('#hit-status')?.value;if(hit&&data.watch_format_version)list=list.filter(r=>Object.values(r.watch?.models||{}).some(m=>m.selection_state===hit));
      const ref=data.forecast_kind==='current_reference';$('#agreement option[value="same"]').textContent=ref?'D39 3モデル一致':'4モデル一致';
      if(ref)$('#phase-note').textContent=`参考予想の生成完了：${clock(data.source?.generated_at)}。出馬表による最新の保存版を表示しています。T−10正式記録・結果集計には混ぜません。HはT−10まで待機します。`;
      $('#coverage').innerHTML=stats(ref?[['対象レース',data.coverage.registered_races],['参考予想保存済み',data.coverage.prediction_races],['D39 3モデル揃い',data.coverage.three_model_races],['予想対象の頭数',data.coverage.registered_runners]]:[['対象レース',data.coverage.registered_races],['予想保存済み',data.coverage.prediction_races],['4モデル揃い',data.coverage.four_model_races],['保存待ち・欠損',data.coverage.registered_races-data.coverage.four_model_races]]);
      $('#races').innerHTML=list.map(predictionRace).join('')||'<div class="panel">対象の保存記録はまだありません。記録器のT−10保存後に自動表示します。</div>';
      if($('#watch-scoreboard')){$('#watch-scoreboard').hidden=ref;$('#watch-scoreboard').innerHTML=ref?'':watchBoard(list);}
    }else{
      $('#coverage').innerHTML=stats([['結果照合対象',data.races.length],['着順確認済み',data.races.filter(r=>r.result_status==='confirmed').length],['確認待ち',data.races.filter(r=>r.result_status!=='confirmed').length+data.pending_prediction_races]]);
      $('#summary').innerHTML=resultSummary(data);$('#races').innerHTML=list.map(resultRace).join('')||'<div class="panel">発走済みの保存予想と確定結果の到着を待っています。開催前・T−10保存前の結果は取得しません。</div>';
      $('#download').href='/api/d39-results.csv?date='+data.date;
    }
    document.querySelectorAll('details').forEach(x=>{if(opened.has(x.dataset.race||x.dataset.resultRace))x.open=true;});
    $('#status').textContent=`● 自動更新中 · 確認 ${clock(data.checked_at)}${data.reason?' · '+reason(data.reason):''}${data.stale?' · 前回確認 '+clock(data.last_source_checked_at)+' の結果を表示中':''}`;
  }
  async function refresh(){if(busy||stopped)return;busy=true;const day=selectedDay();try{
    if(data&&data.date!==day){data=null;$('#races').innerHTML='';$('#coverage').innerHTML='';if($('#summary'))$('#summary').innerHTML='';}
    const chosen=phase,api=mode==='results'?'/api/d39-results':phase==='reference'?'/api/reference':'/api/watch';
    const response=await fetch(api+'?date='+day,{cache:'no-store',credentials:'same-origin'});const body=await response.json();if(!response.ok||body.version!=='keiba-lab-live.20261024.1')throw Error();if(day!==selectedDay()||chosen!==phase)return;
    for(const value of body.available_dates||[]){if(!Array.from($('#day').options).some(o=>o.value===value)){const option=document.createElement('option');option.value=value;option.textContent=value;$('#day').appendChild(option);}}
    data=body;render();
  }catch{$('#status').textContent='接続を確認できません。新規表示を保留し、前回の保存内容を維持しています。';}finally{busy=false;}}
  $('#day').addEventListener('change',()=>{data=null;$('#races').innerHTML='';$('#coverage').innerHTML='';if($('#summary'))$('#summary').innerHTML='';refresh();});$('#venue').addEventListener('change',render);$('#agreement')?.addEventListener('change',render);
  $('#race-no')?.addEventListener('change',render);$('#hit-status')?.addEventListener('change',render);
  document.querySelectorAll('[data-phase]').forEach(b=>b.addEventListener('click',()=>{phase=b.dataset.phase;data=null;$('#races').innerHTML='';$('#coverage').innerHTML='';document.querySelectorAll('[data-phase]').forEach(x=>x.classList.toggle('active',x===b));$('#phase-note').textContent=phase==='reference'?'出馬表による参考予想です。生成時刻を表示し、T−10正式記録・結果集計には混ぜません。HはT−10まで待機します。':'T−10で実際に保存された4モデルの固定予想です。まだ保存されていないレースは待機表示です。';refresh();}));
  $('#toggle-details')?.addEventListener('click',()=>{const items=[...document.querySelectorAll('#races details')],open=items.some(x=>!x.open);items.forEach(x=>x.open=open);$('#toggle-details').textContent=open?'全レースを折りたたむ':'全レースの順位を展開';});
  window.addEventListener('pagehide',()=>{stopped=true;});setInterval(refresh,15000);refresh();
}
