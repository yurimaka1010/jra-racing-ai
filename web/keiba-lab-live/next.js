const NAMES=['old_D39','new_D39','control_D39','H','next_D39'];
const LABELS={old_D39:'旧D39',new_D39:'新D39',control_D39:'D39対照型',H:'主力H',next_D39:'次世代47項目（改善前）',enriched_D39:'次世代47項目（前走タイム接続）'};
export function candidateView(data,selection='improved'){
  return selection==='improved'&&data?.improved?.status==='active'?data.improved:data;
}
export function csvDownloadState(data,selection='improved'){
  const info=data?.csv_exports?.[selection];
  if(info?.status!=='saved')return {ready:false,note:'正式予想と保存済み結果のCSVを準備しています。参考予想は結果CSVへ混ぜません。'};
  const query='date='+encodeURIComponent(data.date)+'&candidate='+encodeURIComponent(selection);
  return {ready:true,detail:'/api/next-results.csv?'+query+'&kind=detail',summary:'/api/next-results.csv?'+query+'&kind=summary',
    note:`CSV自動保存済み：${info.races}レース・明細${info.rows}行。未確定の着順・払戻は空欄、主力Hの結果は含みません。`};
}
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>typeof x==='number'&&Number.isFinite(x)?(100*x).toFixed(2)+'%':'—';
const number=x=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(6):'—';
const horseName=(r,h)=>r.horse_names?.[h]||'馬名未取得';
const reasonLabel=x=>({model_not_active_before_cutoff:'このモデルの起動前にT−10を通過',save_deadline_missed:'予想の保存期限を超過',
  original_on_time_input_unavailable:'締切前の元入力が未保存',forecast_not_saved:'正式予想の保存なし',saved_record_unavailable:'保存記録を確認できません',
  prediction_error:'予想処理エラー',forecast_vetoed:'発走時刻・出走馬の変更等により対象外',awaiting_T10:'T−10の記録時刻を待機',
  prediction_in_progress:'予想の保存を確認中',formal_prediction_unavailable:'正式予想なし・評価対象外',
  official_result_pending:'公式結果待ち'}[x]||x||'');
export function displayedRaces(view,onlyUnsaved=false){
  return onlyUnsaved?view.races.filter(r=>r.candidate_status&&r.candidate_status!=='saved'):view.races;
}
const stamp=s=>{const d=new Date(s);return s&&Number.isFinite(d.getTime())?new Intl.DateTimeFormat('ja-JP',{timeZone:'Asia/Tokyo',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(d):'未取得';};
export function predictionView(data,mode='reference'){
  const coverage=data.existing_coverage||{};
  if(mode==='reference'){
    const ref=data.reference||{};
    return {races:ref.races||[],names:[data.candidate_name||'next_D39'],results:false,
      note:`出馬表による参考予想 · 生成完了 ${stamp(ref.generated_at)}（日本時間）。最新の保存版を15秒ごとに確認します。正式T−10・実戦成績とは別枠です。`,
      stats:[['対象レース',coverage.registered_races??0],['参考予想保存済み',ref.races?.length??0],['対象頭数',ref.runners??0],['生成完了',stamp(ref.generated_at)]],
      empty:'参考予想の保存版を待っています。未取得の予想は補完しません。'};
  }
  const results=mode==='results';
  const names=data.candidate_name?[...NAMES,data.candidate_name]:NAMES;
  const complete=data.display_coverage;
  return {races:results?(data.display_result_races??data.results?.races??[]):(data.display_races??data.races??[]),names:results?names.filter(n=>n!=='H'):names,results,
    note:results?'全対象レースを表示し、予想未保存・公式結果待ちは理由付きで残します。集計は正式に保存されたD39系だけ。主力Hの結果は含めません。':'全対象レースのT−10固定予想です。本モデルが未保存でも他モデルの保存予想は表示し、参考予想で欠測を埋めません。',
    stats:complete?[['全対象',complete.target_races],['本モデル正式保存',complete.selected_forecast_races],['本モデル未保存',complete.missing_forecast_races+complete.pending_forecast_races],['本モデル結果確認',complete.selected_result_races]]:
      [['既存対象',coverage.registered_races??0],['既存4モデル保存',coverage.four_model_races??0],['新モデル正式保存',data.races?.length??0],['D39共通結果',data.results?.common_result_races??0]],
    empty:results?'公式結果の到着後に表示します。':'正式T−10予想の保存を待っています。「現在の参考予想」で保存済み出馬表予想を確認できます。'};
}
export function metricsRows(data){
  const r=data?.results;if(!r||r.H_results_included!==false)return '';
  return Object.entries(r.metrics||{}).filter(([n])=>n!=='H').map(([n,m])=>{let f=r.financial?.[n]||{};
    return `<tr data-model="${esc(n)}"><td>${esc(LABELS[n])}</td><td>${m.races}</td><td>${number(m.race_log_loss)}</td><td>${number(m.brier_score)}</td><td>${number(m.roc_auc)}</td><td>${number(m.within_race_auc)}</td><td>${number(m.top1_accuracy_percent)}%</td><td>${number(m.top2_percent)}%</td><td>${number(m.top3_percent)}%</td><td>${number(m.winner_prediction_mean_rank)}</td><td>${f.bet_count??0}</td><td>${f.net_profit_yen??'未精算'}</td><td>${f.return_rate_percent==null?'—':f.return_rate_percent.toFixed(2)+'%'}</td><td>${f.unsettled_count??0}</td></tr>`;
  }).join('');
}
export function raceTable(r,results=false,selectedNames=NAMES){
  const names=results?selectedNames.filter(n=>n!=='H'):selectedNames;
  const horses=results?(r.roster||Object.keys(r.finishes||{}).sort()):r.roster||[];
  const label=r.candidate_status==='saved'?(results&&r.candidate_result_status!=='confirmed'?'本モデル：結果待ち':'本モデル：保存済み'):r.candidate_status==='pending'?'本モデル：待機中':r.candidate_status==='missing'?'本モデル：未保存':'';
  const displayReason=[reasonLabel(r.candidate_reason),reasonLabel(r.reason)].filter(Boolean).join(' / ');
  return `<details class="race" data-race="${esc(r.race_id)}"><summary><div class="race-heading"><strong>${esc(r.venue)} ${r.race_no}R</strong><span>${esc(r.scheduled_start)}</span>${label?`<span class="tag">${esc(label)}</span>`:""}</div><div class="picks research-picks">${names.map(n=>{const m=r.models[n],top=m?.rows?.find(v=>v.rank===1);return `<div class="pick"><small>${esc(LABELS[n])}</small><strong>${top?'馬番 '+esc(top.horse_no):'未保存'}</strong><span>${top?esc(horseName(r,top.horse_no))+' · '+pct(top.probability):'待機・欠損'}${results&&m?.top_finish!=null?' / '+esc(m.top_finish)+'着':''}</span></div>`;}).join('')}</div></summary><div class="race-body">${displayReason?`<p class="notice">${esc(displayReason)}</p>`:''}<div class="scroll"><table><thead><tr><th>馬番 / 馬名${results?' / 着順':''}</th>${names.map(n=>`<th>${esc(LABELS[n])}</th>`).join('')}</tr></thead><tbody>${horses.map(h=>`<tr data-horse="${esc(h)}"><td>${esc(h)} · ${esc(horseName(r,h))}${results?' / '+esc(r.finishes?.[h]??'未確定'):''}</td>${names.map(n=>{const v=r.models[n]?.rows?.find(x=>x.horse_no===h);return `<td class="${v?.rank===1?'top':''}" data-model="${esc(n)}" ${v?`data-probability="${v.probability}" data-rank="${v.rank}"`:''}>${v?v.rank+'位 / '+pct(v.probability):'—'}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div><p class="provenance">${names.map(n=>`${esc(LABELS[n])}：${esc(r.models[n]?.model_id||'未保存')} / ${esc(r.models[n]?.predicted_at||'')}`).join('<br>')}</p></div></details>`;
}
if(typeof document!=='undefined'){
  let data=null,mode='reference',busy=false,renderedMode=null;
  const $=s=>document.querySelector(s);
  function render(){
    if(!data)return;
    const active=candidateView(data,$('#candidate').value);
    const view=predictionView(active,mode),results=view.results;
    const onlyMissing=mode!=='reference'&&$('#next-missing-only').checked;
    const visibleRaces=displayedRaces(view,onlyMissing);
    $('#next-missing-only').disabled=mode==='reference';
    const opened=renderedMode===mode?new Set([...document.querySelectorAll('#next-races details[open]')].map(x=>x.dataset.race)):new Set();
    $('#next-status').textContent=active.status==='active'?'● 独立記録を自動更新中 · '+active.checked_at+(active.result_poll?.status==='pending'?' · 結果取得は保留。前回確認 '+(active.results?.checked_at||'未取得')+' の内容を維持':''):'新モデルの完成・起動を待っています。';
    $('#next-model').textContent=active.activation?`モデル ${active.activation.model_id} / T=${active.activation.temperature} / 開始 ${active.activation.activated_at}`:'';
    $('#next-feature-note').textContent=active.feature_note||'改善前の保存予想です。予測値は変更せず保持しています。';
    const csvState=csvDownloadState(data,$('#candidate').value);
    for(const [id,key] of [['download-next-detail','detail'],['download-next-summary','summary']]){
      const link=$('#'+id);link.setAttribute('aria-disabled',String(!csvState.ready));
      if(csvState.ready)link.setAttribute('href',csvState[key]);else link.removeAttribute('href');
    }
    $('#next-csv-note').textContent=csvState.note;
    $('#next-coverage').innerHTML=view.stats.map(([k,v])=>`<article><strong>${esc(v)}</strong><span>${esc(k)}</span></article>`).join('');
    $('#next-view-note').textContent=view.note+(onlyMissing?` 一覧の表示${visibleRaces.length}/${view.races.length}レース。絞込みで集計母数は変わりません。`:'');
    document.querySelectorAll('[data-view]').forEach(button=>{const chosen=button.dataset.view===mode;button.classList.toggle('active',chosen);button.setAttribute('aria-pressed',String(chosen));});
    const measured=metricsRows(active);
    $('#next-metrics').innerHTML=results&&measured?'<h2>共通レースの前向き探索集計</h2><div class="scroll"><table><thead><tr><th>モデル</th><th>レース</th><th>レースLL</th><th>Brier</th><th>AUC</th><th>レース内AUC</th><th>本命1着</th><th>本命2着</th><th>本命3着</th><th>勝馬の平均予測順位</th><th>仮想購入</th><th>純利益</th><th>回収率</th><th>未精算</th></tr></thead><tbody>'+measured+'</tbody></table></div>':'';
    if(results&&active.results?.condition_diagnostics?.length){
      const parts=active.results.condition_diagnostics.filter(r=>r.model!=='H');
      $('#next-metrics').innerHTML+='<details class="panel"><summary>距離・芝ダート・人気別の探索比較（件数付き）</summary><div class="scroll"><table><thead><tr><th>区分</th><th>条件</th><th>モデル</th><th>レース</th><th>LL</th><th>Brier</th><th>1位的中率</th></tr></thead><tbody>'+parts.map(r=>`<tr><td>${esc(r.axis)}</td><td>${esc(r.group)}</td><td>${esc(LABELS[r.model])}</td><td>${r.races}</td><td>${number(r.race_log_loss)}</td><td>${number(r.brier_score)}</td><td>${number(r.top1_accuracy_percent)}%</td></tr>`).join('')+'</tbody></table></div></details>';
    }
    $('#next-races').innerHTML=visibleRaces.map(r=>raceTable(r,results,view.names)).join('')||'<div class="panel">'+esc(onlyMissing?'未保存の対象レースはありません。':view.empty)+'</div>';
    document.querySelectorAll('#next-races details').forEach(x=>{if(opened.has(x.dataset.race))x.open=true;});
    renderedMode=mode;
  }
  async function refresh(){if(busy)return;busy=true;try{const response=await fetch('/api/next?date='+$('#day').value,{cache:'no-store'});if(!response.ok)throw Error();data=await response.json();render();}catch{$('#next-status').textContent='接続・保存確認を保留しています。前の表示を維持します。';}finally{busy=false;}}
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{mode=button.dataset.view;render();}));
  $('#candidate').addEventListener('change',()=>{renderedMode=null;render();});
  $('#next-missing-only').addEventListener('change',render);
  setInterval(refresh,15000);refresh();
}

