import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
// Import as ESM without changing the existing repository's package format.
const text=await readFile(new URL('./app.js',import.meta.url),'utf8');
const {D39,FOUR,picks,agreement,summarize,resolveDay,watchSummary,watchBoard,watchPick,watchOutcome}=await import('data:text/javascript;base64,'+Buffer.from(text).toString('base64'));
const forecast=()=>({models:Object.fromEntries(FOUR.map(n=>[n,{status:'saved',rows:[{horse_no:'01',rank:1,probability:.6},{horse_no:'02',rank:2,probability:.4}]}]))});
test('four-model agreement uses predictions only',()=>{assert.equal(agreement(forecast()),'same');let r=forecast();r.models.H.rows[1].rank=1;r.models.H.rows[0].rank=2;assert.equal(agreement(r),'different');});
test('missing control is not an agreement or a zero probability',()=>{let r=forecast();r.models.control_D39={status:'missing'};assert.equal(picks(r).control_D39,null);assert.equal(agreement(r),'pending');});
test('reference agreement is D39-only and does not invent an H pick',()=>{let r=forecast();r.forecast_kind='current_reference';delete r.models.H;assert.equal(agreement(r),'same');assert.equal(picks(r).H,null);});
function result(){return {result_status:'confirmed',models:Object.fromEntries(D39.map(n=>[n,{status:'saved',top_hit:true,top_finish:1,paper_ticket:{status:'settled',stake_yen:100,payout_yen:450}}]))};}
test('summary uses only common D39 races, never an H result',()=>{let r=result();r.models.H={status:'saved',top_hit:false,paper_ticket:{stake_yen:1e9,payout_yen:0}};let s=summarize([r]);assert.equal(s.models.length,3);assert.equal(s.models[0].investment,100);assert.equal(s.models[0].roi,4.5);assert.ok(s.models.every(m=>m.name!=='H'));});
test('missing one D39 excludes that race from all common summaries',()=>{let r=result();r.models.control_D39={status:'missing'};assert.equal(summarize([r]).common,0);});
test('unsettled payouts do not become zero return',()=>{let r=result();r.models.new_D39.paper_ticket={status:'pending',stake_yen:100,payout_yen:null};let s=summarize([r]).models.find(m=>m.name==='new_D39');assert.equal(s.payout,null);assert.equal(s.roi,null);});
test('frozen purchases before result confirmation count as unsettled',()=>{let r=result();r.result_status='pending';for(let n of D39)r.models[n].paper_ticket={status:'pending',stake_yen:100,payout_yen:null};let s=summarize([r]);assert.equal(s.common,0);assert.equal(s.financial_common,1);assert.equal(s.models[0].pending,1);assert.equal(s.models[0].investment,100);assert.equal(s.models[0].payout,null);});
test('no purchases means ROI is unavailable',()=>{let r=result();for(let n of D39)r.models[n].paper_ticket={status:'no_bet',stake_yen:0,payout_yen:0};assert.equal(summarize([r]).models[0].roi,null);});
test('automatic date follows JST midnight without reopening the page',()=>{assert.equal(resolveDay('today',new Date('2026-10-24T14:59:59Z')),'2026-10-24');assert.equal(resolveDay('today',new Date('2026-10-24T15:00:00Z')),'2026-10-25');assert.equal(resolveDay('2026-10-24',new Date('2026-10-25T01:00:00Z')),'2026-10-24');});
test('prediction HTML contains no outcome summary section',async()=>{let html=await readFile(new URL('./index.html',import.meta.url),'utf8');assert.ok(!html.includes('id="summary"'));assert.ok(html.includes('data-mode="predictions"'));});
test('result HTML is explicitly D39 exploratory and not H evaluation',async()=>{let html=await readFile(new URL('./results.html',import.meta.url),'utf8');assert.ok(html.includes('探索'));assert.ok(html.includes('主力HとStage 7の途中成績'));assert.ok(!html.includes('Hの的中率'));});

function watched(state='win'){
  return {watch:{models:Object.fromEntries(FOUR.map(n=>[n,{prediction_saved:true,eligible:true,top_hit:state==='pending'?null:state==='win',selection_state:state,top_finish:state==='pending'?null:state==='win'?1:2,
    ticket:{status:state==='pending'?'pending':'settled',stake_yen:100,payout_yen:state==='pending'?null:state==='win'?450:0}}]))}};
}
test('watch scoreboard includes H and settled subset, without treating pending as losses',()=>{
  const s=watchSummary([watched(),watched('pending')]).H;
  assert.equal(s.hits,1);assert.equal(s.hit_rate,1);assert.equal(s.investment_yen,100);assert.equal(s.payout_yen,450);assert.equal(s.unsettled_count,1);
  assert.ok(watchBoard([watched()]).includes('主力H'));assert.ok(watchBoard([watched()]).includes('450円'));
});
test('vetoed and missing models do not enter R1 investment or hit rate',()=>{
  const r=watched();r.watch.models.H={prediction_saved:true,eligible:false,selection_state:'held',top_hit:null,ticket:{status:'excluded',stake_yen:0}};
  const s=watchSummary([r]).H;assert.equal(s.held,1);assert.equal(s.investment_yen,0);assert.equal(s.hit_rate,null);assert.equal(s.return_rate,null);
});
test('refund is separated from hit denominator',()=>{
  const r=watched();for(const n of FOUR)r.watch.models[n]={prediction_saved:true,eligible:true,selection_state:'refunded',top_hit:null,ticket:{status:'refunded',stake_yen:100,payout_yen:100}};
  const s=watchSummary([r]).H;assert.equal(s.refunded,1);assert.equal(s.net_profit_yen,0);assert.equal(s.return_rate,1);assert.equal(s.hit_rate,null);
});
test('watch winner uses official yen and never decimal odds',()=>{
  const html=watchPick(watched().watch.models.H);assert.ok(html.includes('単勝的中'));assert.ok(html.includes('450円'));
});
test('winner-only official record labels missing full order explicitly',()=>{
  const html=watchOutcome({horse_names:{},watch:{result_status:'winner_confirmed',podium:[{horse_no:'02',finish:1}],payout_status:'confirmed',reason:'full_order_not_received',checked_at:'2026-11-01T12:00:00+09:00'}});
  assert.ok(html.includes('公式勝馬・単勝払戻確認済み'));assert.ok(html.includes('2着以降の全着順は未取得'));
  assert.ok(!html.includes('公式着順確認済み'));
});
test('full-order flash is labeled separately from final confirmed order',()=>{
  const r={watch:{result_status:'confirmed',result_basis:'official_full_order_SE5_6_with_confirmed_HR',podium:[],payout_status:'confirmed'}};
  assert.ok(watchOutcome(r).includes('公式全着順速報・単勝払戻確認済み'));
  r.watch.result_basis='official_SE7';assert.ok(watchOutcome(r).includes('公式確定着順確認済み'));
});
