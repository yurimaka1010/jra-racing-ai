import { test } from 'node:test';
import assert from 'node:assert/strict';
import { metricsRows, raceTable, predictionView, candidateView, csvDownloadState, displayedRaces } from './next.js';
const race={race_id:'synthetic',venue:'テスト',race_no:1,roster:['01','02'],models:{
  H:{model_id:'sealed-H',rows:[{horse_no:'01',rank:1,probability:.6}]},
  next_D39:{model_id:'new-fake',rows:[{horse_no:'02',rank:1,probability:.7}]}
},finishes:{'01':2,'02':1}};
test('prediction view includes saved H and independent new-model scores',()=>{
  const html=raceTable(race);
  assert.match(html,/sealed-H/);assert.match(html,/new-fake/);
  assert.match(html,/data-probability="0.7"/);
});
test('answer view never exposes H model or its scores',()=>{
  const html=raceTable(race,true);
  assert.doesNotMatch(html,/sealed-H|主力H|data-model="H"/);
  assert.match(html,/new-fake/);
});
test('metrics renderer refuses a result marked H included',()=>{
  assert.equal(metricsRows({results:{H_results_included:true,metrics:{H:{}}}}),'');
});
test('unexpected H metrics are not rendered even in otherwise valid packet',()=>{
  assert.equal(metricsRows({results:{H_results_included:false,metrics:{H:{}}}}),'');
});
test('zero-purchase ROI and pending profits are not rendered as zero gain',()=>{
  const html=metricsRows({results:{H_results_included:false,
    metrics:{next_D39:{races:1}},financial:{next_D39:{bet_count:0,return_rate_percent:null,net_profit_yen:null}}}});
  assert.match(html,/未精算/);assert.match(html,/—/);
});
test('local source display escapes labels and cannot inject markup',()=>{
  assert.doesNotMatch(raceTable({...race,venue:'<img src=x onerror=alert(1)>'}),/<img/);
});
test('startup reference renders only the new model and never fabricates the four others',()=>{
  const html=raceTable(race,false,['next_D39']);
  assert.match(html,/new-fake/);assert.doesNotMatch(html,/sealed-H|旧D39|D39対照型/);
});
test('current saved reference is the default view without formal or result substitution',()=>{
  const data={reference:{races:[{race_id:'reference'}],runners:3,generated_at:'2026-10-25T07:51:09+09:00'},
    races:[{race_id:'formal'}],results:{races:[{race_id:'answer'}]},existing_coverage:{registered_races:36}};
  const view=predictionView(data);
  assert.deepEqual(view.races,[{race_id:'reference'}]);assert.deepEqual(view.names,['next_D39']);
  assert.equal(view.results,false);assert.match(view.note,/07:51:09/);assert.match(view.note,/別枠/);
});
test('empty formal view never backfills from available references',()=>{
  const view=predictionView({reference:{races:[race]},races:[]},'formal');
  assert.equal(view.races.length,0);assert.equal(view.results,false);
});
test('missing reference never presents a formal forecast as a current reference',()=>{
  assert.deepEqual(predictionView({races:[race]}).races,[]);
});
test('horse names appear on top-pick cards and complete runner rows',()=>{
  const html=raceTable({...race,horse_names:{'01':'テストホース一','02':'テストホース二'}},false,['next_D39']);
  assert.match(html,/テストホース一/);assert.match(html,/テストホース二/);
  assert.match(html,/data-horse="02"/);assert.match(html,/data-probability="0.7"/);
});
test('unknown horse names stay explicitly unavailable and names are escaped',()=>{
  assert.match(raceTable(race),/馬名未取得/);
  assert.doesNotMatch(raceTable({...race,horse_names:{'01':'<img onerror=x>','02':'<script>'}}),/<img|<script/);
});
test('new candidate defaults to its own saved packet while original remains selectable',()=>{
  const data={activation:{model_id:'original'},improved:{status:'active',activation:{model_id:'improved'}}};
  assert.equal(candidateView(data).activation.model_id,'improved');
  assert.equal(candidateView(data,'original').activation.model_id,'original');
  assert.equal(data.activation.model_id,'original');
});
test('new reference selects only the improved model and does not invent old probabilities',()=>{
  const view=predictionView({candidate_name:'enriched_D39',reference:{races:[]}});
  assert.deepEqual(view.names,['enriched_D39']);assert.deepEqual(view.races,[]);
});
test('optional candidate absence keeps the existing model usable',()=>{
  const data={activation:{model_id:'original'},improved:null};
  assert.equal(candidateView(data),data);
});
test('result view of improved candidate always removes sealed H',()=>{
  const view=predictionView({candidate_name:'enriched_D39'},'results');
  assert.ok(view.names.includes('enriched_D39'));assert.ok(!view.names.includes('H'));
  assert.ok(view.names.includes('next_D39'));
});
test('CSV links follow the selected candidate and keep detail and summary distinct',()=>{
  const data={date:'2026-10-25',csv_exports:{original:{status:'saved',races:2,rows:32},improved:{status:'saved',races:1,rows:20}}};
  assert.match(csvDownloadState(data).detail,/candidate=improved&kind=detail/);
  assert.match(csvDownloadState(data,'original').summary,/candidate=original&kind=summary/);
  assert.match(csvDownloadState(data).note,/主力Hの結果は含みません/);
});
test('missing candidate CSV never downloads another model as a fallback',()=>{
  assert.equal(csvDownloadState({csv_exports:{original:{status:'saved'}}},'improved').ready,false);
});
test('pending result CSV links stay unavailable instead of exporting references',()=>{
  assert.equal(csvDownloadState({reference:{races:[race]},csv_exports:{improved:{status:'pending'}}}).ready,false);
});
test('formal and result displays retain missing targets separately from the measured cohort',()=>{
  const data={races:[race],display_races:[race,{race_id:'missing',candidate_status:'missing'}],
    display_result_races:[{race_id:'result'},{race_id:'missing'}],results:{races:[race]}};
  assert.equal(predictionView(data,'formal').races.length,2);
  assert.equal(predictionView(data,'results').races.length,2);
  assert.equal(data.races.length,1);assert.equal(data.results.races.length,1);
});
test('unsaved-only filter changes the visible list without mutating statistics',()=>{
  const view={races:[{candidate_status:'saved'},{candidate_status:'pending'},{candidate_status:'missing'}]};
  assert.equal(displayedRaces(view,true).length,2);assert.equal(view.races.length,3);
});
test('result waiting rows show the known roster with unknown finishes and no H scores',()=>{
  const html=raceTable({...race,candidate_status:'missing',candidate_reason:'model_not_active_before_cutoff',finishes:{}},true);
  assert.match(html,/data-horse="01"/);assert.match(html,/未確定/);assert.match(html,/起動前にT−10を通過/);
  assert.doesNotMatch(html,/sealed-H|undefined|主力H/);
});
