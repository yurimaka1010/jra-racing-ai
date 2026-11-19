// Offline DOM-logic test. This is not a browser, UI automation, or a layout test.
// No network API, browser engine, model inference, or real-data output is used.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const htmlPath = path.resolve(process.argv[2]);
const html = fs.readFileSync(htmlPath, 'utf8');
const blocks = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const encoded = blocks.find(x => x[1].includes('id="reviewData"'))[2];
const data = JSON.parse(encoded);
let checked = 0;
const check = (a, b) => { assert.deepEqual(a, b, 'offline_report_logic_mismatch'); checked++; };

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this._text = ''; this.value = '';
    this.hidden = false; this.checked = false; this.style = {}; this.dataset = {};
    this.events = {}; this.classList = { toggle() {} };
  }
  get textContent() { return this._text + this.children.map(x => x.textContent).join(''); }
  set textContent(value) { this._text = String(value); this.children = []; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = ''; this.children = [...nodes]; }
  addEventListener(type, callback) { this.events[type] = callback; }
  trigger(type) { this.events[type](); }
}
const ids = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(x => [x[1], new Element()]));
for (const match of html.matchAll(/<table id="([^"]+)"/g)) {
  ids[match[1]].tHead = new Element('thead'); ids[match[1]].tBodies = [new Element('tbody')];
}
ids.reviewData.textContent = encoded;
for (const id of ['day', 'venue', 'model']) ids[id].value = 'all';
ids.raceDetail.hidden = true;
if (ids.showMarketRows) ids.showMarketRows.checked = true;
const document = {
  documentElement: new Element('html'),
  getElementById(id) { assert.ok(Object.hasOwn(ids, id), 'unknown_report_element'); return ids[id]; },
  createElement(tag) { return new Element(tag); },
  createTextNode(text) { const n = new Element('#text'); n.textContent = text; return n; }
};
const scripts = blocks.filter(x => !x[1].includes('application/json')).map(x => x[2]).join('\n');
vm.runInNewContext(scripts, { document, Node: Element }, { timeout: 10000 });
check(document.documentElement.dataset.reportReady, 'true');
const format = (x, d = 6) => Number(x).toLocaleString('ja-JP', { minimumFractionDigits: d, maximumFractionDigits: d });
for (const [scopeKey, scope] of Object.entries(data.report.scopes)) {
  const [day, venue] = scopeKey.split('|'); ids.day.value = day; ids.venue.value = venue;
  ids.model.value = 'all'; ids.day.trigger('change');
  check(ids.targetCount.textContent, scope.target_races + 'レース');
  check(ids.accuracyCount.textContent, scope.P1_races + 'レース');
  check(ids.marketCount.textContent, scope.P2_races + 'レース');
  check(ids.missingCount.textContent, scope.odds_missing_races + 'レース');
  if (scope.P1_races) {
    check(ids.metricsTable.tBodies[0].children.length, 3);
    const first = ids.metricsTable.tBodies[0].children[0].children;
    check(first[2].textContent, format(scope.P1.control39.within_race_auc));
  }
  check(ids.metricsPlot.innerHTML, data.views[scopeKey].metrics);
  if (data.report.market_favorite) {
    const market = data.report.market_favorite.scopes[scopeKey];
    check(market.P3_races, scope.P3_races);
    check(market.old_three_common_races, scope.finance_common.control39.bet_count);
    check(ids.moneyTable.tBodies[0].children.length, 4);
    const last = ids.moneyTable.tBodies[0].children[3].children;
    const rate = market.finance_on_three_common.market_favorite.return_rate_percent;
    check(last[7].textContent, rate == null ? '未算出' : format(rate, 2) + '％');
  }
  if (data.report.popularity) {
    const pop = data.report.popularity.scopes[scopeKey];
    check(pop.P3_races, scope.P3_races);
    check(ids.popularityPanel.hidden, false);
    check(ids.popularityTable.tBodies[0].children.length, 3);
    check(ids.popularityPerformanceTable.tBodies[0].children.length, 12);
    check(ids.popularityPlot.innerHTML, data.views[scopeKey].popularity.all);
    for (const n of ['control39', 'pedigree45', 'reference47']) {
      check(pop.models[n].bands.reduce((sum, b) => sum + b.count, 0), pop.P3_races);
    }
  }
}
ids.day.value = 'all'; ids.venue.value = 'all'; ids.model.value = 'pedigree45'; ids.model.trigger('change');
check(ids.metricsTable.tBodies[0].children.length, 1);
check(ids.marketTable.tBodies[0].children.length, 2);
check(ids.moneyTable.tBodies[0].children.length, data.report.market_favorite ? 2 : 1);
if (data.report.popularity) {
  check(ids.popularityTable.tBodies[0].children.length, 1);
  check(ids.popularityPerformanceTable.tBodies[0].children.length, 4);
  check(ids.popularityPlot.innerHTML, data.views['all|all'].popularity.pedigree45);
}
ids.model.value = 'all'; ids.model.trigger('change'); ids.raceButton.trigger('click');
check(ids.raceDetail.hidden, false); check(ids.overview.hidden, true);
const rowModels = data.report.market_favorite ? 4 : 3;
check(ids.raceTable.tBodies[0].children.length, data.report.target_races * rowModels);
ids.differentOnly.checked = true; ids.differentOnly.trigger('change');
check(ids.raceTable.tBodies[0].children.length, data.report.details.filter(d => d.top_disagreement).length * rowModels);
ids.normalOnly.checked = true; ids.normalOnly.trigger('change');
check(ids.raceTable.tBodies[0].children.length, data.report.top_disagreement_P1_races * rowModels);
if (data.report.popularity) {
  ids.differentOnly.checked = false; ids.normalOnly.checked = false;
  const all = data.report.details.flatMap(d => [...['control39', 'pedigree45', 'reference47'].map(n => d.models[n]), ...(data.report.market_favorite ? [d.market_favorite] : [])]);
  ids.popBand.value = 'fav1'; ids.popBand.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, all.filter(m => m.T10_popularity_rank === 1).length);
  ids.popBand.value = 'unknown'; ids.popBand.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, all.filter(m => m.T10_popularity_rank == null).length);
  ids.popBand.value = 'all'; ids.popMin.value = '2'; ids.popMax.value = '3'; ids.popMax.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, all.filter(m => m.T10_popularity_rank >= 2 && m.T10_popularity_rank <= 3).length);
  ids.popMin.value = ''; ids.popMax.value = '';
  const expected = all.map(m => m.T10_popularity_rank);
  for (const direction of ['popular_first', 'unpopular_first']) {
    ids.popSort.value = direction; ids.popSort.trigger('change');
    const actual = ids.raceTable.tBodies[0].children.map(row => {
      const match = row.children[4].textContent.match(/^\d+/); return match ? Number(match[0]) : null;
    });
    const ordered = [...expected].sort((a, b) => a == null || b == null ? a == null && b == null ? 0 : a == null ? 1 : -1 : direction === 'popular_first' ? a - b : b - a);
    check(actual, ordered);
  }
  ids.model.value = 'pedigree45'; ids.popBand.value = 'fav2_3'; ids.model.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, data.report.details.filter(d => d.models.pedigree45.T10_popularity_band === 'fav2_3').length);
  check(ids.targetCount.textContent, data.report.target_races + 'レース');
  check(ids.accuracyCount.textContent, data.report.P1_eligible_races + 'レース');
  check(ids.marketCount.textContent, data.report.P2_eligible_races + 'レース');
}
if (data.report.market_favorite) {
  ids.popBand.value = 'all'; ids.popSort.value = 'race'; ids.model.value = 'all'; ids.model.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, data.report.target_races * 4);
  check(ids.raceTable.tHead.children[0].children.length, 10);
  ids.showMarketRows.checked = false; ids.showMarketRows.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, data.report.target_races * 3);
  ids.showMarketRows.checked = true; ids.showMarketRows.trigger('change');
  check(ids.raceTable.tBodies[0].children.length, data.report.target_races * 4);
}
ids.overviewButton.trigger('click'); check(ids.raceDetail.hidden, true); check(ids.overview.hidden, false);
const result = { status: 'passed', assertions: checked, tested_scopes: Object.keys(data.report.scopes).length,
  date_venue_model_filters: true, disagreement_filter: true, numeric_table_matches_saved_report: true,
  reference_popularity_sort_and_filters: Boolean(data.report.popularity),
  market_favorite_finance_and_rows: Boolean(data.report.market_favorite),
  real_browser_or_layout_verified: false, network_used: false, individual_rows_printed: false };
if (process.argv[3]) fs.writeFileSync(path.resolve(process.argv[3]), JSON.stringify(result, null, 2), { flag: 'wx' });
process.stdout.write(JSON.stringify(result) + '\n');
