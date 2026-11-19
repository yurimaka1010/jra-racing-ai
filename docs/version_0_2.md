# Version 0.2：単勝の期待値判断・仮想バックテスト

実装日：2026-10-16。対象は**単勝のみ**。Version 0.1の学習・複勝モデル・既存CLIは維持します。実購入・投票・外部API・自動ダウンロード機能はありません。追加インストールは不要です。

本人PCで予測・仮想購入・[既存HRと固定台帳の直接精算](v02_hr_settlement.md)・1番人気との比較まで完了しました。元の判断を変更せず、両戦略の未精算0を匿名集計で確認しました。新しい払戻パーサーや手入力の参照JSONを作らず、既存デコーダーと精算処理を接続しています。実収益レポート・個別明細はOneDrive外に保持し、以下は再実行と入力形式の手順です。

予測からの初回実行は [発表時刻基準の実行手順](v02_publication_run.md)を参照します。JRA発表時刻をPC実取得時刻へ置換せず、暫定的な事後シミュレーションと明記します。公式払戻未入力の場合でも購入判断を保存でき、利益・回収率は保留します。以下の実観測時刻を要求する既存モードと区別します。元の予測を別日付のレースへ付け替えることはしません。

## 最初の実行

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$taskV01 = Join-Path $env:LOCALAPPDATA 'JraRacingAI\ml-v01'
$taskV02 = Join-Path $env:LOCALAPPDATA 'JraRacingAI\ml-v02'

# 架空データのみ。購入→精算→月別・1番人気比較を実行してローカル保存。
& $taskPython -B -m jra_racing_ai.win_sim_cli demo --save
```

出力のartifact_idに対応する `$taskV02\demo\<ID>\report.md` と `summary.json` をPC上で確認します。`ledger.json` は個別明細なので外部共有しません。架空デモは12対象・9オッズ取得・8仮想購入・1件保留です。利益や回収率は意図的な架空例で、実データの収益を意味しません。

新規の実データ・派生データ・設定・個別台帳・レポートは **`C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v02`** 内のみです。既存モデル・予測の読込だけは `ml-v01` を許可します。OneDrive・クラウド属性・リンク・ネットワークドライブを既存の保存先検証で拒否します。OSや別ソフトの全バックアップ設定を検出するものではないため、本人も同期対象外であることを確認してください。

## A. 実オッズの可用性を少数・読み取り専用で確認

```powershell
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGDATABASE = 'pckeiba'
$env:PGUSER = 'jra_audit_reader'
& $taskPython -B -m jra_racing_ai.win_sim_cli availability --execute-local --sample-rows 100
```

パスワードは本人が非表示入力します。チャット・引数・ソース・ファイルへ保存しません。PostgreSQL起動、ユーザー作成、DB更新は行いません。

`public.jvd_o1` / `public.apd_sokuho_o1` の列構造と、直近キー順の**上限100行のDB内集計**を確認し、`inspection/<ID>/availability.json` へ保存します。標準出力はテーブル名・件数・集計日付範囲のみ。これは全期間の最古日を調べた結果ではありません。行数は1～1000、1SQL既定5秒、全体はバッチ・クエリ間の時間検査で制限します。通常heap/RLSなし以外は取得しません。全件COUNT・SQL更新・実オッズの画面表示はありません。

同じフォルダーに `db-config.draft.json` を作ります。**PCの列名・倍率・時刻の意味は空欄、候補テーブルは無効状態**です。指定のローカル公式PDFを参照し、O1の原形式（28組×8、各組2/4/2、倍率・特殊値・区分）を確認しました。[照合記録](jv_data4901_v02_review.md)に根拠ページを記録し、ドラフトのofficial_definitionへ参照定義を入れます。PCの実保存列との対応は自動承認しません。

2025年1月に履歴があるとは仮定しません。直近サンプルで保存中の日付を確認し、対象期間のsimulateで取得率を測ります。履歴0件・表なしなら全対象を母数に0%とし、プログラムは集計を完成できます。単位・観測時点が不明な設定を有効にしようとすると `mapping_unconfirmed` で停止します。

## B. 既存モデル・予測を使い、方針を先に固定

既存Version 0.1の実データ学習・未学習72レースの成功は利用者報告です。この作業では実レコードを取得していません。保存済み `test_ledger_id` の**単勝確率だけ**を再利用し、旧CLIの5分前・閾値10%・複勝判断は引き継ぎません。

```powershell
$taskModelId = Read-Host 'Version 0.1のモデルartifact_id'
$taskLedgerId = Read-Host '対応するVersion 0.1のtest_ledger_id'
$taskModel = Join-Path $taskV01 "models\$taskModelId"
$taskPredictions = Join-Path $taskV01 "predictions\$taskLedgerId"
$taskModelInfo = Get-Content -LiteralPath (Join-Path $taskModel 'model.json') -Raw | ConvertFrom-Json
$taskTestStart = $taskModelInfo.split.test.start
$taskTestEnd = $taskModelInfo.split.test.end

& $taskPython -B -m jra_racing_ai.win_sim_cli make-policy --model $taskModel `
  --test-start $taskTestStart --test-end $taskTestEnd
```

出力のartifact_idを使います。

```powershell
$taskPolicyId = Read-Host 'make-policyのartifact_id'
$taskPolicy = Join-Path $taskV02 "policies\$taskPolicyId\policy.json"
```

初期方針は**発走予定10分前JST、期待利益率5%以上、1レース最大1点、100円**。候補最大期待利益率の馬を選び、同値なら馬番の小さい馬とします。`make-policy --threshold 0.05 --minutes-before-start 10` で変更でき、simulateには上書き引数を持ちません。閾値を最終テスト結果から後付け最適化しません。

各馬のpを0～1・有限値で検査し、レース全体のraw合計を記録します。初期方針は **p / レース内p合計** の固定変換（race_normalize）。これは学習した校正器ではありません。ゼロ合計・不正pはレース保留。`identity` を選ぶ場合は合計が1から1e-6以内を要求し、黙って補正しません。元のp、変換後p、合計分布を記録します。同着も含む厳密な的中確率分布の校正は今後の改善です。

必要なら独立した検証期間で温度校正を行えます。

```powershell
& $taskPython -B -m jra_racing_ai.win_sim_cli make-policy --model $taskModel `
  --test-start $taskTestStart --test-end $taskTestEnd `
  --validation-predictions '<ml-v01またはml-v02内の検証予測フォルダー>' `
  --validation-payouts '<ml-v02内の確認済み検証結果JSON>'
```

検証データは**モデルの最終検証日（早期終了に使用）より後、最終テスト開始より前**の別期間が必須です。単独の確定的中対象を持つレースで温度0.75/1/1.25/2のレースLog Lossを比較し、選択結果・期間・件数を保存します。最終テスト情報は読みません。閾値は固定値で、収益による自動探索は未実装です。将来閾値を調整する場合は校正期間とも分けて決め、さらに後のテストへ固定適用します。既に見たVersion 0.1のテスト期間の再利用は、未使用の最終ホールドアウトとは呼ばず、記述的バックテストと記録します。

## 判断時点の発走予定・出走馬

```powershell
$taskInputs = Join-Path $taskV02 'inputs'
New-Item -ItemType Directory -Path $taskInputs -Force | Out-Null
```

本人が `schedules.json` をこのinputs内へ配置します。以下は**架空の形式例**です。実際の値をこのOneDriveプロジェクトに貼り付けないでください。

```json
{"format_version":2,"source":"local_reviewed","records":[{
  "race_key":["2030","0105","05","01","01","01"],
  "scheduled_start":"2030-01-05T15:00:00+09:00",
  "observed_at":"2030-01-05T09:00:00+09:00",
  "horses":["01","02","03"],"availability_verified":true
}]}
```

scheduled_startは当時把握していた予定時刻、observed_atはその時刻と名簿を実際に把握した時刻です。事後のRA最終行や作成日だけから自動生成しません。複数の予定変更がある場合、既存の判断時刻までに届いた変更だけを時刻順に反映します。判断後に判明した発走延期で購入判断を遡って移動しません。予測・名簿・オッズの馬番集合が一致しなければそのレースを除外します。

## オッズの入力経路

DB設定では `odds_sources` 内の確認できたソースだけをenabledにします。

| 設定 | 契約 |
| --- | --- |
| table | jvd_o1 / apd_sokuho_o1のみ。public固定 |
| layout | per_horse（馬ごと）、packed（固定長の繰返し） |
| columns | status / published / observed / horse / price / payload の実列名。使わない列はnull |
| time_formats | iso（TZ付き）、timestamp_jst（意味確認済みnaive日時）、timestamptz、yyyymmddhhmmss、mmddhhmm、hhmm_on_race_date |
| allowed_statuses | 実対応を確認した1（中間）/2（前日売最終）のみ。3/4/5（最終/確定）は利用不可 |
| price_encoding | digitsまたはdecimal、digitsのwidth、確認済みのnumerator/denominator、確認済みの欠測トークン |
| packed | slots / stride / horse_offset / horse_width（2） / price_offset / price_width / empty_horse_codes。offsetは0始まり |
| confirmed | official_format / pc_keiba_mapping / observed_time_semantics / odds_unit / revision_policy を本人が根拠確認 |
| official_reference / local_mapping_reference | 確認した仕様・取込定義のローカル参照。単なるファイル存在では確認済みにしない |

価格は**元文字列の桁・文字を検証した後**、値×numerator÷denominatorで倍率へ変換します。JV原形式の単勝は4文字・小数1桁ですが、PC保存値に同じ1/10を適用するかは別確認です。fixed charをtextへcastして空白や先頭ゼロを失う読込は行いません。packed内の空欄は確認済み設定だけを処理し、桁補完しません。年跨ぎのmmddhhmmを前年と推測せず、該当ケースはより完全な日時列を確認します。原形式の9999（999.9倍以上）は正確な価格にせずodds_cappedで保留します。

observedは実際の観測・利用可能時刻であることが必須です。発表時刻を取込時刻や作成日で代用しません。published列がない場合は、確認済み実観測時刻を保守的に両方へ使います。各スナップショットの published / observed が共に判断時刻以前のものから最新を選びます。最新が不完全・未知区分・不正値なら古い正常値へ戻りません。異なる時点を混ぜて全頭を揃えません。

本人が既に保存した観測済みオッズなら、DBの代わりに `--odds inputs/odds.json` を使えます。

```json
{"format_version":2,"source":"local_reviewed","records":[{
  "race_key":["2030","0105","05","01","01","01"],
  "published_at":"2030-01-05T14:48:00+09:00","observed_at":"2030-01-05T14:49:00+09:00",
  "odds":[{"horse_number":"01","value":"1.8"},{"horse_number":"02","value":"4.0"},{"horse_number":"03","value":"7.0"}],
  "availability_verified":true,"unit_verified":true
}]}
```

**最終オッズをこの形式に詰め替えて、過去に観測した価格として扱ってはいけません。** 履歴がなければ未取得として集計します。各レースの不正入力はそのレースを保留し、他の正常レースは進みます。レースキー自体を特定できない不正ファイルや設定は固定コードで停止します。

## 確定公式払戻の入力

未確認のHR b列を単に円へ変換しません。最初は本人が照合済みの公式公表額を `payouts.json` に配置できます。

```json
{"format_version":2,"source":"local_reviewed","records":[{
  "race_key":["2030","0105","05","01","01","01"],"status":"normal_final",
  "currency":"JPY","base_stake_yen":100,"reviewed":true,
  "evidence_basis":"official_published_payout","complete_winner_set":true,
  "payouts":[{"horse_number":"03","yen_per_100":820}]
}]}
```

同着は全対象とそれぞれの公式金額を列挙します。JRAプラス10も公式額をそのまま使います。通常の最終版・単勝・全的中対象・100円あたりJPYを本人が確認した場合だけ通常精算します。確認できない状態をreviewed=trueにしないでください。statusはnormal_final/refund/special/canceled/no_sale/unestablished/pending。通常以外は保留し、0円負けや自動返還にしません。

DBのHRから読む場合は `payout_source` を追加します。**確認根拠・対象年/場・正式意味・単位が揃う正常な単勝の範囲だけ**を有効にします。架空の形を示します（数値は実単位の確認結果ではありません）。

```json
{"table":"jvd_hr","reviewed_normal_conditions":true,
 "currency":"JPY","base_stake_yen":100,"final_category":"2",
 "revision_policy":"single_reviewed_final","semantics_reviewed":true,
 "official_reference":"確認済み資料のローカル参照",
 "local_reconciliation_reference":"当該範囲での本人照合の記録",
 "years":[2030],"venues":["05"],"numerator":1,"denominator":1,
 "columns":{"status":"確認した区分列","unestablished":"確認した単勝不成立列",
            "special":"確認した単勝特払列","refund":"確認した単勝返還列"}}
```

実列はASCII識別子です。この日本語説明値は実行用ではありません。払戻列は既知の `haraimodoshi_tansho_1a..3c` を既存形式分類器で検証します。全枠・通常対象・9桁・空白等の確認後に、承認範囲の換算だけを行います。不成立/特払/返還は0、区分は確認済み2、キー当たり1版を要求します。複数版・訂正/削除・00・不正/空の全対象・未確認範囲はそのレースだけ保留です。HR区分2という数値だけで訂正削除対応済みとは判断しません。Phase 1のdecode_hr_payoutsゲートや複勝決済は変更しません。

## D. 本人による実データシミュレーション

```powershell
# DBから判断時点オッズを読む場合。設定はinspectionで得たドラフトを本人が確認して別名保存。
$taskDbConfig = Join-Path $taskInputs 'db-config.reviewed.json'
& $taskPython -B -m jra_racing_ai.win_sim_cli simulate `
  --policy $taskPolicy --predictions $taskPredictions `
  --schedules (Join-Path $taskInputs 'schedules.json') `
  --db-config $taskDbConfig --execute-local `
  --payouts (Join-Path $taskInputs 'payouts.json') `
  --max-races 200 --max-rows 50000 --max-seconds 120

# 本人保存の時系列オッズファイルを使う場合（DB接続なし）。
& $taskPython -B -m jra_racing_ai.win_sim_cli simulate `
  --policy $taskPolicy --predictions $taskPredictions `
  --schedules (Join-Path $taskInputs 'schedules.json') `
  --odds (Join-Path $taskInputs 'odds.json') `
  --payouts (Join-Path $taskInputs 'payouts.json')
```

payout_sourceを確認してDBからHRも読む場合は `--payouts` を省略します。払戻未取得でも購入判断と件数は保存し、仮想購入を未精算にします。時系列オッズ未取得・予定時刻未確認は理由付き見送り。入力全対象レースを取得率の母数に残します。予測期間がpolicyと異なる場合、DB接続前に止めます。

DBは専用読取ユーザー、READ ONLY / REPEATABLE READ、SELECTのみ。実行は本人が明示した場合だけです。既定1SQL30秒、取得1000行ずつ、全ソース合計50000行、絶対上限200000行。時間はクエリ/バッチ間でも検査し、SQL自体には接続時タイムアウトを設定します。超過時は部分結果から利益を確定しません。`--max-races` は切り捨て件数ではなく上限で、超える入力は停止します。

成果物は `simulations/<ID>/ledger.json`、`summary.json`、`report.md`、使用policy、ハッシュmanifestです。毎回新規フォルダーへ排他的に保存し、read-only属性を付けます。個別キー・馬番・価格・SQL行・認証情報は標準出力に出しません。

## 月次集計と比較の読み方

- 全対象、オッズ取得数/取得率、シミュレーション可能数、購入/見送り、的中/的中率、投資、払戻、純利益、回収率、未精算、期待利益率分布を全体・月別に保存します。
- 判断の期待利益率はp×表示オッズ−1。精算は公式確定払戻−100円。**10分前オッズで払戻が保証されるわけではなく、実購入・約定実績ではありません。**
- 1件でも未精算のある全体/月の総払戻・純利益・回収率はnull。確認済み部分はsettled_subsetとして別掲します。確認済み通常レースを一律停止しません。
- comparisons.all_common_eligible_racesは、同じ判断時点でオッズが揃い、全候補を同じ通常払戻条件で精算できる共通集合。両戦略を同じレース数で比較します。
- comparisons.ai_purchased_common_racesは、そのうちAIが実際に仮想購入したレースだけの比較。1番人気の同率は馬番の小さい方へ100円を固定ルールで選びます。
- 1番人気の回収率は実際の公式金額から計算し、80%などの固定値にしません。購入0件の回収率・的中率はnullです。
- monthly_stabilityは精算が完了した購入月の平均・標準偏差・最悪月・利益月数。未精算月は除外件数を隠さず月別表示します。既知部分だけを全体の実績利益と呼びません。

## 履歴期間が合わない場合・Version 0.3

利用可能なオッズ期間が後年なら、Version 0.1のexport/trainの開始終了・train-end/valid-end/test-endを時系列のまま変更できます。[Version 0.1手順](version_0_1.md)を使い、その後の対象CSVに対して単勝だけの新規予測を作れます。

```powershell
& $taskPython -B -m jra_racing_ai.win_sim_cli predict-win --model $taskModel `
  --entries '<ml-v01内の新しい対象期間runners.csv>' --history '<ml-v01内の過去runners.csv>'
```

モデルの最終検証日以前は予測評価不可です。新しい予測・評価期間に合わせpolicyを新規固定します。Version 0.1の複勝モデルはロード互換性を確認するだけで、複勝推論・購入シミュレーションへ流用しません。

指定のローカル公式PDFは読み取り専用で参照済みです。外部通信はしていません。公式のO1/HR原形式の定義と、**未確認**のPC実列・倍率・観測時点・保存期間、予定時刻/名簿履歴、HR金額単位/版管理を分けています。実データの収益結果は未作成で、本人のローカル確認待ちです。A～CとDの実行経路は架空/モックで検証済みです。

Version 0.3は、確認できた実保存形式の設定確定、予定時刻/名簿の履歴連携、保存期間に合う複数月の新しいホールドアウト、独立期間での確率校正・閾値選定、取消/返還/訂正の公式精算規則、月別安定性と資金上限の検証を優先します。現在のRA/SE最終スナップショットには事後訂正が含まれ得るため、特徴量の完全なpoint-in-time再現も改善対象です。
