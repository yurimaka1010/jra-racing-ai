# Version 0.2：登録済み時系列オッズで最初の実データ実行へ

2026-10-16。監査機能の追加はせず、既存のexport / predict-win / make-policy / simulateを使います。

**最新結果：本人実行で2026-10-04の24レース・362頭の予測が成功しました。** RA/SEはいずれも区分2、現在レースの結果は除外済みです。予測の再実行や区分監査は不要です。次は [発表時刻基準の1行コマンド](v02_publication_run.md)で登録済みオッズと対応付けます。既存572件＋追加25件＝597件の架空・モックテストが成功しました。以下は再利用手順と修正経緯です。

## 今回確認できた範囲

本人実行のinspection `53e856cbd011435a89a4a3eb9a7f664d` は成功。

| 対象 | 確認結果 |
| --- | --- |
| jvd_o1 | 22列。直近100行の日付は2026-09-19～09-27 |
| apd_sokuho_o1 | 22列。直近100行の日付は2026-10-04 |
| data_kubun | varchar(1) |
| odds_tansho | varchar(224)。公式の28組×8文字に対応する候補 |
| happyo_tsukihi_jifun | varchar(8)。発表月日時分の候補 |
| data_sakusei_nengappi | varchar(8)。これだけではPCの実取得時刻にならない |
| PCの実取得時刻 | 上記22列には専用列がない。別の取得ログは未確認 |
| 保存済みモデル | 学習/早期終了の最終日2025-01-19、旧テスト2025-01-25～01-26 |

直近100行は全履歴の期間ではありません。2025年の履歴がないと断定しません。存在を確認できた2026-10-04について、本人実行で24レース・362頭の予測まで完了しました。

## 1. 保存済みモデルで対象日の単勝確率を作る

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
.\scripts\run_v02_local_predictions.ps1 -RaceDate '2026-10-04'
```

スクリプトは本人PC用です。実行ポリシーの変更は行いません。PostgreSQLが既存の手順で起動済みである必要があります。

- 既存の本人確認済みRA/SEマッピングと保存モデルを再利用します。
- EV5%以上・発走10分前・最大1点100円のpolicyを先に新規保存します。
- 本人の非表示パスワード入力後、予測専用の `export-entries` で当日のRA/SE区分1～7の出走情報を読取専用で取得します。最大2000行、全体60秒、1SQL15秒。削除0・中止9・地方/海外A/Bは含めません。現在レースの着順・走破タイムはSQLでNULLを選び、保存時にも空欄を強制します。切り捨てた部分データで続行しません。
- DB接続終了後、既存LightGBMの単勝モデルで予測します。複勝推論・再学習・馬券購入はしません。
- 元の学習データを記録済みSHA-256で特定して履歴に再利用します。**直近の過去走を追加取得していない初回動作確認**です。
- データセットはOneDrive外ml-v01、予測とpolicyはOneDrive外ml-v02に保存します。実レコードを画面・チャットに出しません。

成功すると `status / dataset_id / predictions_id / policy_id / race_count / runner_count / source_status_counts / current_results_excluded` 等の集計JSONが出ます。区分別件数を同じ取得から計算するため、事前に別の区分監査は不要です。予測入力は `purpose=prediction_only` とし、学習・評価ラベル用の読込では拒否します。学習用の既存 `export` は区分7限定を維持します。認証失敗や対象データ不足は固定理由コードで停止します。過去走更新やモデル改善より、まず新しい日付へ推論できることを確認します。

## 2. オッズの時点を明示する

既存の厳密なsimulateは実観測時刻の確認を要求します。発表時刻とPC実取得時刻を同一の列で埋めて確認済みにはしません。新しいpublication-runは別policyで発表時刻を使い、PC実取得時刻はnullのまま維持します。

本人から、実取得時刻・購入可能時刻・通信遅延を未検証と明記した**発表時刻基準の暫定的な事後シミュレーション**を先行する承認を受領済みです。発走予定10分前より後に発表されたオッズ・確定結果・事後情報は判断に使いません。新たな承認待ちではありません。実行手順は [v02_publication_run.md](v02_publication_run.md)に移しました。以下の手入力simulateは実観測根拠がある場合の既存経路です。

実観測時刻を確認できる場合は既存の正規化済み `odds.json` を使えます。発表月日時分と取得時刻はそれぞれ保存します。予定時刻・出走馬集合も判断時点の根拠が必要です。RA最終行の時刻や最終SEだけから、当時の予定・名簿が証明されたとは扱いません。

## 3. 既存simulateへ渡す

以下は入力確認後のコマンドです。IDは手順1の集計JSONから使います。実行可能な入力をまだ確認していないため、この時点で購入・損益の実績が得られたという意味ではありません。

```powershell
$taskPython = 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$taskV02 = 'C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v02'
$taskPredictionId = Read-Host '手順1のpredictions_id'
$taskPolicyId = Read-Host '手順1のpolicy_id'
$taskInputs = Join-Path $taskV02 'inputs'

& $taskPython -B -m jra_racing_ai.win_sim_cli simulate `
  --policy (Join-Path $taskV02 "policies\$taskPolicyId\policy.json") `
  --predictions (Join-Path $taskV02 "predictions\$taskPredictionId") `
  --schedules (Join-Path $taskInputs 'schedules.json') `
  --odds (Join-Path $taskInputs 'odds.json') `
  --payouts (Join-Path $taskInputs 'payouts.json') `
  --max-races 50 --max-rows 10000 --max-seconds 60
```

実列・倍率・観測時刻を確認済みのDB設定がある場合は `--odds` を `--db-config <ローカルの確認済み設定> --execute-local` へ置き換えます。未確認のドラフトをそのまま有効化しません。

公式100円払戻を確認済みの `payouts.json` がまだなければ、`--payouts` 行を省略して購入判断を未精算で保存できます。最終オッズで払戻を推定しません。AIと同時点の1番人気の選択は保存され、収益比較は共通の正常精算条件を満たしたレースだけです。

入力形式・本人確認済みHRを使う経路・月次レポートの読み方は [既存Version 0.2手順](version_0_2.md)を使います。ファイルの実値や個別明細をチャットへ貼る必要はありません。

## 今回の検証

- 既存555件すべて成功（3.106秒、skipなし）。実DBドライバーとsocket接続はテストで禁止。
- 追加PowerShellは構文エラー0。外部プログラムを架空の関数に置換し、引数引渡し・固定エラーの保持・不正出力の停止の3点を確認。
- Codex自身によるDB接続・取得・実シミュレーションは実行していません。追加インストール・外部通信はありません。

## database_errorの修正（2026-10-16）

当初の本人実行は実際にexportまで到達していました。回答待ちで未実行だったのではありません。その時点では未完了のデータセットフォルダーがあり、予測フォルダーは未作成でした。個別データの内容は読んでいません。以下の修正後、現在は予測完了しています。

モックで確定した不具合は、接続中の `MLFailure` が `read_only_session` の汎用例外処理に捕捉され、元の固定理由まで `database_error` になっていたことです。MLの固定コード例外を安全な例外型として扱い、`insufficient_history` / `invalid_schema` / `duplicate_entry` 等を保持するよう修正しました。未知例外・実際のクエリ失敗の本文は引き続き出しません。

再実行するコマンドは手順1と同じです。失敗時には `stage`（preflight / make_policy / export / predict）も出ます。実SQLエラーには許可した固定 `query_id` と5文字のSQLSTATEだけを付記し、SQL・パラメーター・行・エラー本文は出しません。本人の再実行結果は `insufficient_history / stage=export` でした。指定日のRA/SE区分7の完全キー結合が0行だったことまで確認済みです。過去走の不足やLightGBM学習の失敗ではありません。O1だけの登録成功でRA/SEまで揃ったとは判断しません。

修正後は既存555件＋回帰7件＝562件が成功（3.148秒、skipなし）。PowerShellの構文確認と固定診断コードの受け渡しもモックで成功しました。

### 当日の0行を一度だけ切り分ける

プロジェクトフォルダー内のPowerShellで、以下の1行を実行します。Pythonを直接起動するため、PowerShellの実行ポリシー変更は不要です。

```powershell
& 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe' -B '.\scripts\check_v02_source.py' --race-date '2026-10-04'
```

非表示の本人認証後、既存の読取専用接続から当日・国内10場だけを対象にSELECTを1回実行（上限10秒）します。出力はRA全行、RA区分7、SE全行、SE区分7、完全キーで結合した区分7の馬行数の5件数のみ。実レコード・馬番・金額を取得・保存・表示しません。通常の接続時の読取専用・権限確認も維持します。

`ra_se_missing` / `ra_missing` / `se_missing` なら該当データが未収録、`final_category_missing` なら区分7が不足、`race_keys_unmatched` なら両表に区分7はあっても完全キーが不一致です。件数に基づかず区分条件やキーを緩めません。収録が不足する場合のPC-KEIBAでの通常の追加取得は本人が行い、Codexや確認コマンドはDBを更新しません。

この単発の原因確認を含む架空・モックテストは564件すべて成功（既存562＋追加2、3.145秒、skipなし）。

## 区分7がない日の予測を止めていた原因と修正

本人確認の2026-10-04はRA24行・SE362行が存在し、両方の区分7が0行でした。従来のスクリプトは学習用 `export` を予測入力の作成にも利用し、区分7がないだけで出走情報まで0行にしていました。

公式PDF11/12ページのRA/SE定義では、1/2は出馬情報、3/4は一部着順の速報、5/6は全馬着順の速報、7は月曜成績です。今回の推論は現在レースの着順・タイムを必要としないため、上記の予測専用 `export-entries` へ切り替えました。速報を確定ラベルに昇格させる処理ではありません。

コード修正後の再実行コマンドは手順1と同じ1行です。`check_v02_source.py` は旧来の区分7条件を調べるための単発確認で、推論が成功した後に再実行する必要はありません。

予測専用CSVは `source=local_ra_se_prediction_snapshot`、`current_results_excluded=true` と記録し、既存 `train` と履歴データ読込は拒否します。`predict-win` だけが明示的に受け付け、CSVに現在着順等が再混入していればSHA-256が一致していても停止します。予測manifestには元CSVのハッシュと入力種別を記録します。学習条件・複勝モデル・払戻金額ゲートは維持します。

元の学習データだけを履歴に使う初回動作確認であり、最新過去走は未更新です。RA/SEは保存スナップショットで、全列の過去時点の利用可能性を独立に証明したものではありません。確率を出せたことを収益性・前向き運用の検証成功とは扱いません。

この修正時の検証は既存564件＋追加8件＝572件すべて成功（3.525秒、skipなし）。非確定の架空出走情報→現在結果を除いたCSV→保存済みLightGBM単勝モデル→架空オッズでの判断・精算・1番人気比較まで確認しました。その後の本人実行はRA/SE区分2、24レース・362頭で成功しています。
