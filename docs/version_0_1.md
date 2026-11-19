# Version 0.1：ローカル学習・予測

Phase 1のCLI・HR金額ゲートを維持し、`jra_racing_ai.ml_cli` を独立して追加しました。着順に基づく確率モデルはHR金額未確認でも学習できます。CPUのLightGBMを使い、GPU設定・クラウド・購入機能はありません。

## 最初の実データ学習

PostgreSQLを従来の手順で起動し、既存の `jra_audit_reader` を使用します。パスワードは本人が端末の非表示プロンプトに入力します。チャット・ファイル・コマンド引数には書きません。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGDATABASE = 'pckeiba'
$env:PGUSER = 'jra_audit_reader'
& $taskPython -B -m jra_racing_ai.ml_cli first-run --execute-local
```

初回は **2025年1月・国内10場・RA/SE区分7・最大10,000行** に限定します。ローカルの列構造から必要な4項目（RA区分、SE区分、馬番、確定着順）と、存在すれば過去走追跡用の馬識別列の候補を表示します。列名だけから意味を確定しません。本人がPC上の列定義で対応を確認できた場合だけ `CONFIRM` と入力します。候補が合わない場合は停止し、下記の手動マッピングを使います。

対応確認後、読み取り専用でCSVを作り、DB接続を閉じてから学習します。日付単位で約70%/15%/15%に分割し、最後の期間の各馬の単勝・複勝圏内確率を保存します。**最終テストの結果を学習や早期終了に使いません。** この小規模学習は動作確認用で、収益性を判断する期間としては不十分です。

実行結果は集計JSONだけです。`artifact_id` はモデル、`dataset_id` は学習CSV、`mapping_id` は対応表、`test_ledger_id` は未学習期間の予測履歴のフォルダーIDです。個別行の表示はありません。

補助スクリプトも同じ処理を実行します。実行ポリシーの変更は行いません。

```powershell
.\scripts\run_v01_first_training.ps1 -Start '2025-01-01' -End '2025-01-31' -MaxRows 10000
```

## ローカル成果物

保存ルートを次に固定しています。

`C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v01`

| フォルダー | 内容 |
| --- | --- |
| `schema/<ID>` | RA/SE列構造、候補対応表、本人確認済み対応表 |
| `datasets/<ID>` | `runners.csv`、SHA-256と集計・対応表を持つ `dataset.json` |
| `models/<ID>` | `win.lgb.txt`、`place.lgb.txt`、`model.json`、`paper_evaluation.json` |
| `predictions/<ID>` | 各馬の確率・判断の `predictions.json` と `manifest.json` |
| `evaluations/<ID>` | 予測履歴とは別に保存する事後評価 |

実データ・モデル・個別予測はこのルート以外へ書きません。実行時にOneDrive、同期の環境変数、クラウド属性、ネットワークドライブ、junction/symlink、ハードリンクを検査します。OSや別ソフトの任意のバックアップ設定まで自動検出する機能ではないため、利用者側でも同期対象外であることを確認します。プロジェクト内の `.gitignore` だけに保護を依存しません。

## 特徴量とラベル

初回最小モデルは競馬場・馬番と、識別列を確認できた場合の過去走を使います。追加可能な任意項目は距離、コース種別コード、馬場状態コード、事前出走頭数、枠、馬齢、性別、負担重量、予定発走時刻です。存在と意味が未確認なら欠損にし、追加確認のために初回学習全体を止めません。

過去走特徴は過去出走数、前走からの日数、前走着順・タイム、過去平均着順、同距離・同コース平均着順です。現在レースの着順・タイムは特徴量に入りません。馬ID・レースID・最終オッズ・HRも学習行列に入りません。

過去結果は **対象日より前、かつ結果日＋`history_lag_days`以下** のものだけを使用します。既定7日、最小1日です。同日結果は使いません。1日や7日という遅延は保守的な運用仮定で、実際の公表時刻の証明ではありません。最新状態のRA/SEは事後の条件訂正や取消を含み得るため、過去時点を完全に再現した評価とは区別して `retrospective_snapshot_not_verified_point_in_time` を記録します。

カテゴリ変数はpandas `category`。カテゴリ辞書は学習期間だけで作り、モデルと共に保存します。予測で新カテゴリは `__UNKNOWN__`、欠損は `__MISSING__` に分け、学習時と同じカテゴリ順を使います。数値欠損はLightGBMの欠損として渡します。欠損件数と理由分類をモデルメタデータへ保存します。

ラベルは暫定的な **rank proxy** です。単勝は確認済み確定着順1、複勝は観測されたSE行数が8頭以上なら3着以内、5～7頭なら2着以内、4頭以下は不明です。着順0・欠損・不正は負例にせずラベル不明として除外します。SE行数はラベル判定にのみ使い、特徴量の事前頭数と混同しません。取消・除外・競走中止・同着・2着払い等を完全に扱う公式馬券ラベルではありません。複勝モデルの最初の精度比較を進めるための暫定定義であり、HRの確定払戻対象による正解照合は次の改善です。

## 構造確認・特徴量追加・学習期間拡大

```powershell
$taskMlRoot = Join-Path $env:LOCALAPPDATA 'JraRacingAI\ml-v01'
& $taskPython -B -m jra_racing_ai.ml_cli inspect --execute-local
```

`schema/<ID>/schema.json` と `mapping.draft.json` をPC内で確認します。ドラフトの `reviewed` はfalseです。必要な列の意味を照合してから対応表を編集し、trueにします。実行時の列名はパラメーターで自由なSQLへ挿入せず、許可されたRA/SEの識別子として検証します。

`columns` は実列名またはnullです。任意の事前特徴量は、入手時点を確認できた論理名だけを `pre_race_fields` に追加します。例：距離を確認したら `"distance"`。負担重量・走破タイムは、保存値をkg・秒にする乗数を `scales` に必ず指定します。独自の時分秒表現を乗数だけで変換できない場合、その列はnullのままにします。芝/ダート別の馬場状態を1列から無条件に推定しません。`going` は正しい列を1つ選べる場合だけ使います。

```powershell
# IDは直前の集計JSONで得た値を指定。参照ファイルの中身をチャットへ貼る必要はありません。
$taskMapId = '<mapping_id>'
$taskMapping = Join-Path $taskMlRoot "schema\$taskMapId\mapping.reviewed.json"
& $taskPython -B -m jra_racing_ai.ml_cli export --execute-local --mapping $taskMapping `
  --start '2019-01-01' --end '2025-12-31' --max-rows 500000 --max-seconds 600

$taskDataId = '<exportのartifact_id>'
$taskDataset = Join-Path $taskMlRoot "datasets\$taskDataId\runners.csv"
& $taskPython -B -m jra_racing_ai.ml_cli train --dataset $taskDataset `
  --train-end '2023-12-31' --valid-end '2024-12-31' --test-end '2025-12-31'
```

利用可能な年はローカルの `schema.json` の年別集計で確認します。実在する期間に調整してください。上記期間は本人が選ぶ例で、現時点で収録を確認したという意味ではありません。分割日無指定ならデータの日付順70/15/15です。各期間が空、学習ラベルが片クラス、馬単位キー重複は固定コードで停止します。

SQLは固定SELECT、完全な6列キー結合、READ ONLY / REPEATABLE READ、同一PCの `127.0.0.1:5432/pckeiba` と専用ユーザーに限定します。既定の1SQLタイムアウト60秒、サーバーカーソル1000行ずつ、実行時間180秒、最大行数100000（絶対上限500000）。初回コマンドはさらに既定10000・上限50000行。超過時に途中までのデータで学習せず停止します。SQLタイムアウトとバッチ間の時間検査を併用し、CPU学習はDB接続を閉じてから行います。DB構造・ユーザー・インデックスは変更しません。

## 学習済みモデルで別期間を予測

学習の検証終了日より後のレースだけを受け付けます。検証期間は早期終了に使うため、未学習評価と混同しません。別期間を同じ確認済みマッピングでexportし、予測に渡します。

```powershell
$taskModelId = '<trainのartifact_id>'
$taskModel = Join-Path $taskMlRoot "models\$taskModelId"
$taskFutureId = '<別期間exportのartifact_id>'
$taskEntries = Join-Path $taskMlRoot "datasets\$taskFutureId\runners.csv"
& $taskPython -B -m jra_racing_ai.ml_cli predict --model $taskModel `
  --entries $taskEntries --history $taskDataset --mode backtest
```

`--history` は対象レースと重複しない過去データにします。未来の履歴行が含まれていても特徴量には使いません。矛盾した重複行を黙って優先しません。保存はLightGBMのテキスト形式で、pickleをロードしません。カテゴリ定義とSHA-256を検証して再読込します。

## 購入時点オッズと仮想購入

既定方針は100円・期待収益率0.10以上・1レース最大2件、期待値降順です。trainの `--min-ev` / `--stake-yen` / `--max-bets-per-race` を学習前に指定でき、モデルと共に固定されます。学習後のpredictには閾値の上書き引数がありません。結果を見た最終テストの後付け調整は禁止です。v0.1は自動閾値探索を行いません。将来の方針比較は検証期間で行い、その後新しい最終期間に固定方針を適用します。

購入オッズは `--odds <local JSON>` で指定できます。スキーマは以下です（値は架空）。ファイルは `ml-v01/inputs/odds.json` 等の同期対象外へ配置します。

```json
{
  "format_version": 1,
  "source": "pre_race_snapshot",
  "records": [{
    "kaisai_nen": "2030", "kaisai_tsukihi": "0105", "keibajo_code": "05",
    "kaisai_kai": "01", "kaisai_nichime": "01", "race_bango": "01",
    "horse_number": "01", "wager": "win",
    "captured_at": "2030-01-05T14:40:00+09:00",
    "decision_at": "2030-01-05T14:45:00+09:00",
    "price_kind": "scalar", "price": 4.2
  }]
}
```

`wager` はwin/place、`price_kind` はscalar/rangeです。rangeのpriceは上下限2個の配列で、購入を保留します。取得時刻≦判断時刻＜予定発走時刻を検証します。最終オッズや取得時点不明の価格を、このファイルで購入時点の価格と偽って使わないでください。O1の履歴からの抽出・時点検証は今回未実装で、自動採用はありません。

期待収益率は `p * price - 1`。単勝・複勝別、月別の予測数、購入数、投資額、暫定ラベル的中数、期待値分布を集計します。**公式払戻と特殊ケースが未確定なので、払戻金・回収率・ROI・純損益はnull**。未算出を0円の不的中と混同しません。オッズなしは全件見送りで、確率評価は続きます。

## 開催前の前向きシミュレーション

本人が事前に用意した正規化済みCSVを `ml-v01/inputs/entries.csv` に置きます。ヘッダー順は `ml.features.DATA_FIELDS`（export CSVと同じ）です。全6列キーと2桁馬番が必須、任意列は空欄可。現在の `finish_position` / `finish_time_seconds` は空欄にします。事前に確認した項目だけを埋めます。未公開の結果を追加してはいけません。発走時刻の意味を確認できた場合だけ `start_time` をHHMMで指定します。時刻不明なら当日0時を締切とし、当日予測を許可しません。

```powershell
& $taskPython -B -m jra_racing_ai.ml_cli predict --model $taskModel `
  --entries (Join-Path $taskMlRoot 'inputs\entries.csv') --history $taskDataset --mode forward
```

前向き予測はシステム現在時刻を使い、過去への時刻指定を受け付けません。推論後と保存時にも締切を確認します。毎回新IDへ排他的に保存し、元の予測を更新するAPIを持ちません。完了ファイルをread-only属性にし、SHA-256で変更を検出します。本人が両ファイルを書き換えることまで防止するWORM保存・第三者の時刻証明ではありません。

開催後の結果照合は別成果物にします。

```powershell
$taskLedgerId = '<predictのartifact_id>'
$taskOutcomeId = '<対象期間の結果exportのartifact_id>'
& $taskPython -B -m jra_racing_ai.ml_cli evaluate `
  --ledger (Join-Path $taskMlRoot "predictions\$taskLedgerId") `
  --outcomes (Join-Path $taskMlRoot "datasets\$taskOutcomeId\runners.csv")
```

履歴にはbacktest/forwardを記録します。事後評価は確率指標と暫定的中数を計算し、元の予測ファイルを変更しません。

## テスト・環境再現・次の改善

```powershell
& $taskPython -B tests\run_tests.py
& $taskPython -B -m jra_racing_ai.ml_cli demo
```

テストは架空データとモックのみ。socket・実DBドライバーを禁止します。ML依存がなければ依存が必要なテストクラスだけskipし、Phase 1テストは引き続き標準ライブラリで動きます。Windows通常端末では一時フォルダーへ架空モデルを保存して再読込します。

環境はPython3.12.10 / Windows、固定した全関連依存は `requirements-ml-lock.txt` です。ソースからライブラリを導入する機能はありません。同環境を作り直す場合のみ、承認されたパッケージ取得として次を本人が実行できます。

```powershell
& $taskPython -m pip --isolated install --index-url https://pypi.org/simple `
  --only-binary=:all: -r requirements-ml-lock.txt
```

次は初回学習の集計結果をPC上で確認し、期間を年単位に拡大します。その後、確認済み距離・馬齢等の追加、取消/DNF/同着を考慮したラベル、HRによる公式的中対象、O1購入時点履歴、前向き出走馬入力の自動化、検証期間だけでの校正・投資方針比較の順で改善します。金額ゲートを解除せず、先に確率モデルの改善を進められます。
