# Phase 1.5.1 H1・HRのローカル整合性検証

文書日：2026-10-15（Asia/Tokyo）。実装と架空/モックテストだけを実施しました。Codexは実DB、実H1/HR、実参照ファイル、公式サイトへアクセスしていません。利用者が提示した個別レースの観測値・日付・名称・キーは本資料へ転記しません。

## 根拠と適用範囲

利用者提示の [JV-Data Ver.4.9.0.1](https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf) と今回の確認条件を根拠として、**HR区分2とH1区分5を、それぞれの月曜確定系候補**として検査します。異なるレコード種別の区分数値の一致は要求しません。公式原本はローカル未提供で、独立照合は未実施です。H1がHRと同じ収録範囲を持つとも認定しません。

確認するルールは次のとおりです。これは金額単位・特殊払戻・版の採用規則の確定ではありません。

| 条件 | H1必須モードでの動作 |
|---|---|
| レース識別 | 開催年/月日/場/回/日目/番号の6列を厳密な文字列で完全一致。trim・数値化・補完なし |
| 件数 | 各参照レースのHRとH1がそれぞれ1件。欠損・複数行は保留。inner joinや最新版選択で隠さない |
| データ区分 | HRは文字列2、H1は文字列5。それ以外やNULLは保留 |
| H1発売フラグ | 単勝・複勝の両方が厳密な文字列7。その他のコードの意味を推測しない |
| HR通常条件 | 単勝/複勝の不成立・特払4項目と、既知の返還2フラグがすべて厳密な文字列0 |
| 複勝着払キー | H1が文字列2または3。複勝参照ケースのplace_paid_positionsとも一致すること |
| 版 | 既存のHRレビューと、独立したH1レース別レビューが必要。区分だけで確認済みにしない |
| その他 | 既存のソース/列意味レビュー、返還bitmap、a/b/c形式、全対象集合、通常状態等のゲートも維持 |

H1/HRの発売・成立条件はレース共通の検査として、両券種をまとめて検査します。片方に未知フラグや複勝条件の矛盾があれば、そのレースは両券種とも比較候補にしません。返還ありのレースは旧モードで別の通常対象を比較できても、このH1必須モードの条件には入りません。

着払キー2と参照2が一致し、通常の数値対象/空白未使用枠だけなら金額仮説比較の候補にできます。**馬番00の意味解決は追加していません。** 00を含む場合は従来どおり `zero_horse_unconfirmed` で保留します。H1条件を通っても、全対象集合やHRの版等が未確認なら金額比較を行いません。

## 旧形式との互換性

既存の参照JSONはformat_version=1のままで変更不要です。新しい `--h1-config` を指定したときだけ **H1必須モード**になります。指定なしのPhase 1.5互換モード・既存API・既存348件のテストを維持します。旧モードの結果はH1確認済みを意味しません。Phase 1.5.1の検証には必ずこのオプションを付けます。

H1必須モードでは、参照JSONの旧 `flag_rules` をHR取得・通常判定に使用しません。H1設定のHR不成立/特払4列と既知の返還2列、H1発売フラグを使用します。旧 `flag_rules` は既存書式として検証しますが、空配列にもできます。発売なしを表す架空のHR列を追加する必要はありません。H1取得失敗や欠損時に旧モードへ戻る処理はありません。

`reconcile_h1_payouts()` は純粋なPython関数です。既存の3仮説比較・対象集合検査・形式検証を再利用し、NormalizedPayout生成・決済呼出・単位自動確定は行いません。`production_settlement_enabled=false` と `unit_confirmed=false` は固定です。

## H1設定ファイル

参照JSONと同じOneDrive外の固定入力ディレクトリに、`h1-config.json` を置きます。

```text
C:\Users\acbd5\AppData\Local\JraRacingAI\audit-reports\reconciliation-input\h1-config.json
```

[架空ひな形](../examples/h1-config.synthetic.json) は列名・キーが架空で、レビュー未確認です。実データを記入する前にAppData側へコピーし、プロジェクト側は編集しません。JSONのUTF-8、256KiB、重複キー、同期/リンク等の拒否は既存と共通です。

| 設定 | 内容 |
|---|---|
| format_version | 整数1 |
| h1_columns | status / tansho_sale / fukusho_sale / place_paid_positionsの**確認した実列名**4項目 |
| hr_columns | tansho_not_established / tansho_special_payment / fukusho_not_established / fukusho_special_paymentの**確認した実列名**4項目 |
| column_meanings_reviewed_locally | 上記8項目の意味・型を本人が仕様/PC-KEIBA資料で確認したらtrue。未確認をtrueにしない |
| h1_revision_reviews | race_key（既存と同じ6列）とrevision_reviewの配列。最大20レース |

各レビューは `unconfirmed` / `snapshot_reviewed_locally` / `preliminary` / `correction` / `deleted` です。通常比較には `snapshot_reviewed_locally` を要求します。欠けたレビューをHRからコピーして確認済みにする処理はありません。確認日時の最大や区分5だけで訂正・削除の採用を決めません。

設定のレースキーは参照JSONにあるキーだけを許可し、重複・矛盾は拒否します。キーの開催回/開催日目は、日付やレース名から推測せず、本人がローカルで完全な6列を確認してください。設定にない参照レースはH1版未確認で保留します。

## SQL・資源・出力

HR/H1は同じ既存のREAD ONLY / REPEATABLE READセッションで読みます。固定接続先・専用ロール・権限検査・SQL書込み拒否は維持します。各表を独立して、同じ参照キーをパラメーターにした限定SELECTで取得します。H1もONLYの通常heap、RLSなし、組み込み文字列型だけを許可し、指定幅を超える値は取得せず保留にします。票数・馬番配列等のH1実データは取得しません。

既定はHR40行、H1も40行（`--max-h1-rows`、最大200）。参照レース上限は既存どおり既定5・最大20です。行数超過は実行を中止し、部分成功レポートを保存しません。

H1必須モードは権限検査1＋両表それぞれのカタログ2/データ1で計7クエリです。接続5秒＋7×SQLタイムアウト＋余裕が実行予算内に入ることを入力確認時にも検査します。SQL5秒なら **`--max-run-seconds 45`** を指定します。旧モードの30秒を黙って延長せず、不足時は `h1_time_budget` で停止します。両表は同じ単調時計の期限を共有します。

通常の一致/不一致/照合不能に加え、集計JSON/Markdownへ `h1_validation` を追加します。要求レース、H1件数、文脈条件の整合/保留レース、区分別件数、固定理由コードだけを保存します。`compatible_races` はH1とHRの発売/成立文脈の条件を通った数であり、払戻形式・公式対象・金額単位・全体決済の承認ではありません。レース単位の文脈理由と券種ごとの保留理由は、他の不備が併存すれば異なることがあります。

主な固定理由は `missing_h1`、`multiple_h1_records`、`h1_category_mismatch`、`h1_sale_flag_unconfirmed`、`h1_place_key_unconfirmed`、`h1_place_terms_mismatch`、`h1_revision_unreviewed`、`h1_correction_unreviewed`、`h1_deletion_unreviewed` 等です。個別キー・馬番・払戻額・取得行・入力パス・秘密は出力しません。集計の実保存先も既存AppData配下だけです。

## 本人が行う最初の安全な操作

まず既存venvで架空テストを実行します。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
& $taskPython -B tests\run_tests.py
```

既存のローカル参照JSONを保ったまま、同期対象外の入力フォルダーへ新しい架空設定を上書きせずコピーします。フォルダーがない場合の作成は既存の[Phase 1.5手順](payout_reconciliation.md)を参照してください。

```powershell
$localInputDir = Join-Path $env:LOCALAPPDATA 'JraRacingAI\audit-reports\reconciliation-input'
$localH1Config = Join-Path $localInputDir 'h1-config.json'
if (Test-Path -LiteralPath $localH1Config) { throw 'H1設定が既に存在します。上書きを中止しました。' }
Copy-Item -LiteralPath '.\examples\h1-config.synthetic.json' -Destination $localH1Config
```

AppData側のコピーだけで実列名と、既存参照JSONに対応する完全なレースキーを設定します。未確認のレビューはfalse/unconfirmedのままにします。次の **入力だけの検査** を最初に実行します。DB・ドライバー・パスワード・出力ファイルを使いません。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$localReferenceFile = Join-Path $localInputDir 'reference.json'
& $taskPython -B -m jra_racing_ai.cli reconcile-payouts --check-input-only --confirm-local-unsynced --reference $localReferenceFile --h1-config $localH1Config --max-races 1 --max-hr-rows 5 --max-h1-rows 5 --statement-timeout-ms 5000 --max-run-seconds 45
```

最初は1レース分だけの参照を本人がOneDrive外で用意してください。既存ファイルに複数レースがある場合、上限だけで自動的に1件へ切りません。別名の少数参照ファイルをAppData側で用意し、コマンドの参照先をそのファイルに変更します。JSON編集前後の実データをチャットへ送る必要はありません。

意味・列型・HR/H1それぞれのスナップショットを本人が確認した後だけ、同じ条件の `--check-input-only` を `--execute-local` に置き換えて実行します。接続設定・非表示パスワードはPhase 1.5と同じです。CodexはこのDB実行を行いません。

H1が存在しない年/レースや版の不明なケースを除外して全体の金額解釈を確定しません。未確認ケースは保留のまま、規則の追加確認と架空テストへ戻します。本番決済・損益計算の開放は別途承認が必要です。
