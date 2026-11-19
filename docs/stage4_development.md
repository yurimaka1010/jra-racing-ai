# Stage 4：有限の開発実験

Stage 1〜3のコード・定数・ローダー・保存物・最終評価計画を変更しない専用実装です。入力期間は2020-01-01〜2025-09-30。評価は2023・2024・2025年の各1〜9月、既参照の開発期間だけです。2026年用の実行入口はありません。

## 実行順序

本人の既存Windows仮想環境を使用します。パッケージ追加、外部通信、DB書込み、タスク登録、実購入は行いません。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$stage4Python = 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe'

# 新しい実験開始時だけ。新runのIDを表示します。
& $stage4Python -B '.\scripts\stage4_local.py' --stage init

# 承認済み今回runを再開する場合はこのIDを使用します。
$stage4Run = '2f26b6573d2249d38ad96b4456f89c0f'
& $stage4Python -B '.\scripts\stage4_local.py' --stage reproduce --run-id $stage4Run
& $stage4Python -B '.\scripts\stage4_local.py' --stage oof --run-id $stage4Run
& $stage4Python -B '.\scripts\stage4_local.py' --stage direct --run-id $stage4Run

# 既存DPAPI資格情報を同じWindowsユーザーで読む、唯一のDB処理。
# 完了キャッシュがあればDBへ接続せず再利用します。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -NonInteractive -ExecutionPolicy Bypass -File '.\scripts\stage4_local.ps1' -Stage quotes60 -RunId $stage4Run

& $stage4Python -B '.\scripts\stage4_local.py' --stage h60 --run-id $stage4Run
& $stage4Python -B '.\scripts\stage4_local.py' --stage accuracy --run-id $stage4Run
& $stage4Python -B '.\scripts\stage4_local.py' --stage finance --run-id $stage4Run

# 実DB・ネットワーク禁止の架空回帰テスト。ログは架空テストのみ。
$env:TEMP = 'C:\Users\acbd5\AppData\Local\JraRacingAI\synthetic-tests'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
& $stage4Python -B '.\tests\run_tests.py' *> '.\reports\stage4_regression_tests.txt'
& $stage4Python -B '.\scripts\stage4_local.py' --stage freeze --run-id $stage4Run
```

今回の完了済み結果を確認する場合、最後のfreezeコマンドだけで十分です。ハッシュを検証して集計結果を返し、学習・オッズ取得・精算をやり直しません。新しいrun IDへの再実験は今回の承認済み上限の追加枠ではありません。

## 固定した実験

- H OOFの新規LightGBMは2021・2022年用の2本だけ。2023〜2025年用HとABCD OOFは旧保存物を使用します。
- Hcal/Hstack × ridge 0.01/0.1 ×3年＝12最適化。中心化log確率の尺度とJSD境界は統合学習OOFだけから計算します。正則化は前年9/8〜10/31で選択し、再学習しません。
- H60は市場8列＋T60→T30対数市場確率変化・実公表時刻差・欠損フラグだけ。3年用各1本、T=1、300ラウンド上限・早期終了20。
- 唯一の新しい購入ルールは学習OOFのJSD90百分位以上のレース限定ABCD。EV5%以上・1点100円・1レース最大1頭を維持し、最低勝率などの追加フィルターは使いません。

T60は既存apd_sokuho_o1の対象キーだけを32レース単位、最大256行、SQL上限60秒で読みます。分終端が基準以前・鮮度300秒・同一馬番集合を維持し、T30/T60欠損はNaNとフラグで残します。元DBへの更新・バックフィルは行いません。

## 統計と候補

主指標はレース勝馬Log Loss。新−対照が負なら改善です。月曜起点・全競馬場共通の週で5,000回、seed20261019の対応ありbootstrapを使用し、同じ母集団には同じ再標本を使います。主要3比較に95%と98.333…%区間を併記します。既参照の開発期間のため独立検証とは呼びません。

精度候補を保存してから収益を集計します。精度と収益の保持基準は別です。収益は購入ゼロ月を含む27か月の平均、年次、ROI、的中週、最大払戻依存、評価月だけの時系列DD、対照との利益差を保存します。購入ゼロのROIは未算出、未精算は0円へ置換しません。

## 保存と保護

全データ・モデル・詳細台帳・匿名レポートは次の新規フォルダーだけに保存します。

`C:\Users\acbd5\AppData\Local\JraRacingAI\stage4-development\evaluations\2f26b6573d2249d38ad96b4456f89c0f`

`references.json`は旧参照ファイル、`old_code_hashes.json`は旧コード、`tasks_before.json`はタスクXMLのハッシュです。終了時に不変を照合します。旧ファイル全体のハッシュ計算と、禁止期間の結果を解析・学習・評価することは区別します。2025年10月以降のレコードは専用読込で解析前に遮断します。

`plan.json`、`experiment_inventory.json`、`lineage/`、`oof_origins.json`、`direct/`、`T60/`、`accuracy.json`、`finance.json`、`candidate_freeze.json`、`development_report.md/json`を保存します。`paper/*.local.json`は非公開の馬券明細、`lineage/*.csv`は非公開の行別OOF系譜です。チャットへ貼る対象は匿名集計だけです。

未完了の学習・最適化があっても自動で再学習しません。保存完了済みモデルが1つに特定でき、ハッシュ・仕様が正しければ再利用できます。保存不完全なら停止し、予算を増やしません。完了済み各単位も依存ハッシュを検証します。
