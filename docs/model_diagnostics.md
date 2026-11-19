# 保存モデルの重要度・補助分類指標（再学習なし）

`model_diagnostics`は、既知の保存モデルとStage 6までの開発評価の再集計専用です。DB接続、学習、モデルの変更、閾値探索、Stage 7記録の読み込みは行いません。

## 起動

プロジェクトディレクトリのPowerShellから、既存のOneDrive外venvを使います。パスワード入力は不要です。

```powershell
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
& $taskPython -B '.\scripts\model_diagnostics.py' --init
# 上で出力したrun_idを指定。毎回、新規IDを使います。
& $taskPython -B '.\scripts\model_diagnostics.py' --run-id '<run_id>'
```

出力先は`%LOCALAPPDATA%\JraRacingAI\model-diagnostics\evaluations\<run_id>\`です。既存ファイルを上書きしません。

- `diagnostics.html`：ダブルクリックで閲覧。外部通信・CDN・JavaScriptを使わない比較表と重要度グラフ。
- `feature_importance.csv`：全特徴量の元順序、gain、split回数、gain割合、表示順位、モデルID・役割・対象範囲。
- `classification_metrics.csv`：全体・年別、モデル別、判定規則別の混同行列、正解率・適合率・再現率・F1。
- `diagnostics.json`：既存6指標の再現、補助指標、モデルごとの重要度、来歴・ハッシュ・保全結果。
- `protection_before.json`：処理前の既存ソース、固定H、Stage 7計画、タスク定義の保全記録。

初回の確認済みrunは`7ebb5992bf2a4cde9bb12fa2419d0f9a`です。そのrunのHTMLを開くだけならPythonは不要です。結果・詳細ファイルは外部へアップロードする必要がありません。

## 重要度の意味

モデルのメタデータと保存木ファイルのハッシュ、特徴量名・順序を確認してから読むだけです。単勝の保存木を使い、特徴量を自動選択・削除しません。H/ABCD/H_style/H_paceの年別モデル、旧V0.3・best242の計14モデルを対象とします。旧2モデルは重要度のみで、今回の比較母集団の保存予測とは扱いません。

Hは外部市場offsetに残差を加えるモデルです。**木のgain/split重要度には、その外部offsetの寄与は含まれません。** 重要度は因果効果や将来の有用性の証明ではありません。全gainが0なら割合は未定義として保持します。

## 補助指標の意味

対象は2023〜2025年各1〜9月の同一開発母集団です。H・raw_market・ABCD・H_style・H_paceの保存確率を使用し、新たな推論・較正・学習は行いません。単独勝馬、同一レース・同一出走馬集合を確認します。

1. `probability_ge_0_5`：確率が0.5以上を陽性とする固定規則。
2. `race_top1_horse_number_tie_break`：レース内1位だけを陽性。同率は馬番昇順。

混同行列は馬単位のTP/FP/TN/FNです。適合率等の分母が0ならJSONはnull、CSVは空欄にします。0.5判定は購入ルールではありません。勝馬が少ないため正解率だけを良さの根拠にしません。

既存のAUC、レース内AUC、レース勝馬Log Loss、馬単位Log Loss・Brier、同率按分1位的中率は変更せず、同じ既存関数で計算して保存レポートと照合します。序数1位判定と同率按分の違いも残します。

## 研究・運用との境界

これは再利用した開発データの診断です。新しいモデルの学習・性能改善を完了したという扱いにはしません。2026年データ、Stage 7の記録・途中成績を受け付ける入力オプションはありません。固定H、既存朝7時タスク、Stage 7ソースと計画・タスク定義を変更しません。Stage 7の内部実装ハッシュに含まれるファイルは追加・変更していません。

新規学習・探索が必要な多クラス分類、回帰、LambdaRank、Optuna、Boruta等は別の実験です。対象データ・目的変数・最大fit数・比較基準・予算の事前確認後に実装・実行を進めます。今回のCLIには学習用の入口はありません。

## テスト

既存の`tests/run_tests.py`で実DBドライバー・Pythonネットワークを遮断し、架空データ・モックの重要度で検証します。追加テスト自体はfitを行いません。旧回帰テストに含まれる合成データfitは実データ学習と区別します。
