# Stage 6：前走位置取り代理値×同走馬構成

承認済みの1仮説だけを検証します。H_style（個体対照10列）とH_pace（候補14列）、2023/2024/2025年の最大6 fit・各300ラウンドです。H・raw_marketは保存予測を再利用します。H_styleは候補へ昇格しません。

入力は既存Stage 2のキー・日付・`c_previous_corner4_percentile`と、Stage 4のT−10市場フレームです。前走値は7日遅延済み、元式は`(過去出走頭数−4角順位)/(過去出走頭数−1)`。今回の馬番集合の本人以外について、前寄り（値>=2/3）比率、既知値平均、判明率、本人値との固定相互作用を計算します。欠損で行を除外せず、未知値を後方や前方に補完しません。現在の結果・頭数・通過順位・確定オッズは入力しません。

主指標はレース等重みのレース内AUC。差は新−対照。H_pace−HとH_pace−H_styleの2比較だけに、共有週bootstrap 5,000反復、seed 20261019の97.5%両側区間を使います。実用幅は各+0.002/+0.001、両下限>0、対Hの年別方向・補助品質・再読込一致も事前固定します。2023〜2025年は再利用した開発評価で、正式な最終評価ではありません。

二値LightGBMのtrain/earlyに外部`logit(q)`を設定し、推論でも残差へ明示的に加算します。sigmoid後に既存のレース内正規化、T=1。目的関数、温度、seed、購入条件の探索はしません。

実データ・モデル・比較明細は`%LOCALAPPDATA%\JraRacingAI\stage6-development\evaluations\<run_id>`だけへ保存します。DB・ネットワークは使用しません。2025-09-30より後のデータ解析は拒否し、2026年は履歴生成用にも読みません。

## ローカル実行

既存venvだけを使用します。依存追加・認証・タスク登録は不要です。以下は新しい実験の承認を得た場合の初期化方法です。完了後の確認・再開は既存run IDを使い、別runで追加学習しないでください。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
& $taskPython -B '.\scripts\stage6_local.py' --stage init
# 出力run_idを設定。以後は同じIDで再開する。
$stage6RunId = '<run_id>'
& $taskPython -B '.\scripts\stage6_local.py' --stage run --run-id $stage6RunId
```

`plan.json`とそのSHA-256を学習前に固定。fit開始を排他的に予約し、完了・失敗とも予算を消費します。完了モデルはハッシュ検証して再利用し、未完了・失敗した構成を再学習しません。

`development_report.md/json`に年別・プール比較、2つの区間、採否条件、欠損・テスト・旧資産の保全を保存。`predictions.local.csv`は個別データのため外部共有しません。`experiment_inventory.json`は最大6予約の状態、`candidate_freeze.json`はH_paceまたはnullを保持します。主力Hと朝7時タスクは変えません。不通過なら今回の固定研究を終了し、代替案へ移りません。ROI・収益改善は今回評価しません。
