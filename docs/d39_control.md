# D39対照型を含む4モデル合同記録（2026年10月24・25日）

旧D39・新D39・D39対照型・固定Hを同一レース・同一T−10入力締切で比較します。
新規学習は対照型の1fitだけです。Tuner探索・温度の再選択は行いません。

## 対照型の固定条件

- パラメータ：旧D39の保存済み最適パラメータを完全一致で再利用。
- 木数215、温度T=0.75。早期終了や温度の再較正なし。
- 学習データ：新D39の `final_train.local.csv` と同じバイト列・ハッシュ。
- 2022-01-05〜2026-09-20、14,681レース・201,651頭。
- 特徴量39項目の計算式・順序は共通。カテゴリ辞書も新D39と同一の最終学習データ由来。
- 1fit、CPU4、親子メモリ12GiB、5分上限。失敗も試行を消費し、自動再試行なし。
- 元の新D39研究の9fit計画は変更せず、追加1fitの承認と予算を別ファイルへ固定。

「旧D39→対照型」で学習データ更新の寄与を、「対照型→新D39」でパラメータ・木数・温度の再選択を含む違いを比較できます。
カテゴリ語彙の変化はデータ更新に伴うもので、3モデルの計算式を変更するものではありません。
学習済み期間の事後成績を未使用評価として示しません。

## 開催日の起動

既存run_id：`86ed126ffc8f4e2e85c8740eacf941f0`。
成果物ルート：`%LOCALAPPDATA%\JraRacingAI\d39-generation\evaluations\86ed126ffc8f4e2e85c8740eacf941f0`。

同フォルダーの **start_four_model_comparison.cmd** を、24日・25日の朝にそれぞれダブルクリックします。
従来の3モデル用起動ファイルも残っています。通常は4モデル用だけを起動してください。
PC-KEIBAとPostgreSQLを通常どおり起動・更新し、端末を閉じず、PCを起動したままにします。
タスクの登録・変更は行いません。既存の朝7時タスク・Stage 7はそのままです。

PowerShellで直接起動する場合：

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\d39_current.ps1' -Mode record-four -RunId '86ed126ffc8f4e2e85c8740eacf941f0'
```

開催日前に起動すると出馬表件数だけを確認して終了します。開催日の起動が必要です。
読取り資格情報は既存の本人ユーザー用DPAPIから取得し、パスワードは記録しません。

## 保存と互換性

| 対象 | 既存3モデル | 追加4モデル |
|---|---|---|
| 計画 | forward_plan.json | forward_four_plan.json |
| 予測 | forward/YYYY-MM-DD/races | forward-four/YYYY-MM-DD/races |
| 確認用件数 | readiness | readiness-four |
| 答え合わせ | future-evaluations | future-evaluations-four |
| 起動 | start_three_model_comparison.cmd | start_four_model_comparison.cmd |

既存ファイルを上書きしません。4モデルの勝率・順位・モデルID・温度・生成時刻・同じ入力締切・オッズ時刻・EV・仮想額をCSV/JSONへ保存し、ハッシュを固定します。
既存モデル3本は、第4モデル追加前後で同じ入力なら同じ確率と判断を返すことを検証します。
4モデル用で対照型が読めない場合は、3モデルとして黙って記録せず停止します。
第4モデルを開催前に用意できない場合のみ、従来の3モデル用起動ファイルを使います。

オッズは自分の観測時刻がT−10以前、発表分末尾がT−10以前、鮮度300秒、馬番集合一致を要求します。
過去走は既存の7日遅延、Hは既存の市場offsetとT=1を維持します。
既に校正済みの確率を仮想購入処理へ渡すときの温度は1とし、二重適用しません。
データ欠損・時刻や出走馬集合の変更は見送りです。実馬券購入は行いません。

## 答え合わせ（翌日以降）

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\d39_current.ps1' -Mode evaluate-four -RunId '86ed126ffc8f4e2e85c8740eacf941f0' -RaceDate '2026-10-24'
```

4モデルの保存記録を検証してから、完全キー・同一馬ID・同一出走馬集合の確定結果を読みます。
同じ共通レースでLL・Brier・AUC・1位的中率・校正・距離/芝ダート別・固定EVルールの仮想収支を集計します。
払戻未確認・特殊ケースは保留し、オッズを払戻の代用にしません。
3モデル用の `evaluate` も引き続き利用できます。

## 確認用コマンド

`preflight-four` は4モデルを読込み、モデル・学習データ・計画ハッシュを確認します。
`verify-four` は保存済み実モデル4本と架空3頭を使い、予測からCSV/JSON保存までを検証します。
後者は明示的なsyntheticフラグと独立フォルダーを使用し、実開催の予測や成績には含めません。
新たなfitは行いません。

対照型の学習は `prepare-control` → `train-control -ControlId <id>` で実施済みです。
再実行は不要で、既存開始記録がある学習を再実行しても停止します。
