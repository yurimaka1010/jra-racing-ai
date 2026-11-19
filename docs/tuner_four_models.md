# Gain上位10・20項目と保存済みBoruta19・全39項目の有限比較

2026-10-23の依頼に基づく開発比較。新規学習はA10・B20の各8fit、合計最大16fit。
C19とD39は保存モデル・保存予測・既存集計を再利用し、整合性確認のための予測再現だけを行う。
Boruta、C/Dの再学習、追加refit、探索の自動再試行を行わない。

- Gain順位は元のD39モデル `19e0b6227c4b44349243207d248fef6e` の保存木から固定。
- C19はBoruta研究 `552f89f1716643a39742ce84acb31cff` の19項目をそのまま利用。
- A/BのTunerは従来と同じ共通初期パラメータから開始。Dの最適値を初期値へ流用しない。
- 学習2022-01-01〜2024-08-31、早期終了2024-09-08〜10-31、較正・選定2024-11-08〜12-24。
- 最大300ラウンド、早期終了20、CPU4、private memory12GiB、全体900秒。
- 温度候補0.75/1/1.25/2、元のカテゴリ辞書・列順・欠損処理を共用。
- **2024年レースLog Lossの厳密な最小値で選定。許容差0.002を使わない。**
- 候補を固定後、2025年1〜9月の同一2,169レース・29,879頭で比較。
- 2025年は再利用済み開発評価。結果を見て再調整しない。2024年窓も温度と選定に共用している。
- 距離帯は〜1400m、1401〜1800m、1801〜2200m、2201m〜に事前固定。
- 芝・ダートの表示区分はローカルJV-Data4901.pdfのPDF45頁・表2009を使用。
  サンド27/28、障害51〜59は別群、不明・不整合は除外せずそれぞれ独立した群へ含める。
- 条件別成績は同じレース・同じ出走馬で記述的に比較し、特化型モデルの採否には使わない。
- 元DB接続・外部通信・2026年・Stage7途中成績・ROI最適化を行わない。
- H、既存モデル・成果物、Stage7記録器、朝7時タスクを保持する。

プロジェクトフォルダーからOneDrive外venvで実行する。

```powershell
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B .\scripts\run_tuner_four_models.py prepare
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B .\scripts\run_tuner_four_models.py execute --run-id <prepareが返すrun_id>
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B .\scripts\summarize_tuner_four_models.py <run_id>
```

成果物は `%LOCALAPPDATA%\JraRacingAI\tuner-reduction\evaluations\<run_id>` に新規保存する。
今回の候補決定で既存の運用モデルを自動変更しない。2026年の学習追加は次段階として分離する。
