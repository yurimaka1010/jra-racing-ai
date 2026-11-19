# 保存済み39特徴量TunerのGain順位3項目の除外比較

ユーザーが指定したのは残数・削除数ではなく、元の39特徴量Tunerモデルの
Gain重要度24位・28位・30位の3項目だけの除外である。対象はそれぞれ
`age`、`surface`、`horse_number`。残り36項目を元の列順で使用する。

## 実行前に固定する比較条件

- 対照：保存済み39項目Tuner、モデルID `19e0b6227c4b44349243207d248fef6e`。
- 新構成：36項目の二値LightGBMをLightGBMTunerで1構成だけ最適化。
- Tuner：4段階各2試行、実際の学習呼出上限8fit、失敗も消費、追加refitなし。
- 各最大300ラウンド、早期終了20、CPU4スレッド、12GiB、全体900秒。
- 学習2022-01-01〜2024-08-31、早期終了2024-09-08〜10-31。
- 温度・構成選定2024-11-08〜12-24。温度候補0.75/1/1.25/2。
- 主指標はレースLog Loss。2024年最小値+0.002以内なら少ない特徴量を優先。
- 比較対象は固定した両構成。2025年1〜9月の既存共通2,169レース・29,879頭で
  対応比較し、馬単位LL・Brier・AUC・レース内AUC・1位的中率・較正表も保存。
- 差の符号は36項目−39項目。開催週単位の対応bootstrap1,000回を使う。

2024年窓は温度当てはめと構成選定に共用している。2025年も再利用済みの開発評価であり、
未使用最終テスト・正式な同等性認定ではない。2025年の結果で再探索しない。
既存のデータキャッシュ・行集合・カテゴリ辞書・欠損処理を再利用し、実DBへ接続しない。
H・Stage7・朝7時タスク・旧成果物・封印2026年は変更も評価もしない。

## コマンド

既存のOneDrive外venvのPythonで、プロジェクトフォルダーから実行する。
`prepare`の返す新しいrun_idを指定する。実データ・モデル・詳細予測は
`%LOCALAPPDATA%\JraRacingAI\tuner-reduction\evaluations\<run_id>`だけへ保存される。

```powershell
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B .\scripts\run_tuner_rank_ablation.py prepare
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B .\scripts\run_tuner_rank_ablation.py execute --run-id <run_id>
```
