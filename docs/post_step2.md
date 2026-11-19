# STEP 2後：独立評価・条件付き過去タイム・発走前観測

旧V0.3・STEP 2モデル・2026-10-18 09:26の予測・朝7時タスクは保持する。
実データはすべて `C:\Users\acbd5\AppData\Local\JraRacingAI\post-step2` 以下。
既存のDPAPI認証と読取専用ユーザーを使う。外部通信、ライブラリ追加、実購入はない。

## 固定した実験

- 作業ID: `2d8fd55592234b4a891974d03ee98ffc`。
- 学習2016-07-02～2024-11-30、26,173R。早期終了2024-12-08～12-28。
- 校正2025-01-11～02-02。候補温度0.75/1/1.25/2、既存のレースLog Lossで選ぶ。
- 開発比較2025-03～05、候補確認06～08、選択後の最終評価09～11。
- 主指標は馬単位二値Log Loss。開発LLが下がる候補だけを比較用に選び、その後の最終評価では変更しない。
- 全モデルで同じ出走馬集合、単独勝馬、従来の基本特徴量条件。任意特徴量の欠損で対象を変えない。
- モデルの選択、温度、購入閾値に10/18の結果を使わない。

## 互換性と形式

mapping形式1は従来どおり。形式2に `formats: {finish_time_seconds: jv_mss_tenths_4}` を追加した。
この形式を使う場合、同フィールドの `scales` 指定は禁止。重複変換を拒否する。
JV SE項番44（ローカル公式PDFの12ページ）のMSSdを `M*60+SS+d/10` で秒へ変換する。
0000、空欄、NULL、不正文字、長さ不一致、秒60以上は欠損。値の大小で単位を推測しない。
ローカルsoha_timeは48万行以上で4桁・秒部正常を確認。公式画面との個別照合は未実施。

新モデルだけに `feature_profile=conditioned_previous_time_v1` を保存する。
`previous_time_seconds` は、7日経過した結果のうち、同距離・同一トラックコードの直近有効タイム。
旧モデルはprofile未指定で従来定義のまま。`model.predict` と `v03.predict_win` は保存profileを使用する。
goingは無効のまま。2024/2025のWE履歴や初期状態がなく、最終RA馬場を当時の発走前情報に置換しない。
「馬場のみ」「両方」の比較は未実験として報告する。

## 実行手順

既にinventory/collect/evaluate/featuresは実行済み。再学習するために繰り返す必要はない。
新しい実験は別作業IDと新しい事前固定planで開始する。

以下はプロジェクトディレクトリのPowerShellから実行する。認証情報の入力・表示は不要。

```powershell
# 10/18の固定予測用。1回だけ現在オッズを読み、期限到達済みの判断を排他的保存する。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\post_step2_local.ps1' -Stage capture -WorkId '2d8fd55592234b4a891974d03ee98ffc'

# 必要な場合だけ30秒間隔・最大10分。タスク登録や終日運転は行わない。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\post_step2_local.ps1' -Stage capture -WorkId '2d8fd55592234b4a891974d03ee98ffc' -DurationSeconds 600

# PC-KEIBAが結果・HRを更新した後。再学習せず新規フォルダへ36R比較を出す。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\post_step2_local.ps1' -Stage replay -WorkId '2d8fd55592234b4a891974d03ee98ffc'

# 最新の再生結果から匿名集計レポートを新規作成する。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\post_step2_local.ps1' -Stage report -WorkId '2d8fd55592234b4a891974d03ee98ffc'
```

別の日の新しい予測を自動生成する日次システムではない。captureはこの作業の10/18固定予測に対応する。
PC-KEIBA更新設定、Windowsタスク、旧policyを変更しない。

## オッズ時点・仮想決済

- sourceのO1形式は以前本人確認済みの28×8文字・倍率1/10を再利用。
- DB読込完了時計は本プログラムの受信時点。PC-KEIBAの元の取込時点ではない。
- 発走予定10分前以前の発表時計と実読込時計だけを使用する。分精度の境界を秒0と推測しない。
- 発表時点・取得時点それぞれの最大経過時間120秒。最新値が不正なら古い正常値へ戻さない。
- EV5%以上、最大EV、1R1頭100円。予測1位とは別項目。
- モデルID＋6列キーの判断ファイルを排他的に作成。先に作った見送りも後から書き換えない。
- 新候補を後で計算した判断は事後再現。保存済み事前判断の実績に混ぜない。
- HRは既存の正常単勝デコーダーと本人確認済み円/100円の対応だけを再利用。特殊・不明・取消は保留。
- 未精算があれば全体純利益・ROIは未算出。0件購入のROIも未算出。
- 最大DDは予定発走順で、同時刻はまとめて計算。実際の資金移動時刻によるDDではない。
- 高配当依存は最大払戻シェア、最大的中を除いた純利益など。未精算と精算済み部分を分ける。
- 2025年には発走前オッズ履歴がないため、月間EV収益は未評価。確定市場確率は予測精度の参考のみ。

## ファイル

作業フォルダ直下にplan、列・集計、独立比較、特徴量比較、選択、最終評価を保存。
`odds_snapshots/` はPC内限定原入力、`decisions/` は固定判断。
`evaluations/<新ID>/comparison.local.md` は全36Rの個別比較、`predictions.local.csv` は新旧の出力。
同フォルダのsummary.jsonは匿名集計。reportステージは別の新IDに総合report.mdを作成する。
`.local`・原入力・個別表はチャットや外部へ貼らない。
