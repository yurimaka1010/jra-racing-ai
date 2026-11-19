# KEIBA LABへのローカルCSV変換

Stage 6の保存済み開発評価を、連携用v1.1の25列へ縦持ち変換します。新規学習、DB接続、外部通信、タスク変更はありません。旧モデル・予測は書き換えません。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
& 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe' -B '.\scripts\export_keiba_lab.py'
```

既定では保存ファイルの最初の1レースを選びます。結果やモデルの的中状況で選び直しません。全頭・全4系列（H、raw_market、H_style、H_pace）をまとめて出力します。H_styleは個体対照、H_paceは不採用研究モデルです。主力Hと混同しません。

対象期間を広げる例：

```powershell
& 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe' -B '.\scripts\export_keiba_lab.py' --from '2025-09-01' --to '2025-09-30' --race-limit 20
```

入力は既知のStage 6 run `719a0d3915194bc59b6e55eba2659576` の `predictions.local.csv` と、その結果JSON、Stage 2 run `0311a896a6684d4c8105591160cbf751` のsource manifest・必要な月次RA/SE保存CSVだけです。任意パス指定や保存領域の再帰走査はしません。2023・2024・2025年の各1〜9月以外は拒否します。2026年封印期間とStage 7の保存領域を読みません。

毎回 `%LOCALAPPDATA%\JraRacingAI\keiba-lab-export\evaluations\<新ID>\` を新規作成します。

- `keiba-lab.csv`：UTF-8 BOM、CRLF、25列。書込後に読取専用属性を付けます。再生成は新IDです。
- `manifest.json`：参照ファイルとSHA-256、年別モデルIDを保持した変換の来歴、欠損セル数、照合値、取込阻害コード。
- `report.md`：個別馬・生オッズを含まない簡潔な照合結果。

上限は20,000行・8MiBです。全レース・全モデルを保ったまま上限を超える要求は停止します。日付範囲やレース数を分けて実行してください。KEIBA LABは1ファイルで全件置換するため、表示成績はそのファイルの部分集計です。

## 対応と欠損

`entry_key`を16桁の完全レースキーと2桁馬番へ分離。予測値を再計算・正規化・丸めず保存し、降順・同率馬番昇順で重複しない序数順位を作ります。モデル実IDは保存列を使います。raw_marketは学習モデルではないため、仕様にある方式識別子`no_fit:raw_market:T10`を使います。

確定着順は保存済みRA/SEを完全キーで結合します。勝敗ラベルから2着以下を作りません。全頭の連続着順を確認できない場合（同着、中止などを含む）、全頭の結果を空欄で保留します。勝馬ラベルとの矛盾、キー不足・重複は停止します。

保存元にない馬名・レース名・馬場状態・表示用クラスは空欄です。コースはJV track_codeのまま保存されているため、未確認の表示用変換は行わずsurfaceも空欄にします。元列とファイルは参照ハッシュから追跡できます。距離は保存値を使用します。

実際の予測作成時刻が保存元で確認できないので、`predicted_at`と`start_time`は両方空欄です。出力時刻を過去の予測時刻に流用しません。`evaluation_kind=stage6_development`、`source_stage=stage6`を固定し、真正な発走前予測と表示しません。Stage 6仕様に従い払戻は全行空欄、収益評価は対象外です。

**2026-10-24の受入規則追補**：列・schema_versionはv1.1のまま、`2026-10-24.stage6-missing-display` でStage 6研究用に限って名称・surface・distance・going・classの空欄を許容します。画面の「未取得」は表示ラベルで、元CSVには書き戻しません。確率・馬番・順位・着順・来歴の検証は維持します。旧受入仕様の可否は`legacy_strict_display_import_ready`、新仕様は`receiver_contract_revision`と欠損警告件数で区別します。過去のCSV・manifest・報告は変更しません。列互換や単体照合と、実画面での取込成功は別に報告します。

受入仕様の単一参照先は [KEIBA LAB連携仕様](<../../sites/jra-private-lab/dist/integration-contract.md>) です。既存18列と来歴7列、列ごとの必須性、欠損、行エラー、全件置換、封印・Stage 7拒否をまとめています。通常のエクスポートコマンドが受入サイトを書き換えることはありません。

## 照合指標

実際に書き出したCSVを読み戻し、全確率、予測順位、モデルID、全着順を元の保存値と比較します。既存`stage5.metrics.point_only`（Stage 6と共通）でROC-AUC、レース内AUC、レース勝馬Log Loss、馬単位二値Log Loss、馬単位Brier、同率按分1位的中率を再計算し、差が1e-12を超えたら停止します。元データは変わらないことを使用前後のSHA-256で確認します。

KEIBA LABのBrierはレース内誤差**合計**のレース平均、Stage 6は**馬**平均です。同率1位もKEIBA LABは序数1位の1頭、既存評価は同率頭数で按分します。この2点は規則が異なり、無条件に同値とは報告しません。開催後の開発評価の再集計であり、独立最終評価・真正な前向き評価ではありません。

CSVや元データをチャットへ貼ったり外部へアップロードしたりする必要はありません。標準出力は件数・固定判定コード・ローカル出力先だけです。
