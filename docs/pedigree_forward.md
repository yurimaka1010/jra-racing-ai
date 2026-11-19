# 血統3モデル・独立前向き記録（0fit）

2026-10-26の実行指示と、ユーザーが確定した点推定型の閾値に従う研究専用処理。
正式予測系、KEIBA LAB、既存4モデル、朝7時タスク、H・Stage 7には接続しない。

## 固定条件

| 役割 | モデルID | 列数 | 温度 |
| --- | --- | ---: | ---: |
| control39 | faee8c8f0d5c428ab71a5c0a0b9d2687 | 39 | 1.25 |
| pedigree45 | a380e94c0e09492eada7a26733e27f1a | 45 | 1.0 |
| reference47 | b6888ec9550f437cbad817565a6c39cf | 47 | 1.25 |

木数219、学習・温度の再選定なし。モデルファイル、特徴量順序、型、カテゴリ辞書、
血統仕様をハッシュで固定する。実行コードはOneDrive外へ複写して固定するため、
本体リポジトリで後続研究を行っても記録器のコードは変わらない。
この作業フォルダーにGit管理情報はないため、架空のコミットIDを付けず、
`source-sha256:<コード集合のハッシュ>`をコード識別子とする。

主比較は **pedigree45 − control39 のレース内AUC**。
採用判定は、点推定が+0.004以上、週単位ペアード95%区間の下限が0超、
ΔRace LLが+0.005以下、ΔBrierが+0.00010以下の全条件。
歯止めは点推定で判断する。全体AUC・reference47は参考／診断であり主比較を増やさない。

2026-10-31開催から記録を開始する。正式評価は2026-10-26の週から連続21暦週、
2027-03-21終了。期間中は成績を計算せず、件数・欠測・鮮度のみ確認する。
終了後の一度だけの評価は別作業とし、自動昇格はしない。

## 保存先と起動

研究ID：`c7d80f921b3c42ae910a53a01326f8cb`

```text
%LOCALAPPDATA%\JraRacingAI\pedigree-forward\evaluations\c7d80f921b3c42ae910a53a01326f8cb\
  model_freeze.json / model_freeze.commit.json
  preregistration.json / preregistration.commit.json
  measurement.json / measurement_report.md
  runtime\project\                 固定した実行コード
  replay\<ID>\                      結果を読まない開催後再生、正式記録と分離
  history\<開催日>\                 7日遅延の追加履歴と血統IDスナップショット
  forward\<開催日>\races\<キー>\
    registered.json                 発走予定・全頭の登録スナップショット
    entries\ / odds\                自身が観測した入力
    attempt.json                    再計算禁止の処理開始記録
    control39.encoded.local.csv      モデル別エンコード済み入力行列
    pedigree45.encoded.local.csv
    reference47.encoded.local.csv
    prediction.json / predictions.local.csv
    receipt.json                    保存完了時刻・SHA256・成功／欠測
    veto.json                       保存が遅れた場合のみ（正式対象外）
  operational-logs\ / launcher-logs\
  task-registrations\<ID>\
```

標準出力は匿名の運用集計と固定理由だけ。個別CSV、モデル入力、血統ID、
オッズ等をチャットや同期フォルダーへ移さない。

運用件数だけを確認するPowerShellコマンド（認証不要）：

```powershell
$taskRun = Join-Path $env:LOCALAPPDATA 'JraRacingAI\pedigree-forward\evaluations\c7d80f921b3c42ae910a53a01326f8cb'
& (Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe') -B (Join-Path $taskRun 'runtime\project\scripts\pedigree_forward.py') ops --run-id c7d80f921b3c42ae910a53a01326f8cb
```

新設タスク名：`JraRacingAI_Pedigree3_Research_T10`。
10月31日から毎日07:00に起動し、開催があれば18:00まで15秒間隔で確認する。
既存のWindowsユーザー別DPAPI資格情報を再利用し、対話入力は不要。
PostgreSQLとPC-KEIBAの更新を起動し、当日の出馬表を朝7時までに収録する。
同じWindowsユーザーでログインを維持する（ロック可、ログオフ不可）。
PCを電源接続し、スリープ・休止させず、JST設定を保つ。
出馬表が朝7時に未収録なら、その旨を記録して当日の起動は終了する。
必要ならデータ更新後に同じ新設タスクだけを手動起動する。

DBセッションは既存の読み取り専用ユーザーと接続直列化を使う。
SQLは10秒、追加履歴は月単位最大2万行・読込予算300秒、
履歴Builder準備900秒、プロセスメモリ上限12GiB。DBへ書き込まない。

## 入力と失敗時の扱い

既存の出馬表取得・Builder・T−10締切を再利用する。結果・払戻・確定オッズは予測入力にない。
履歴は対象日の7日前まで。朝から登録した発走予定と馬番集合の変更は見送りとする。
入力はT−10までに自身が観測したもののみ、保存は既存契約の60秒猶予内に完了する。
分精度の発表時刻は分の終端を使い、オッズの鮮度上限は300秒。
記録済みの予測は上書き／再計算せず、失敗も欠測の記録として残す。
一つのモデルが読めなくても、読めるモデルの記録を継続する。
オッズ欠損・読込失敗でもP1用の勝率記録は続け、R1は見送る。

P1は全3モデルが全頭を保存したレースを対象とし、既存`race_reasons`を先に適用する。
P2はそのうち完全で新鮮なT−10オッズがあるレース。
P3は全3モデルと完全で新鮮なオッズがあるレースで、結果由来の異常を除外しない。
P3だけで各モデルの本命単勝100円を仮想記録する（R1）。EV条件は使わない。
取消・除外は返還、中止・失格は不的中。判断オッズを払戻額の代わりに使わない。
既知20レース・較正133レースは正式評価へ含めない。

## 検証と留保

10月24・25日の再生は、保存された出馬表とオッズだけを読み、
当該日の結果・払戻を読まない。新モデルは開催後に作られたため、
この再生を真正な発走前予測・未使用評価とは表示しない。

過去PCの実受信時刻、発走予定変更の完全な履歴、実際の発走時計、
血統マスタの過去訂正履歴は未確認。研究記録器自身のDB観測時刻を別途保存する。
当日の公式結果・HRの取得と21週終了後の最終集計は今回は実行しない。
母集団・返還・公式払戻の純粋関数と終了日ガードは実装済みだが、
将来の公式払戻入力経路への接続は未実施。
途中成績を出すCLIは用意していない。

テストは架空・モックのみ。実モデル再生では入力行列の保存・再読込、
生確率と温度適用後確率の一致、全頭網羅と確率和を確認する。
