# 複勝・ワイド・馬連の独立研究基盤

2026-10-24追補：本人が固定計画の最大2fitを承認し、新run `b45febcdf2e140ce9fddd01b5d1e880a` で複勝直接1fit・共通着順1fitを完了しました。通常HR対象を金額単位と分離して接続し、2025年の共通2,162レースで比較済みです。既存0fit runは変更していません。現行の画面・条件・未精算範囲は [multiwager_lab.md](multiwager_lab.md) を参照してください。以下の「未接続／0fit」は初版作成時点の記録です。

最大2系統：公式複勝対象を直接学ぶ二値モデルと、共通の着順分布から馬連・ワイドを計算するモデルです。
固定H、Stage 5/6、朝7時タスク、Stage 7記録器、単勝の既存購入条件は変更しません。
2026年1〜9月とStage 7途中結果は読みません。DB接続・書込み・自動購入もありません。

## 先行する0fit基準線

Stage 6の6373レース・87601頭の保存予測、Stage 2の月次sourceの着順、Stage 5同時点O1の着払キーを完全キーで結合します。
H・市場・ABCDの年別保存確率からPlackett–Luce（PL）の共通着順分布を定義し、複勝2/3着以内・上位2頭の馬連・上位3頭の組合せワイドを計算します。
同着なしの確率モデルという仮定を明示し、馬同士の独立確率の積では計算しません。
複勝・ワイドのオッズ範囲を逆数や中央値で確率へ置き換える処理もありません。

上位3着内同着24レースは、公式HR対象が未接続の初版では分離します。
残る6349レース・87263頭を順位参考評価として集計します。
公式HRの券種別フラグ、特払、返還、対象・金額単位・訂正削除が未接続なので、公式的中確率検証・実収益検証の完了とは呼びません。

今朝の4学習方式は2025年の保存予測だけを別母集団で比較します。
旧複勝モデルは2025-01-19まで学習/検証したrank-proxyなので、それより後の2025年だけに0fit参考推論します。
Stage 5のH_placeは単勝モデルへの複勝価格追加であり、複勝直接モデルとして扱いません。

```powershell
# 今回の専用フォルダー。既存成果物を上書きしません。
$taskPython = "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe"
& $taskPython -B '.\scripts\multiwager_research.py' plan --run-id '881a828f4b0949388766374ddbf9a3ce'
# baselineは今回実行済み。重ねて実行すると排他保存で停止します。
# & $taskPython -B '.\scripts\multiwager_research.py' baseline --run-id '<新しい承認済みID>'
```

保存先：`%LOCALAPPDATA%\JraRacingAI\multiwager-research\evaluations\<ID>`。
`baseline_report.md/json` は匿名集計、`races.local.jsonl` と `predictions.local.jsonl` は個別データです。

## 新規2系統の学習経路

- `direct_official_place`: ABCD39列に、予測時点で確認できた2/3着払キー1列を追加。公式HRの通常払戻対象集合を直接0/1ラベルにします。単勝のようなレース内総和1への正規化はしません。温度は独立ロジットへ一度だけ適用します。
- `shared_order_probability`: ABCD39列と今朝のLambdaRankアダプターを再利用。raw scoreを強度へ変換し、一つのPL分布から複勝・馬連・ワイドを計算します。温度は較正期間の上位3着の順序負対数尤度だけで選びます。

提案予算：各1fit、最大2fit。各300ラウンド・early stopping20・CPU4。
学習2023-01-01〜2024-08-31、早期終了2024-09-08〜10-31、較正2024-11-08〜12-24、開発評価2025-01〜09。
各ワーカー600秒・全体1800秒・親と子孫プロセスのprivate memory12GiB。
`research-training.lock` をTuner等と共有して高負荷処理を直列化。失敗も予算消費、同承認の再試行は禁止です。

実行には次の両方が必要です。

1. `training_plan.json` のSHA256に対応する、本人の別途承認JSON `training_authorization.json`（source=human_user、allow_real_fits=true、real_fits_max=2）。準備処理は承認JSONを自作しません。
2. OneDrive外の検証済み入力マニフェスト。version、formula_version、history_lag_days=7、sealed_2026_used=false、label_contract=verified_ordinary_HR_place_targets、official_HR_source_sha256と、4分割のfile/sha256/start/end/rowsを指定します。

各入力CSVは既存ABCD列・entry_key/date・finish_position/actual_starters・place_paid_positionsを保持します。
複勝直接モデルは `official_label_status=verified_ordinary_HR` と `official_place_label` が全出走馬で確認された通常レースだけを受け付けます。
単なる順位proxyにこの状態を付け替えてはいけません。公式対象集合・券種フラグ・版・払戻の文脈との照合が必要です。
現在この公式入力は未接続です。新規の実データ学習は0fitのまま保留します。

```powershell
# 将来、本人による学習承認と公式入力確認後のみ使用。
& $taskPython -B '.\scripts\multiwager_research.py' train --run-id '<ID>' --training-manifest '<OneDrive外の検証済み入力JSON>'
```

予測入力は固定した39/40列だけです。着順・実出走頭数・ラベル・払戻・確定オッズを特徴量へ渡しません。
カテゴリ辞書は訓練から作り、計算式版・入力形式・較正・モデル識別子・保存再読込みを維持します。
任意特徴量の欠損だけで対象を削除しません。ラベル未確認は別の欠損理由として母数と併記します。

## 公式払戻・仮想収支の境界

`outcomes.OfficialOutcome` は券種ごとの検証済み正規化データの受け口です。新たにHRの生フィールドを推測してパースしません。
複勝/ワイド/馬連それぞれで、完全な的中対象集合・通常状態のフラグ・JPY_PER_100・訂正版採用条件が確認できた場合だけ、固定済み仮想券を100円単位で精算します。
返還・発売なし・不成立・特払・中止・未確認はpendingで、払戻0円や不的中へ置き換えません。
実際の券種別払戻入力と判断時点価格が未接続なので実ROIは未算出です。既存単勝のEV5%ルールは変更しません。
`summarize_settlements()` は固定済み仮想券の精算結果を券種別・月別に集計します。未精算があれば全体損益・回収率はnull、既知の精算済み部分は別に保持します。購入0件の回収率もnullです。

## KEIBA LAB引渡し

`keiba_lab_sample.local.csv` は `keiba-lab.multiwager.research.v1`。
1レースについてモデルID、元予測、出所ハッシュ、評価区分、券種、1/2馬番、確率、順位を保持します。
`rank_reference_hit` と `official_hit` を別列にし、後者・公式払戻は未確認なら空欄。
`predicted_at` も空欄で、開催後の生成日時だけを `generated_at` に記録し、発走前保存実績と誤表示しません。
これは将来表示用の新仕様であり、現在の単勝用KEIBA LAB v1.1へそのまま取込可能とは表示しません。
