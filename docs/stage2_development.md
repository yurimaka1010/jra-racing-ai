# Stage 2・限定Stage 1.5：有限の開発評価

旧best242・T=0.75、過去の評価・台帳・予測は保持します。新機能は `jra_racing_ai.research` に独立して実装し、既存の `FEATURES` / `DATA_FIELDS` や旧CLIを置き換えていません。実データ、モデル、レース別予測は `%LOCALAPPDATA%\JraRacingAI\stage2-development` 内だけへ新規保存します。

主指標は**レース単位の勝馬Log Loss**、差は**新モデル−対照**です。馬単位二値Log Loss・Brier・1位的中率・確率帯別校正も保存します。2025年の既存3期間、2023〜2025年の年次確認はいずれも候補選択に使用する開発評価です。

## 実行済みの開発単位

作業ID：`0311a896a6684d4c8105591160cbf751`

保存先：`C:\Users\acbd5\AppData\Local\JraRacingAI\stage2-development\evaluations\0311a896a6684d4c8105591160cbf751`

- `plan.json`：最初の学習前に保存した比較条件、探索上限、時系列分割。
- `preserved.json` / `old_artifacts_verified.json`：既存成果物の開始時SHA-256と終了時の保持確認。
- `reproduction.json` / `cache_legacy_parity.json`：Stage 1の保存指標と旧best242の予測再現。
- `mapping.v2.json` / `feature_definitions.v2.json`：確認した物理列と意味・単位・初期値・計算式・利用時点。
- `source_manifest.json` / `feature_manifest.json`：月分割取得のハッシュ、欠損件数、評価可能レース数。
- `models/<新規ID>/model.json`：入力版、特徴量名と順序、学習時カテゴリ辞書、計算式版、較正、分割、LightGBM設定、モデルハッシュ。
- `trials/`：全構成の指標、開催週・レース単位CI、欠損、仮想収支。`.local.json` は個別データなので共有しません。
- `experiment_ledger.json`：全実験の台帳。
- `development_report.md` / `.json`：匿名の比較、期間別指標、校正、月次損益、被覆率。
- `candidate_freeze.json` / `freeze_checksum.json`：候補と手順の固定、チェックサム。
- `final_evaluation_plan.json`：将来の最終評価の事前計画。今回は実行しません。

## 入力と特徴量

RA/SEは必要列だけを月単位・最大2万行で取得します。既存のMAX_ROWSを変更していません。元DBは読み取り専用トランザクション、既存のタイムアウトとSQL検証を使用します。取得後の同じ日次ストリーミング生成器を、学習キャッシュと推論の両方で使用します。

過去成績の反映は過去レース日+7日以降です。ラベル用の当該着順や、履歴用の当該上がり・着差・通過順は、そのレース自身の特徴量に入りません。当該払戻・確定オッズは能力特徴量の入力契約にありません。

| 群 | 初版の定義 | 欠損・注意 |
| --- | --- | --- |
| A | `(実出走頭数−着順)/(実出走頭数−1)` の直近値、直近3走の重み1・0.5・0.25による平均 | 過去のRA `shusso_tosu` を使用。旧 `declared_runners=toroku_tosu` の意味は維持。無効着順は欠損 |
| B | 当該馬齢用のクラスコード、グレード、競走種別、重量種別、過去コード、コード遷移 | 005/010/016等は生カテゴリ。2019年前後の名称変更を数値序列へ変換しない。000/未知値は欠損。グレード空白は公式の一般/初期値カテゴリ |
| C | 過去3F秒・全出走馬内の順位/平均との差、直近3走順位集計、頭数正規化した4角位置、確認済み前馬着差 | 障害の平均1Fは3Fとして扱わない。全実出走馬/3Fが揃わない過去レースの相対3Fは欠損。H/K/T/Zは数値推測しない。追加着差は降着・失格等の補足で、勝馬との差に合算しない |
| D | 騎手/調教師の直近365暦日内・7日遅延済み出走数と、`(勝数+20×遅延済み全体勝率)/(出走数+20)` | 集計対象日は両端を含む `[予測日−365日, 予測日−7日]`。対象馬で絞らず、国内JRA10場の全騎乗/全出走を使用。取消/除外は出走に含めず、不完走/失格は含める。地方/海外を含む成績ではない |

既存の17特徴量は互換性のため保持します。旧比較と同じく `going` と `previous_time_seconds` は今回も全欠損です。任意特徴量が欠損・未確認でも評価レースを除外しません。基本の出走馬集合・単一勝馬・着順の完備性などは既存の比較条件を維持します。現在グレード/騎手に変更前コードが記録されている場合、変更時点が不明なので当該入力を欠損にします。

旧形式は既存のローダーへ委譲します。新形式はversion 2の厳密な特徴量順序・辞書・式とモデルハッシュを検査します。新しい能力モデルの推論は `research.model.predict_inputs(bundle, rows, history=...)` を使用します。市場アンカーは `market_data.augment` で同一馬番集合の市場入力を付加し、`research.model.predict` を使用します。

## 限定Stage 1.5

T−10市場確率 `q=(1/odds)/Σ(1/odds)` をoffsetにします。

1. 学習と早期終了用Datasetの双方に `init_score=logit(q)` を設定。
2. `boost_from_average=False`、少数固定のLightGBM設定で残差だけを学習。
3. 推論では保存モデルの `raw_score` に外部offset `logit(q)` を加算。
4. sigmoidを適用してレース内で再正規化。残差ゼロならqを再現。
5. アンカーTは1.0固定。best242のT=0.75は流用しません。

比較は市場情報のみ、市場+T−30変動、市場+旧特徴量+A+B、市場+旧特徴量+A+B+変動、最終的な全群+変動です。モデルの予測値は特徴量にしません。将来予測値を入力する場合は時系列OOFが必要です。

履歴は正規化確率の対数変化、人気順位変化、実際の経過分、履歴欠損フラグだけです。早期履歴欠損は主評価に残し、履歴が揃う部分集合も別比較します。採用した発表時刻、鮮度、欠損理由はローカル明細へ保存します。

既存Stage 1と同じく、発表分の末尾が判断時点以前・鮮度300秒以内を採用します。発走変更記録あり、同一馬番集合が揃わないもの等は除外します。過去のPC実取得時刻と発走情報の当時版は確認できないため、**JRA公表時刻と保存RAを使う事後評価**です。購入可能性や約定実績を証明するものではありません。

## 実験上限と時系列

能力対照/A/B/A+B → 限定市場4構成 → C/D/全群 → 有望な場合だけ最大4つの一群除外 → 市場全群1構成、の順です。

基本設定：最大300ラウンド、早期終了20ラウンド。242は新モデルの固定木数にしません。能力モデルのT候補は0.75/1.0/1.25/2.0で、別の較正期間だけで選択します。主学習は2016-07-02〜2024-11-30、早期終了2024-12-08〜28、較正2025-01-11〜02-02、探索3期間は2025年3〜5月/6〜8月/9〜11月です。市場入力を使う学習は2020年以降です。

2025年のプール勝馬Log Lossで能力・アンカー各1候補を選び、2023/24/25年の各年初起点で過去だけから学習し直します。前年8月末まで学習、前年9月8日〜10月末で早期終了、前年11月8日〜12月24日で較正、当年1〜9月で評価します。各候補の対照も同じ条件で再学習します。

年次確認前に「年次プール差が負、かつ3年中2年以上で改善方向」を開発候補の保持条件として保存します。信頼区間が0を含む場合は不確実と表記し、効果なしと断定しません。仮想収支を特徴量や候補の選択基準には使いません。

不確実性は開催週（全競馬場を同じ月曜起点の週へまとめる）ペアードbootstrap、1000反復、開発95%区間で測定します。既存のレース単位95%区間も参考として保存します。

## 本人用の再実行手順

既に完了した保存先を再利用して上書きしません。確認はまずローカルの `development_report.md` と `candidate_freeze.json` を開いてください。新しい開発単位を再現する場合だけ、新IDを作成して以下を実行します。新しい年やパラメータを探索する入口はありません。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
# initはDBへ接続しません。出力のwork_idを次の変数へ設定します。
& $taskPython -B '.\scripts\stage2_local.py' --stage init
$taskWorkId = '<上で発行された32文字のwork_id>'

# DBを読む2段階だけ、既存のDPAPI認証とWindows PowerShell 5.1を使用。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\stage2_local.ps1' -Stage collect -WorkId $taskWorkId
& $taskPython -B '.\scripts\stage2_local.py' --stage build --work-id $taskWorkId
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\stage2_local.ps1' -Stage market -WorkId $taskWorkId

# 以後は保存データだけ。各行のstatus:okを確認して次へ進みます。
& $taskPython -B '.\scripts\stage2_local.py' --stage ability-ab --work-id $taskWorkId
& $taskPython -B '.\scripts\stage2_local.py' --stage anchor-ab --work-id $taskWorkId
& $taskPython -B '.\scripts\stage2_local.py' --stage ability-cd --work-id $taskWorkId
& $taskPython -B '.\scripts\stage2_local.py' --stage anchor-full --work-id $taskWorkId
& $taskPython -B '.\scripts\stage2_local.py' --stage annual --work-id $taskWorkId
& $taskPython -B '.\scripts\stage2_local.py' --stage freeze --work-id $taskWorkId
```

既存の同一WindowsユーザーのDPAPI資格情報を使います。DBは `127.0.0.1:5432/pckeiba`、ユーザー `jra_audit_reader`。PostgreSQLは本人が通常の方法で起動してください。ライブラリを新規導入せず、モデルごとの `library_versions` に実行環境を保存します。途中失敗時は固定理由コードを確認し、未完の取得/キャッシュを推測して再利用しません。新しい作業単位で初めから再現するか、コード上で検証された完了済み段階だけを継続します。

単体テストはネットワークと実DBドライバーを禁止します。

```powershell
$env:TEMP = Join-Path $env:LOCALAPPDATA 'JraRacingAI\synthetic-tests'
$env:TMP = $env:TEMP
& $taskPython -B '.\tests\run_tests.py'
```

## 収支診断と未使用期間

仮想購入は単勝100円・EV5%以上・最大1点を固定。T−10公表オッズで判断し、既存の確認済みHRデコーダーで通常払戻を精算します。未確認の特殊/複数版/不一致は未精算にし、未精算を0円へ置換しません。未精算がある全体の純利益/回収率は未算出とし、精算済み部分集合を分けて報告します。月別損益、購入数、未精算、最大払戻依存、最大ドローダウン、1番人気の共通条件比較を保存します。閾値・賭け金・購入ルールの最適化はしません。

2026-01-01〜09-30はDB取得・キャッシュ読込・学習・評価の開発経路から拒否します。元の2026年成果物は保持確認のためハッシュするだけです。別入口 `scripts/stage2_final.py` は**IOも評価もせず固定コードで停止**します。将来の別途指示後にのみ、最終評価を有効化します。主要2比較を行う場合は各97.5%区間（Bonferroni）とし、新−対照の上限が0未満の場合だけ正式な改善と認定します。
