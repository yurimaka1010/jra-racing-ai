# JRA競馬予想AI — Version 0.3

[Stage 3の開発評価と再実行手順](docs/stage3_development.md)を追加しました。保存木のλ収縮・温度診断、固定したABCDの時系列OOF、4構成の市場Softmax統合、条件付きE群を有限の実験数で比較し、収益集計前に候補を選択します。2026年の最終評価・再学習、実購入、タスク変更は行いません。

[Stage 2・限定Stage 1.5の開発評価](docs/stage2_development.md)を追加しました。旧形式を維持したA/B/C/D特徴量、市場logit-offset、有限の群比較、2023〜2025年の年次開発評価、候補凍結を独立した `research` パッケージで実行します。2026年1〜9月は開発経路から拒否し、将来の最終評価は別入口で停止しています。既存モデル・予測・台帳は上書きしません。

Version 0.3は単勝モデルの基本特徴量と過去走履歴を改善します。[本人用の実行手順](docs/version_0_3.md)に、144レース規模の再学習、独立した温度校正、未使用期間の確率評価、固定したVersion 0.2の価格・公式HRを使う参考比較をまとめています。旧モデル・複勝モデル・旧policy・旧台帳は上書きしません。新しいモデルとデータセットを発行し、学習・推論は共通の特徴量生成関数を使用します。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
& 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe' -B '.\scripts\run_v03_local.py'
```

実DB操作は本人が実行します。非表示のパスワード入力と基本列・負担重量の保存単位の確認後に進みます。新しい実データ・モデル・比較結果はすべてOneDrive外の既存ローカル領域へ保存し、最後のJSONは匿名集計だけです。架空データ・DBモックで既存619件を含む全638件が成功しました。実データの改善幅は、このコマンドの結果で判断します。

[市場確率ベースラインの測定](docs/market_baseline.md)では、保存済みの未使用評価66レース・949頭とT=2.0を固定し、市場・V0.3・一様予測のLog Loss・Brier・校正・順位相関を比較します。本人用コマンドは既存フォルダーで `python.exe -B .\scripts\measure_v03_market.py`。発走前と確定オッズは別集計にし、モデル・購入条件を変更しません。実測結果はOneDrive外の新しい評価フォルダーに保存します。

## Version 0.2の比較基準

Version 0.2は**単勝のみ**の期待値判断・仮想バックテストです。既存モデルの単勝確率と、判断時刻以前のオッズを使い、初期方針は発走予定10分前・期待利益率5%以上・1レース1点100円です。公式確定払戻で精算し、全対象を母数とするオッズ対応率、月次損益、同条件の1番人気戦略比較を保存します。

本人PCで、予測・発表時刻基準の購入判断・既存HRによる通常払戻精算・同時点1番人気との比較まで完了しました。[オッズ対応の手順](docs/v02_publication_run.md)と [HR精算の手順](docs/v02_hr_settlement.md)を利用できます。発表時刻は実取得・購入可能時刻と区別し、暫定的な事後シミュレーションと記録します。未確認のケースは保留し、収益性は機能の動作確認と別に評価します。

[既存HRによる精算](docs/v02_hr_settlement.md)は、6列SQL JOINと既存デコーダーを使い、固定した判断へ結果だけを追加します。本人提供の匿名集計で settlement_status=complete / decisions_unchanged=true / 未精算0 を確認しました。初回の対象HR0行は解消済みです。個別明細・実収益レポートはOneDrive外に保持します。既存597件＋追加22件の619件が成功しています。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $taskPython -B -m jra_racing_ai.win_sim_cli demo --save
```

[Version 0.2の本人用手順](docs/version_0_2.md)に、読取専用の可用性確認、方針固定、既存予測の再利用、DB/ローカルファイル入力によるシミュレーションを記載しています。Version 0.2の実成果物はOneDrive外 `AppData\Local\JraRacingAI\ml-v02` のみ。公式O1/HR原形式は本人指定PDFで照合済みですが、PC保存倍率は本人のローカル確認前に自動採用しません。[PDF照合記録](docs/jv_data4901_v02_review.md)を参照してください。本人確認済みの対象範囲では実データの通常単勝精算まで完了し、未確認の範囲には確認ゲートを維持します。追加ライブラリ・外部通信・DB更新・実馬券購入はありません。

## Version 0.1の学習・予測

Phase 2.0で独立したLightGBM学習・予測CLIを追加しました。[Version 0.1の実行手順](docs/version_0_1.md)に、最初の小規模学習、期間拡大、保存モデルの再予測、前向きシミュレーションをまとめています。HR金額の確認が未完でも着順の確率モデルは学習できます。複勝モデル・既存金額ゲートは維持し、Version 0.2の確認済み通常単勝の仮想精算は別モジュールです。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $taskPython -B -m jra_racing_ai.ml_cli demo
# 実DBは本人の非表示認証と必要列の確認後、小規模読込→学習→未学習期間予測。
& $taskPython -B -m jra_racing_ai.ml_cli first-run --execute-local
```

Python3.12.10の既存OneDrive外venvへ、承認を得てNumPy 2.5.4 / pandas 2.3.3 / LightGBM 4.7.0 / scikit-learn 1.9.1を導入済みです。全関連依存の固定版は [requirements-ml-lock.txt](requirements-ml-lock.txt)。外部通信はこのパッケージ取得だけです。学習データ・モデル・個別予測の保存先は `C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v01` で、既存監査レポートと分かれています。

## 既存Phase 1の利用方法

以下は既存の監査CLIについての説明です。HR監査、返還判定、形式分類、架空データの仮想決済に加え、本人が少数HRとローカル公式公表額を照合できます。監査CLIから学習は行いません。

## 今回の作業と境界

- 既存監査機能の検証は架空データとモックです。Version 0.1の本人による初回実DB実行は別手順です。
- コードの追加・変更はこのプロジェクト内だけです。他のプロジェクトを編集しません。
- 外部API、テレメトリー、クラウド処理、アップロード、自動依存インストールはありません。
- 実DB由来の構造・集計・レポートはすべて `C:\Users\acbd5\AppData\Local\JraRacingAI\audit-reports` に保存します。この保存先は実行時に初めて作成します。
- プロジェクト内 `reports/` は架空結果・非機密資料のみ。実パスワード、実レコード、抽出ファイルをこのOneDriveプロジェクトに置かないでください。
- Gitの初期化・追加・コミット・送信はしていません。`.gitignore` は補助であり、OneDriveの同期を止める機能ではありません。

## 必要環境

WindowsのPython **3.12以上**、PostgreSQL 16、Psycopg 3が必要です。Phase 1のテスト・デモは標準ライブラリで動きます。MLのテスト・デモには上記の追加ライブラリを使います。

Phase 1.1の架空テストでは、既存のOneDrive外の仮想環境の **Python 3.12.10** を使用しました。Psycopg **3.3.6** は利用者申告の導入済み環境です。Codexはドライバーをimportせず、実DBを使わずにテストしています。

```powershell
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
& $taskPython --version
```

`requirements.txt` はPhase 1の互換範囲です。Version 0.1の任意依存は `requirements-ml.txt`、今回の固定版は `requirements-ml-lock.txt` にあります。アプリ自身による自動インストールはありません。

## 架空テストとデモ

次は実DBを使わずに実行できます。`-B` によりbytecodeを保存しません。テストランナーはPythonのsocket接続を禁止します。ドライバーの接続はテスト内の偽物だけです。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
& $taskPython -B tests\run_tests.py
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $taskPython -B -m jra_racing_ai.cli demo
```

デモは説明用の架空データをPython内で集計します。PostgreSQL SQLの実行例ではありません。DB監査モジュールの単体テストは別途モックで検証します。デモ成果物には `source: synthetic` を付けます。

## 利用者自身による実DB実行の準備

1. PostgreSQLを従来の手順で手動起動します。本ツールは起動・停止・サービス登録を行いません。
2. 既存の専用ユーザー `jra_audit_reader` を使用します。新規環境で必要なら[専用読取ユーザーの手順](docs/readonly_user_setup.md)を別途確認します。作成は今回行いません。管理者 `postgres` では本ツールを実行できません。
3. 既存のAppData内の仮想環境を使用します。更新・追加導入は別途承認が必要です。
4. 採用したAppData内の保存先が、ご自身の環境でも同期・バックアップの外部送信対象でないことを確認します。UNC、ネットワークドライブ、junction、symlink、OneDrive配下、プロジェクト内の実結果出力は拒否します。
5. ローカルTCP認証がパスワード認証（SCRAM等）であることを管理者側で確認します。外部認証・外部関数に依存するDBの追加監査は対象外です。

実データ、レポート内容、秘密をチャットや外部サービスへ送る必要はありません。

## 最初は構造だけを監査

以下は**利用者自身が実行するコマンド**です。環境変数は現在のPowerShellセッションにのみ設定します。パスワードはコマンド・ファイルに書かず、非表示プロンプトで入力します。非表示入力ができない端末では平文入力にフォールバックせず停止します。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGDATABASE = 'pckeiba'
$env:PGUSER = 'jra_audit_reader'
& $taskPython -B -m jra_racing_ai.cli schema --execute-local
```

`.env.example` は設定項目の見本であり、自動読込しません。実 `.env` は作成しないでください。既存の `PGSERVICE` / `PGSERVICEFILE` / `PGHOSTADDR` / `PGOPTIONS` に値がある環境は、安全に接続先を固定できないため拒否します。設定一覧やパスワードをチャットへ貼り付ける必要はありません。

接続先は127.0.0.1または::1だけ、dbname/userは単純な識別名だけです。接続文字列・URIは受け付けません。受け継いだpassfileは使用せず、SSL/GSS暗号化の自動交渉も無効にします。libpqの `require_auth` でSCRAM・MD5・password認証だけを許可し、外部認証やtrust認証への迂回を拒否します。この接続オプションに対応しないlibpqでは接続を中止します。実機互換性は導入時に確認が必要です。実行時パスワード入力を推奨します。`PGPASSWORD` の一時利用も可能ですが、他のツール・子プロセスへの継承に注意し、永続設定にはしないでください。

構造監査は `pg_class` 等のカタログを使います。全件COUNT、ANALYZE、インデックス作成は行いません。概算件数には未取得・古い統計・統計リセットの可能性があり、0と不明を区別します。

## 対象年を絞った監査

構造結果をPC上で確認した後、まず2025年・国内10場の集計を行います。

```powershell
& $taskPython -B -m jra_racing_ai.cli audit --execute-local --year 2025
```

既定でレース・馬・オッズを監査します。この既定動作は変更しません。`--sections races`、`--sections horses odds` で処理を絞れます。`--venues 01,02` で競馬場を絞れます。Phase 1.1の払戻監査は `--sections payouts` または `--sections races horses odds payouts` で明示します。

**データ区分列・馬識別列の意味は自動採用しません。** 名前の候補は構造から示しますが、無指定ではその集計を「要確認」として省略します。仕様で確認した実列名を次のオプションで指定できます。

- `--race-status-column <確認済みのjvd_ra区分列>`
- `--horse-status-column <確認済みのjvd_se区分列>`
- `--odds-status-column <確認済みのjvd_o1区分列>`
- `--payout-status-column <確認済みのjvd_hr区分列>`
- `--horse-identity-columns <確認済み識別列>`（複数列はカンマ区切り）

馬識別列と6列レースキーを合わせて重複を数えます。DB主キーを馬の意味上のキーと断定しません。複数版・時点・区分の正当な繰り返しとの区別は要確認です。

6列は前後空白を除いたtextとして比較し、ゼロ埋めを補いません。たとえば整数1と文字列01は同一視しません。列型が数値の場合、国内場の01～10という条件が実表現に合うか確認が必要です。0件でも「データがない」と即断しないでください。現段階では文字列形式に合わせた既知条件を使い、異なる格納形式は要確認とします。

NULLキーは集合比較から分離し、非NULLキー全体の差分と暫定の形式正常キーの差分を報告します。SELECTで個別キーをPythonに返しません。区分7/9の参考件数は2025年・国内10場だけに適用し、差異を欠損と即断しません。

## Phase 1.1 払戻情報の監査

HRの構造も既存のschema一覧に含まれます。最初は区分列を指定せず、件数・キー品質・レース当たり件数分布・RAとの集合比較・構造候補を確認できます。区分を使う集計は「要確認」とします。

次は**利用者自身が確認後に実行するコマンド**です。Codexは実行していません。パスワードは非表示プロンプトで入力し、コマンド・ファイルへ書きません。実レポートをチャットへ転送する必要はありません。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGDATABASE = 'pckeiba'
$env:PGUSER = 'jra_audit_reader'
& $taskPython -B -m jra_racing_ai.cli audit --execute-local --year 2025 --sections payouts
```

RAとHRの区分列がともに `data_kubun` で、仕様の意味を保持しているとPC上で確認した場合だけ、次を実行します。異なる列なら確認した名前に置き換えます。

```powershell
& $taskPython -B -m jra_racing_ai.cli audit --execute-local --year 2025 --sections payouts --race-status-column data_kubun --payout-status-column data_kubun
```

HR区分1/2だけのキーを払戻レコード存在の候補として数えます。HR0の削除、9、未知区分、相反する状態の混在を分け、RAの7実施/9中止との関係を集計します。これは金額・的中対象が正しく展開できることの証明ではありません。HRの存在数、非NULL集合、形式正常集合、状態別の母集団の違いに注意してください。

依頼文で提示されたHR仕様は参照情報です。公式ページは外部通信禁止のため今回未アクセスで、PC-KEIBAの列・変換・配列は要確認です。払戻金の生値・配列・特殊フラグは取得せず、自動分類・金額展開・損益計算はしません。レース当たり複数行を無条件に破損と判定しません。

詳細な集計定義・未確認項目は [hr_mapping.md](docs/hr_mapping.md) にまとめています。HR監査も `--include-global-summary` を指定した場合だけ全期間の年別・期間・キー品質を集計します。

## Phase 1.2 返還対象馬番の判定

`refunds.determine_refund_targets()` は `henkan_umaban_joho` の厳密な28文字と `henkan_flag_tansho` / `henkan_flag_fukusho` を検証し、01～28の返還対象tupleを返します。複数頭を保持し、NULL・空白・不正文字・長さ不一致を補正しません。

フラグと対象の矛盾、フラグ混在（0/1・1/0）は `RefundValidationError`（needs_confirmation）で停止します。混在時の扱いは利用者回答により未確認のままとし、券種別対応を推測しません。馬番照会も厳密な2文字01～28に限定します。

このAPIはDB・ログ・ファイル・ネットワークを使用しません。既存の集計CLIやSQLは変更せず、実レコード取得やレポートへの対象馬番出力を追加しません。通常払戻・特払・不成立・発売なしの決済、返還金額、損益計算とは分離しています。

API契約・架空例・安全なエラー診断・残る確認項目は [refund_targets.md](docs/refund_targets.md) を参照してください。テストは上記の `-B tests/run_tests.py` で実行できます。実結果をrepr以外の方法でログ/レポートに展開せず、ローカルのメモリ内で扱います。

## Phase 1.3 / 1.4 払戻形式分類と仮想決済

`payout_decoder.py` は単勝最大3組・複勝最大5組のa/b/c（幅2/9/2）を構造として解析します。Phase 1.4ではASCII数字と指定幅全体のASCII空白を検証し、未使用の全空白枠・通常形式候補・00/ゼロ値の要確認枠に分類します。NULL/空文字/混在空白/幅不一致を補正せず、複数対象と9桁の先頭ゼロを保持します。正式な意味・金額単位・公式のゼロ埋め/初期値・特殊値・訂正/削除採用は、外部通信禁止で原仕様未照合のためneeds_confirmationです。形式正常でも金額変換ゲートは閉じており、解析・分類結果は決済に使えません。

`settlement.py` は明示した架空の正規化済み入力だけで、100円単位の通常的中・不的中・同着・返還を決済します。特払・不成立・発売なし・中止・未確定・不正は別状態・固定理由とNoneの金額で保留します。最終オッズから金額を推定せず、実HR由来の入力は単位を宣言しても停止します。正規化の既定来歴はHR_UNCONFIRMEDで、実HRの確認ゲートを解除する引数はありません。

重複票と不完全な合計を防ぐ純粋な合計関数も用意しました。月はJSTの開催月を使う契約ですが、実レースの月別集計や損益計算は実行しません。既存CLI・監査SQL・実レポート形式を維持し、実レコード取得や決済コマンドを追加しません。

- [hr_mapping.md](docs/hr_mapping.md)：確認済み構造と、a/b/c・単位・特殊値等の未確認項目。
- [payout_formats.md](docs/payout_formats.md)：Phase 1.4の形式分類、空白/00の区別、根拠の4区分、架空例、ローカル照合手順。
- [settlement_api.md](docs/settlement_api.md)：API契約、架空例、状態・理由コード、次段階の確認ゲート。
- [backtest_policy.md](docs/backtest_policy.md)：仮想決済と事前判断/事後評価・月別集計との境界。

次のローカル操作は、既存venvで架空テストを実行し、仕様書を確認することです。新しい依存はありません。Phase 1.5の少数照合は、以下の専用手順で本人が内容を確認して明示実行できます。本番決済・LightGBM開発は別途確認と承認後に進めます。

## Phase 1.5 少数HRと公式公表払戻のローカル照合

`reconcile-payouts` はAppData内の参照JSONから、同一の6列キー・券種・的中馬番で単勝/複勝を照合します。金額は「そのまま/10分の1/10倍」の未確認仮説で比較し、元の9文字を保持します。同着は各対象を比較し、プラス10も公表額のまま扱います。最終オッズは使用しません。

最初は `--check-input-only` で参照ファイルだけを検証します。`--execute-local` の本人実行時だけ既存の読取専用セッションを使い、既定2025年・国内01～10・最大5レース/40行・SQL5秒/DB取得比較30秒に限定します。接続は127.0.0.1:5432/pckeiba/jra_audit_readerだけです。

実参照JSON・任意のPDFは `C:\Users\acbd5\AppData\Local\JraRacingAI\audit-reports\reconciliation-input` 内だけに配置します。既存保存先検証とリンク/クラウド属性検査に加え、本人が同期対象外と確認して `--confirm-local-unsynced` を指定します。出力は同じ固定ルートの日時フォルダーへ集計だけを保存し、標準出力にも個別キー・馬番・金額・取得行・入力パスを表示しません。

HR区分2だけでは版を承認しません。未レビューの特殊フラグ、00、複勝2着払い、訂正/削除/速報/中止、複数HR、対象集合の差などは保留します。一致しても `unit_confirmed=false`、`production_settlement_enabled=false` は固定です。既存のauditコマンドの集計とSQLは変更しません。

- [payout_reconciliation.md](docs/payout_reconciliation.md)：参照契約、仕様の未確認事項、入力だけの確認から始める本人用PowerShell手順、不一致のローカル目視手順。
- [reconciliation.synthetic.json](examples/reconciliation.synthetic.json)：すべて架空・未確認のひな形。実データ記入前にAppDataへコピーし、プロジェクト側は編集しません。

## Phase 1.5.1 H1・HRの整合性検証

`reconcile-payouts --h1-config <AppData内の設定JSON>` でH1必須モードを追加しました。6列キーでHR/H1を完全照合し、HR区分2・H1区分5を別々に確認します。H1の両発売フラグ7、HRの不成立/特払/返還フラグ0、複勝着払キー2/3と参照条件、独立したH1版レビューが必要です。欠損・複数行・未知値・未確認の訂正/削除は保留し、旧モードへ自動復帰しません。

既存の参照JSONと、H1設定なしのPhase 1.5互換モードは維持します。互換モードの結果をH1検証済みとは扱いません。H1必須モードでは参照JSONの旧HR発売関連ruleを使わず、別設定で確認したHR4フラグ列とH1発売情報を使用します。返還2列・bitmap・既存の金額/形式/対象ゲートも維持します。

最初は [H1・HR手順](docs/h1_hr_reconciliation.md) に従い、AppData側の設定と参照を用意して `--check-input-only` を実行します。SQL5秒の場合は7クエリ分を収めるため `--max-run-seconds 45` が必要です。金額単位の自動確定・本番決済は引き続き無効です。

- [h1-config.synthetic.json](examples/h1-config.synthetic.json)：架空列・架空キー・未確認レビューだけの設定ひな形。

## 全期間の調査と負荷

年限定の抽出では、年・競馬場そのものがNULLや不正で対象条件に入らない行は検出範囲外です。全期間の年別件数・期間・キー品質を確認する場合は、利用者の判断で追加します。

```powershell
& $taskPython -B -m jra_racing_ai.cli audit --execute-local --year 2025 --include-global-summary
```

これは全表走査を伴い得ます。年指定でもcast/trimや索引状態によって全走査となる可能性があります。負荷の低い時間に小さな範囲から実行してください。既定タイムアウトは接続5秒、クエリ60秒、ロック待ち2秒です。必要な場合 `--statement-timeout-ms 120000` 等で1～600秒に調整できます。プログラムはDBの統計更新や索引追加をしません。

各実行は同一のREAD ONLY / REPEATABLE READトランザクションで集計し、終了時にrollbackします。同一実行内で更新による比較の揺れを抑えます。長いスナップショットはDBの保守に影響し得るため、処理を必要な範囲に限定します。タイムアウト時は実行全体を中止し、不完全な結果を正常レポートとして出しません。

通常の組み込みheapテーブルだけを `FROM ONLY` で監査します。継承子は含めません。foreign table、partition親、view、materialized view、RLS有効表、独自access method、独自型は構造だけを記録して要確認とします。これらが中心の場合は追加設計が必要です。

## 成果物と解釈

各実行に新しい日時フォルダーを作り、上書きしません。

| ファイル | 内容 |
|---|---|
| audit.md / audit.json | 全監査レポート・機械可読結果 |
| schema.md / schema.json | テーブル・列・PK・FK・索引・サイズ・概算・統計状態 |
| anomalies.json | 欠損・重複・集合差分を含む監査集計 |
| lightgbm_tasks.md | 各テーブルの適性確認と次Phaseの課題 |

ファイル名・format_version=1・既存のトップレベル項目を維持し、払戻結果は `audits.payouts` に追加します。架空デモにもHRの匿名集計とケース方針を含めますが、実HRの展開結果ではありません。

個別の競走馬名・馬ID・レースキー・オッズ生値・レコードの時点の生値は出力しません。構造監査の `last_analyze` / `last_autoanalyze` はDB統計の更新時刻として含めます。コメント、列default、index expression/predicate、`pg_stats` のサンプル値も取得しません。例外本文・SQL・引数・DSNはログへ出しません。

オッズは名前・型からの候補提示と件数・粒度の集計に留めます。列の倍率、単勝/複勝の格納形式、時点列の意味、履歴保持・上書き、予測時点/最終オッズの区別は **要確認** です。複数行/レースだけでは時系列オッズの存在を証明できません。

HR仕様の一部は依頼文で提示されていますが、仕様原本とPC-KEIBA列定義は今回未照合です。各テーブルについてレース前情報・事後情報・ラベル・特徴量・漏洩リスクを整理しますが、利用可能時点を認定しません。新しいHRのSQL互換性・性能・実レポートは利用者の実行で確認が必要です。

## Phase 2 データ仕様（学習コードなし）

- [phase2_data_contract.md](docs/phase2_data_contract.md)：対象馬、単勝/複勝ラベル、時点、論理データセット、欠損・カテゴリ、開始ゲート。
- [phase2_feature_registry.md](docs/phase2_feature_registry.md)：候補列の状態、利用可能時点、漏洩リスク。未確認列は採用不可。
- [backtest_policy.md](docs/backtest_policy.md)：時系列分割、公式HRでの事後評価、事前オッズでの判断、月間利益・資金制約。

今回レースのHRと最終/確定オッズは特徴量として禁止します。依頼文のO1区分5は確定（月曜）オッズであり、レース前の価格として採用しません。複勝を固定の3着以内、同着の単勝を1頭だけと決めつけません。分割年・具体的資金上限は確認後に決めます。

利用者がローカルのHR監査と対応表を確認した後に、別途承認してLightGBM基準モデルの実装へ進みます。

## STEP 2後のローカル比較

[実行手順と制約](docs/post_step2.md)：固定期間の独立評価、条件付き過去走タイム、時刻付きO1保存、10月18日の事後比較。
既存モデル・朝タスク・事前予測を保持し、結果・モデルはOneDrive外へ新規保存します。

## Stage 1 市場・AI相補性診断

[実行手順と評価条件](docs/stage1_market.md)：追加登録済みの過去O1を使う30/15/10/5分前の被覆率、best242/T=0.75と市場・診断用混合の比較、既存HRによる参考仮想精算。
市場の再較正・混合重みは2025年3～5月だけで選択し、6～8月・既存9～11月期間で固定評価します。2026年1～9月は性能評価に使わず、モデル・既存予測・購入条件は変更しません。
保存RAを基準にした発表時刻ベースの暫定事後評価であり、実取得時刻や当時の発走予定版を保証する前向き実績ではありません。
