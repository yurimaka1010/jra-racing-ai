# Phase 1.5 実HRと公式公表払戻のローカル照合

文書日：2026-10-15（Asia/Tokyo）。実装・架空テスト済み、実データ照合は未実施です。利用者が明示実行する検証機能です。金額単位の確定、本番決済、実レース損益、学習は有効化しません。

Phase 1.5.1の [H1・HR整合性検証](h1_hr_reconciliation.md) を追加しました。以下はH1設定なしの互換モードの契約です。`--h1-config` 指定時はH1の発売条件とHRの不成立/特払/返還を別々に検査し、専用の版レビューと7クエリの予算を要求します。参照JSONの書式は変えません。

## 根拠と未確認事項

| 区分 | 内容 |
|---|---|
| 利用者提示の公式参照 | [JV-Data4901.pdf](https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf)、Ver.4.9.0.1の「4. 払戻（HR）」・719バイト・区分/各券種フラグ/a/b/c、および「3. 特記事項」のHR・複勝2着払い/00。今回は原本未提供で、Codexによる独立確認なし |
| 利用者確認済みの前提 | JRA公表払戻の基準は100円馬券。これはHRの内部単位の確定とは別。Phase 1.2の28文字返還情報・両返還フラグも維持 |
| PC-KEIBAの観測 | 幅2/9/2、最大3/5枠、9桁数字・空白・00のローカル形式はPhase 1.4を再利用。特定範囲の観測を全年度に一般化しない |
| 要確認 | a/b/c正式意味、通貨・基準額・倍率、初期値、人気順特殊値、全物理列、発売/成立/特払フラグのコード・優先関係、00/2着払い、訂正削除・上書き/履歴、bpchar/取込時の空白保持 |
| 実行保留 | 実HRからNormalizedPayout生成、本番の払戻・損益・返還決済、照合済み規則の自動採用 |

外部通信・参照データの自動取得はありません。公式PDFは利用者が別途保有するローカル原本を、オフラインのPDFビューアーで確認します。`--spec-pdf` は許可されたローカルパス・サイズ・`%PDF-` ヘッダーだけを確認します。本文抽出・内容/版/署名の認証は行いません。指定しただけで仕様確認済みにはなりません。新しいPDFライブラリは不要です。

## 構成と責務

| ファイル | 責務 |
|---|---|
| payout_decoder.py | 既存の厳密な形式分類。金額の意味変換ゲートは閉じたまま |
| payout_reconciliation.py | 参照契約、範囲制限、通常ケースだけの金額仮説比較。外部入出力なし |
| reconciliation_local.py | 安全な参照ファイル読込、少数HRの取得、時間/行数制限、集計だけの保存 |
| payout_reconciliation_cli.py | 本人の明示操作、固定接続先、既存の読取セッション、非表示パスワード |
| cli.py | `reconcile-payouts` を追加。既存audit/schema/demoの挙動は維持 |

元の9文字は `LocalHRRecord` / `RawPayoutSlot` 内に保持します。分類に合格した文字列だけを、照合モジュール内で一時的に整数として比較します。既存デコーダーの金額変換を開放する処理ではありません。取得行・参照明細はメモリ内だけで扱い、一時ファイルへ書きません。

## 参照ファイル契約

[架空ひな形](../examples/reconciliation.synthetic.json) は説明用です。レース・金額・列名はすべて架空で、確認フラグはfalseです。`source=synthetic` のままではDB実行を拒否します。実データを記入する前に、次のOneDrive外へコピーしてください。

```text
C:\Users\acbd5\AppData\Local\JraRacingAI\audit-reports\reconciliation-input\reference.json
```

UTF-8（BOM可）のJSON、最大256KiBです。自由記述・未知キー・JSONの重複キー・NaN/Infinityを拒否します。公式ページのURL、HTML、認証情報、オッズのフィールドはありません。

| トップレベル項目 | 契約 |
|---|---|
| format_version | 整数1 |
| source | `synthetic` / `jra_official_manual`。後者は利用者の手動参照という来歴宣言で、自動認証ではない |
| official_base_yen | 整数100。HR内部単位を意味しない |
| source_reviewed_locally | 公式公表を本人が確認・転記照合したらtrue。未確認のままtrueにしない |
| abc_mapping_reviewed_locally | a/b/cの意味・全物理列対応を公式原本/PC-KEIBA資料で本人が確認したらtrue。bの金額単位の確定とは別 |
| status_column | 意味を確認した実HR区分列。候補名を自動採用しない |
| flag_rules | 単勝/複勝ごとの不成立・特払・発売なしの対応。未確認なら空配列、またはreviewed_locally=falseで保留 |
| cases | レース×券種ごとの参照ケース。最大40件、最大20レース（既定上限5） |

各caseの `race_key` は既存の6列すべてを、幅4/4/2/2/2/2のASCII数字文字列で明記します。日付の実在・国内01～10・開催回/日目/レース番号の非00を検査します。公式ページのIDを単純に分割せず、PC-KEIBAの6列との対応を本人が確認してください。整数をゼロ埋めして同一化しません。

| case項目 | 契約 |
|---|---|
| wager_type | `tansho` / `fukusho` |
| state | `normal` / `refund` / `not_offered` / `not_established` / `special_payment` / `cancelled` / `unconfirmed` / `invalid` |
| revision_review | `unconfirmed` / `snapshot_reviewed_locally` / `preliminary` / `correction` / `deleted` |
| dead_heat | bool。単勝の複数対象ではtrue必須。複勝の複数対象から同着を推測しない |
| place_paid_positions | 単勝はnull。複勝は2/3またはnull。2は専用理由で保留、nullも条件未確認で保留 |
| targets | 通常は公表された全対象。単勝最大3、複勝最大5。特殊状態は空配列で、通常対象の金額を混在させない |

各targetは `horse_number`（01～28の2文字）、`payout_yen_per_100`（正の整数）、`plus10`（true/false/null）です。同一馬番の二重登録は同額でも拒否します。同一レース×券種の重複、同一レースの相反する版確認も拒否します。

`plus10=true` は公表でプラス10を確認したという注記です。金額に10円を足す処理ではありません。nullは未確認の注記として残し、公表額自体を比較します。最終オッズから正解金額・プラス10の有無を作りません。

### 券種別フラグの明示対応

各ruleの `wager_type`、`kind`（`not_established` / `special_payment` / `not_offered`）、実 `column`、`normal_value`、`active_value`、`reviewed_locally` を記入します。値は1～4文字のASCII数字/空白の厳密な文字列です。空白を除去しません。

通常比較には、その券種の3種類の意味がすべてレビュー済みで、取得値が明示した通常コードであることが必要です。activeコードはそれぞれ別状態で保留、その他の値は未知として保留します。実際に存在する列とコードを確認するまで、ひな形の架空列・0/1を実規則として使わないでください。

一つの実列が不成立/未発売など複数のコードを持つ場合、同一券種内で明示的に列を共有できます。その場合は通常コードを統一し、activeコードは異なる値とします。例として架空の同一列に通常0、不成立1、未発売2という対応を指定できますが、これは公式コードを認定した例ではありません。コードを表現できない場合は架空の列を補わず、要確認のままとします。

返還は既知の `henkan_umaban_joho`、`henkan_flag_tansho`、`henkan_flag_fukusho` をPhase 1.2で検査します。フラグ混在・不正bitmap・払戻対象との重複は停止します。別の馬の返還があるレースでも、条件が確認できた通常的中対象は比較できます。返還対象馬券そのものの金額比較・返還決済は行いません。

## 比較条件と集計単位

同一の6列キー・券種・的中馬番で比較します。通常対象集合が参照と完全一致しない場合、ケース全体を照合不能とし、余剰のHR対象数も別掲します。未参照の対象を黙って除いて一致率を上げません。同着は全対象を個別比較します。

| 金額仮説 | 比較式 |
|---|---|
| raw_integer_as_yen_per_100 | 9文字の整数をそのまま円/100円と仮定 |
| raw_divided_by_10_as_yen_per_100 | 整数を10で割ると仮定 |
| raw_times_10_as_yen_per_100 | 整数を10倍すると仮定 |

いずれも未確認の仮説です。浮動小数点・四捨五入を使わず、整数の分子/分母で比較します。割った結果が整数円でなければ `fractional_yen_hypothesis` で照合不能です。9文字の全ゼロは0円へ変換しません。

一致・不一致・照合不能は **券種×仮説×参照の通常的中対象** の件数です。一致率の分母は一致＋不一致で、分母0ならnullです。別に、対象なしの特殊ケースを含む要求ケース・比較可能ケース・保留ケース・固定理由・形式・返還あり・同着・2着払い・プラス10注記の件数を残します。理由件数は主にケース単位で、`fractional_yen_hypothesis` だけは対象×仮説単位です。

速報1、削除0、中止9、未知区分は保留します。HR区分2でも、ケースの `snapshot_reviewed_locally` がなければ採用しません。同じキーが複数行なら全件を保留し、区分・日時の最大で選びません。PC-KEIBAが旧版を上書きした場合、単一行という観測だけで訂正履歴がないとは証明できません。`snapshot_reviewed_locally` は今回比較するスナップショットを本人が確認したという宣言で、将来の訂正採用規則を承認しません。

00・複勝2着払い・ゼロ値・不成立・特払・発売なし・中止・返還対象馬券は通常的中/不的中に置き換えません。この版で特殊規則は解決せず、保留件数として残します。将来のルールは年・券種・仮説・確認条件を別に記録します。現在の集計JSONも年/券種/仮説別の証拠ですが、`unit_confirmed=false`、`production_settlement_enabled=false` は固定です。一部サンプルの一致だけで他年・全券種へ適用しません。

## 保存先と取得制限

参照JSON・任意の公式PDFは、固定ルート `C:\Users\acbd5\AppData\Local\JraRacingAI\audit-reports` の直下の `reconciliation-input` にある通常ファイルだけを受け付けます。集計は同ルート内の新しい日時フォルダーへ `reconciliation.json` / `reconciliation.md` として保存し、上書きしません。

既存の保存先検証に加え、UNC/ネットワークドライブ、相対パス、OneDrive名/環境変数の同期領域、project内、symlink/junction/reparse/cloud placeholder、hard linkを拒否します。その他の同期・バックアップアプリの設定すべてを自動検出することはできません。本人が対象外と確認して `--confirm-local-unsynced` を指定します。監査中にフォルダーや入力ファイルを移動・置換しないでください。

`.gitignore` も参照/照合ファイル名を補助的に除外し、架空ひな形だけを例外にします。Git除外はOneDrive同期を止めません。実入力・実結果をプロジェクト内へ配置する運用は認めません。

| 制限 | 既定 / 最大 |
|---|---|
| 対象年・場 | 2025年 / 国内01～10。別年は明示指定、1実行1年 |
| レース数 | 5 / 20。参照にある全レースを照合し、勝手に先頭N件へ切らない |
| HR行数 | 40 / 200。LIMITとfetchmanyで上限+1を検知し、超過時は実行を中止 |
| SQLタイムアウト | 5秒 / 10秒、接続5秒、ロック待ち2秒 |
| DB取得・比較の予算 | 30秒 / 120秒。接続＋4クエリの最悪タイムアウトを予算内に制限し、各クエリ前後・比較後も期限検査 |
| 参照JSON | 256KiB / 40ケース / 20レース・最大160通常対象 |

上記4クエリは既存セッションの権限検査1、HR relation確認1、HR列確認1、選択HR取得1です。権限/カタログも同じREAD ONLY / REPEATABLE READ接続で実行します。入力編集・パスワードの手動入力・ローカルディスクへの保存時間はDB実行予算に含みません。通常heap・RLSなし・組み込み文字列型の列だけを許可します。整数キーや数値金額列は自動変換せず、対応要確認で停止します。

キー条件・LIMITはパラメーター化し、HR区分で行を除外せず訂正/削除等を見落とさないようにします。金額等をtrim/castせず、超過長のフィールドは元の巨大文字列を取得せず拒否マーカーとNULLに置き換えて保留します。NULLを正常値へ補正する処理ではありません。対象年が少数でもキー索引等によっては全走査になり得るため、既存のタイムアウトを守ります。索引作成・ANALYZE・更新SQLはありません。

標準出力/エラー/集計ファイルは、固定ステータス、件数、形式別件数、照合率、固定理由コード、年/券種/仮説等の条件だけです。個別キー・馬番・金額・参照パス・SQL・取得行・パスワードを表示しません。reprも詳細非表示です。明細ダンプ・debug・CSV出力・tracebackのlocalsを保存する機能はありません。

## 利用者本人が行う最初の手順

以下は**本人が確認して実行する手順**です。Codexは実行していません。既存venvを使用し、インストール・DB更新は不要です。

1. まずDBなしのテストを実行します。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
& $taskPython -B tests\run_tests.py
```

2. AppDataの固定ルートが同期・外部バックアップ対象外であることを確認し、本人が入力フォルダーを作成・架空ひな形をコピーします。既存の実参照ファイルへ上書きしない手順です。

```powershell
$localReferenceDir = Join-Path $env:LOCALAPPDATA 'JraRacingAI\audit-reports\reconciliation-input'
New-Item -ItemType Directory -Path $localReferenceDir -Force | Out-Null
$localReferenceFile = Join-Path $localReferenceDir 'reference.json'
if (Test-Path -LiteralPath $localReferenceFile) { throw '参照ファイルが既に存在します。上書きを中止しました。' }
Copy-Item -LiteralPath '.\examples\reconciliation.synthetic.json' -Destination $localReferenceFile
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $taskPython -B -m jra_racing_ai.cli reconcile-payouts --check-input-only --confirm-local-unsynced --reference $localReferenceFile
```

この入力確認はDB・ドライバー・パスワード・出力ファイルを使用しません。`source_is_synthetic=true` の架空ひな形として検証し、確認済みフラグ・版レビューの件数も表示します。形式が通っても、レビュー未確認なら比較準備完了ではありません。実キー・払戻額をプロジェクト側のひな形へ書かないでください。

3. オフライン資料と、ご自身がローカルで保有する公式公表払戻を確認し、**AppData側のコピーだけ**を編集します。最初は同じ少数レースの通常単勝・通常複勝を両方用意し、全対象を転記・二重確認します。架空の列名/キー/対象/金額をすべて置き換え、確認できた範囲だけsource・レビュー状態を更新します。未確認のフラグはtrueにしません。PDFは任意で同じ入力フォルダーへ配置し、本文を本人が確認します。

```powershell
& $taskPython -B -m jra_racing_ai.cli reconcile-payouts --check-input-only --confirm-local-unsynced --reference $localReferenceFile
# 任意: --spec-pdf (Join-Path $localReferenceDir 'JV-Data4901.pdf')
```

4. 入力と列対応を確認後、従来の方法でPostgreSQLを起動し、既存 `jra_audit_reader` で本人が明示実行します。専用ロールの権限は既存の実行時ガードで再検査します。パスワードは非表示プロンプトで入力し、履歴/ファイルへ書きません。

```powershell
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGDATABASE = 'pckeiba'
$env:PGUSER = 'jra_audit_reader'
& $taskPython -B -m jra_racing_ai.cli reconcile-payouts --execute-local --confirm-local-unsynced --reference $localReferenceFile --year 2025 --max-races 5 --max-hr-rows 40 --statement-timeout-ms 5000 --max-run-seconds 30
```

`PGSERVICE` / `PGSERVICEFILE` / `PGHOSTADDR` / `PGOPTIONS` 等の既存の拒否条件を維持します。localhost・別ポート・別DB・別ロールもこの照合CLIは拒否します。ライブラリ未導入時は停止し、自動インストールしません。

5. AppData内の最新の `_reconciliation` フォルダーで集計を確認します。不一致/照合不能があれば単位確定を保留します。少数の通常ケースが一致しても確認済みルールを登録したり本番決済を実行したりしません。

### 不一致の原因調査をローカルで完結する

このCLIは個別の不一致キーや金額を出しません。本人が参照JSONの原ケース一覧を使い、A5 Mk-2で同じ6列キーのHRを少数だけ画面表示し、原9文字・a/c・各フラグ・複数版とローカルの公式表示を目視で照合してください。画面内で、レース/券種/馬番の取り違え、対象集合、文字の保存長、プラス10を含む公表額、訂正時点を順に確認します。取得結果のCSV・スクリーンショット・SQLログをOneDriveへ保存したりチャットへ貼ったりする必要はありません。

解決後も記録するのは、OneDrive外の非公開メモと、必要なら非機密の資料名・版・項番・規則です。原レコードをプロジェクトへ持ち戻しません。実HR決済ゲートの変更は、条件ごとの追加確認・架空テスト・別承認の後です。
