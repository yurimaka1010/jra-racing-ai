# Stage 1: 発走前市場・best242相補性診断

モデルを再学習せず、既存best242 (`a8178c2b7523499f9463c1743b41c715`) とT=0.75を固定するローカル研究用CLIです。Stage 2/3の本番実装・実馬券購入・タスク登録は行いません。

## 入出力と秘密保護

- DB: `127.0.0.1:5432/pckeiba` / `jra_audit_reader`。既存の接続・SQL検証を通し、REPEATABLE READ・読み取り専用・SQL60秒上限です。
- 資格情報: 既存のWindowsユーザー別DPAPIファイルをPowerShell内で読み、子プロセスに限って環境変数で渡します。コード・ログへ値を出しません。オフライン段階はPythonコマンドだけで実行可能です。
- 保存先: `C:\Users\acbd5\AppData\Local\JraRacingAI\stage1-market\evaluations\<work-id>`。既存の同期・リンク拒否を再利用します。
- `*.local.json`にはレースキー、予測、選択したO1原形式、HRなどが含まれます。PC内でのみ扱い、チャットやGitへ貼りません。
- `report.md`、`results.json`、`coverage.json`、`selection.json`、`verification.json`は匿名集計です。外部共有は利用者が別途判断します。
- 既存データセット・モデル・10月17～18日予測には書き込みません。モデルファイル・予測CSVのハッシュと既存3期間の精度再現を確認します。

## 時刻・母集団の契約

1. 実DBカタログからO1の候補表・列・型を確認します。既存O1デコーダーで扱える物理列を持つ表だけを候補とし、未知の表を勝手に解釈しません。
2. 公式O1区分1だけを発走前候補とします。2=前日売最終、3=最終、4=確定、5=月曜確定を事前価格に置換しません。0=削除、9=中止や未知状態が混在する場合は保留します。
3. `happyo_tsukihi_jifun`はJRA発表月日時分です。データ作成日、PC-KEIBAへの追加登録日、今回の読込時刻を当時の実取得時刻にしません。`toroku_tosu`は登録頭数であり時刻列ではありません。
4. 保存RAの発走予定時刻を基準に30/15/10/5分前を比較します。変更前時刻が非初期値なら除外。残りも当時のRA版を証明できないため、**保存RAを基準とする暫定的な事後評価**です。実運用で同時点に購入できた実績とは扱いません。
5. 分精度の発表時刻は、その分の末尾が判断時刻以前であることを要求します。判断時刻と同じ分の記録は採用しません。今回の10分前評価で11分前の表示分を使う場合があるのは、この保守的な判定によります。
6. 最新候補が不正・不完全・矛盾なら古い正常価格へ戻りません。鮮度上限300秒を精度を見る前に固定し、時刻差と除外数を保存します。
7. AIと市場は同一レース・同一馬番集合。取消馬は既存デコーダーで明示的に処理できる集合外スロットだけ除き、集合内の欠損・取消はレースを除外します。
8. 2025年の対象はSTEP 2と同じ完全着順・単独勝馬・基本特徴量充足の母集団です。全開催レースや取消・中止・同着を含む運用成績ではありません。2026年10月17～18日は保存予測の出走馬集合で被覆率だけを確認します。

## 設定選択と評価

- 開発: 2025-03-01～05-31。市場温度と混合重みを馬単位二値Log Lossで選択。
- 確認: 2025-06-01～08-31。設定は固定。
- 既存最終期間の再利用: 2025-09-01～11-30。過去に評価済みなので、新しい未使用テストとは呼びません。
- 2026-01-01～09-30は性能・収益を計算せず保持。収録件数を見ることと、正解ラベルでモデルを選択することを区別します。
- 市場確率は `(1/odds)/sum(1/odds)`。市場温度候補は既存と同じ0.75/1/1.25/2。
- 算術混合: `w*p_AI + (1-w)*p_market`。AI重み候補は0/0.05/0.1/0.2/0.3/0.5/0.75/1。
- 市場rawと市場再較正後の双方を混合の基準にし、単なる市場校正の改善をAI追加効果と混同しません。幾何混合は診断用の追加比較です。
- best242のT=0.75は1回だけ適用し、その後の比較・仮想購入はidentity確率として扱います。
- 既存の馬単位LL・Brier・勝馬LL/R・同率按分1位的中率・校正帯・レース単位paired bootstrapを再利用。一致/不一致集計と仮想購入は既存の馬番最小決定規則を使います。
- 開発期間の選択結果を`selection.json`へ排他的に保存してから、別期間の混合予測・成績を計算します。開催日相関や探索回数の補正は未実装です。

## 参考仮想精算

既存の `core.run_simulation`、HRデコーダー、月次集計を再利用します。単勝100円・EV5%以上・最大EV1点を固定。判断時点のオッズから購入を決め、通常確定HRの円/100円だけで精算します。HRと既存SEラベルの勝馬が一致しなければ、その精算だけ保留します。特殊払戻・返還・複数版・未確定は既存の固定理由で保留します。

raw市場確率のEVでは、逆数オッズ合計が1を超える通常の市場なら全馬が見送りになり得ます。購入0件の回収率は未算出です。混合重みを収益で選び直しません。確定オッズは値動きの参考比較にだけ使用します。

## PowerShell実行

既存プロジェクトフォルダーで実行します。インストールは不要です。

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$taskPython='C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe'

# 新規の収録調査。出力のwork_idを次のコマンドへ渡します。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\stage1_local.ps1' -Stage inventory
$taskWorkId='<表示された32桁のwork_id>'

# 保存済みモデル・入力だけを使用。DB接続なし。
& $taskPython -B '.\scripts\stage1_local.py' --stage prepare --work-id $taskWorkId

# 対象キーを32レースずつ読取専用で取得。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File '.\scripts\stage1_local.ps1' -Stage collect -WorkId $taskWorkId

# 以下はオフライン。毎回新しいanalyses/<analysis-id>へ出力します。
& $taskPython -B '.\scripts\stage1_local.py' --stage evaluate --work-id $taskWorkId
& $taskPython -B '.\scripts\stage1_local.py' --stage report --work-id $taskWorkId
```

`report`は最後に完了した解析を使用します。既存レポートを上書きしないため、レポート作成済みの解析で同じ`report`を再実行しないでください。別の解析は`evaluate`から新規作成できます。`prepare`と`collect`は一度完了した収集フォルダーには重ねて実行せず、新しいinventory/work-idを使います。

## Stage 2への引き継ぎ

保存済みRA/SE構造では、`grade_code`、`kyoso_joken_code`、`kishu_code`、`chokyoshi_code`、`chakusa_code_1～3`、`kohan_3f`、`corner_1～4`が存在します。列の存在と、欠損率・原形式・当時の利用可能性の確認は別です。

最小の追加候補は、既存の過去着順・頭数・日付による頭数正規化と近走重視、次にクラス・条件です。着差・上がり・通過順は過去走だけを使い、形式確認後に追加します。騎手・調教師の履歴成績は対象馬限定履歴から計算せず、全騎乗・全出走を母数にする必要があります。
