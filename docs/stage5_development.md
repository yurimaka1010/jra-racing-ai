# Stage 5：保存予測のAUCと同時点複勝価格の有限検証

承認済み指示文と補足3点を実装します。Hを固定比較対象とし、既存最高AUCのモデル名・数値は参考比較で報告します。2着払い・3着払いをO1の着払キーで区別し、AUCの改善と総合採否を分けます。既存コード・旧run・朝7時タスクを変更しません。

## 今回の保存先と再確認

`C:\Users\acbd5\AppData\Local\JraRacingAI\stage5-development\evaluations\598c62b4dbe54142a2abe4a98b435b12`

```powershell
Set-Location 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai'
$stage5Python = 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe'
$stage5Run = '598c62b4dbe54142a2abe4a98b435b12'

# 完了済みrunのハッシュ・旧成果物・タスク設定を検証。DBや学習を再実行しません。
& $stage5Python -B '.\scripts\stage5_local.py' --stage report --run-id $stage5Run
```

## 処理順序

新しい実験を追加するための承認ではありません。今回runの中断再開は以下の順序です。各完了単位はハッシュ検証して再利用します。

```powershell
& $stage5Python -B '.\scripts\stage5_local.py' --stage baseline --run-id $stage5Run

# 未取得の同一T−10レコードがある場合だけDBへ読取専用接続します。
# 既存DPAPI資格情報を同じWindowsユーザーで読み、パスワードは出力しません。
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -NonInteractive -ExecutionPolicy Bypass -File '.\scripts\stage5_local.ps1' -Stage place -RunId $stage5Run

& $stage5Python -B '.\scripts\stage5_local.py' --stage fit --run-id $stage5Run

# 全テストは実DBドライバーとソケットを禁止した架空データ・モックのみ。
$env:TEMP = 'C:\Users\acbd5\AppData\Local\JraRacingAI\synthetic-tests'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
& $stage5Python -B '.\tests\run_tests.py' *> '.\reports\stage5_regression_tests.txt'
& $stage5Python -B '.\scripts\stage5_local.py' --stage report --run-id $stage5Run
```

`init`は新規専用runを発行して計画・入力ハッシュ・タスク設定ハッシュを保存する実装用入口です。新runを作成して今回の学習上限をリセットすることは認めません。旧Stage 2〜4のrun/fit/freezeは呼ばず、副作用のない保存読込だけを使います。

## 指標・比較契約

- 2023・2024・2025年の各1〜9月、Stage 4と同じ6,373レース・87,601頭。勝ちラベルは保存winner_labelだけ。全比較のキー・馬番・ラベルを照合します。
- スコアは保存最終勝率。温度・offset・正規化を二重適用しません。全体ROC-AUCの点推定はsklearn.metrics.roc_auc_score、同点0.5。全体AUCは年別やレース別AUCの平均ではありません。
- 週クラスタは月曜起点・全競馬場共通、5,000反復、seed20261019、95%区間。同じ再標本をAUC・LL・Brier・予測1位に使います。
- 計算を高速化するため、正例週×負例週ごとの順位比較対数を事前計算します。各再標本の週多重度wに対して`w.T @ U @ w / ((w @ positive_count)*(w @ negative_count))`を計算します。週別AUCの平均ではなく、重複行で全体AUCを再計算することと厳密に等価です。sklearn標本重み・重複行展開・手計算との一致をテストします。
- レース内AUCは勝馬対他馬の順位比較率をレース等重み平均。最高勝率同点k頭に勝馬を含めば1位的中は1/k加点です。

## 複勝の固定追加

公式PDF物理15頁（印刷12頁）のO1と物理33頁（印刷29頁）の特記事項を使用します。元PDFは読取専用です。

- `odds_fukusho`は336文字、28組×12文字＝馬番2・最低4・最高4・人気2。4桁数値を10で割る価格です。
- `fukusho_chakubarai_key`は2＝2着まで、3＝3着まで。頭数から推測せず、両区分の被覆率・欠損を別記します。未知キーは欠損です。2/3を新しい特徴量として追加しません。
- 原則同じO1：既存T−10選択の完全キーと公表分に一致するレコードのみ取得します。既存単勝オッズとの一致も検証し、別時刻・確定オッズは使いません。300秒・公表分終端・区分1・同一馬番集合を維持します。
- 未投票・取消・NULL・不正・価格の打切りは欠損。0999の旧仕様記述は曖昧さが残るため、9999とともに保守的に欠損化します。推測した正確な価格として使いません。
- 追加は`log(win)-(log(lower)+log(upper))/2`、`log(upper)-log(lower)`、レース全体の欠損フラグだけ。価格相対値であり確率ではありません。1頭でも不足なら全馬の追加数値NaN・フラグ1。主母集団を減らしません。
- PC実取得時刻・予定発走時刻の完全なas-of性は未確認のままです。今回も公表時刻を使う事後の開発評価です。

各年のtrain/validation/calibration/evaluationすべてで全馬完備レース80%以上、学習数値に変動がある場合だけ、H_placeを年別各1本・最大3本学習します。同じH8列、anchor_parameters、logit初期値、sigmoid、レース内正規化、T=1、300ラウンド・早期終了20を維持します。新規OOF・再較正・代替実験・ROI分析を行いません。

## 採否と保護

H_place−Hだけを主要比較とし、未丸めΔAUC≥+0.002・95%区間下限>0・改善年2以上・年間−0.002以下の悪化なし・プールのレース内AUC/1位的中率がH以上・レースLL/BrierがH以下・整合性確認、の全条件が必要です。最高既存モデルの更新や0.80超えでこの基準を置き換えません。

AUC点推定が改善しても採否条件を満たさなければ、「AUCは改善したが総合採用基準未達」と区別します。区間が0をまたぐ場合は改善が不確実です。既参照の開発評価であり、独立した最終評価や本番優越の証明ではありません。

実データ、複勝原列、モデル、詳細予測は上記OneDrive外runだけに保存します。プロジェクト側にはコード・架空テスト・説明だけを置きます。2026年の取得・学習・特徴量生成・評価・最終評価入口、DB書込み、タスク変更、実馬券購入、外部送信、新規パッケージ導入は行いません。
