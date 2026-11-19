# LightGBM Tuner：準備済み・実データ学習は別承認

公式 `optuna_integration.lightgbm.LightGBMTuner` を使う独立した二値分類の比較です。
既存ABCD 39列、既存objective-pilotの4分割をハッシュ検証して再利用します。
主比較は同条件の `paired_binary_control`、既存Hは学習期間の異なる参考比較です。
2025年は再利用済み開発評価です。2026年封印期間・Stage 7成績・DB・外部通信を使用しません。

学習計画案は公開APIの `tune_feature_fraction(2)`、`tune_num_leaves(2)`、
`tune_bagging(2)`、`tune_regularization_factors(2)` の順で最大8試行。
上限を明確にするため `run()` は呼びません。
実際の `lightgbm.train` 呼出を包み、失敗・中断も学習1回と数えます。
全段階のbest boosterを再利用し、最終再学習は0回です。

探索の指標は早期終了期間のbinary_logloss、最終比較の主指標はレース勝馬Log Loss。
温度は別の較正期間で [.75,1,1.25,2] から選び、評価データを読む前に保存します。
各300ラウンド・early stopping20・CPU4、探索10分・全体20分、親と実行子孫のprivate memory12GiBを上限とします。
探索の時間停止は各ラウンドcallbackとTuner time_budget、全体上限は親プロセス監視です。

```powershell
# 0fit。ライブラリと公開メソッドを確認
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B '.\scripts\run_tuner_pilot.py' compatibility
# 0fit。新規計画・入力ハッシュ・学習予算案をローカルに保存
& "$env:LOCALAPPDATA\JraRacingAI\venv\Scripts\python.exe" -B '.\scripts\run_tuner_pilot.py' prepare
```

保存先：`%LOCALAPPDATA%\JraRacingAI\tuner-pilot\evaluations\<ID>`。
`plan_commit.json` に計画ハッシュと必要な承認レコードの形式を保存します。
`approval.json` に本人の別途承認（allow_real_fits/source/plan_sha256/actual_train_calls_max）がないexecuteは必ず停止します。
準備処理から承認JSONを作ったり、自動承認したりしません。

承認後のexecute経路は、共通の `research-training.lock` を排他作成し、高負荷研究の同時実行を防ぎます。
同じ承認計画ハッシュを別runから再利用する実行、同じrunの再実行、workerの再入場は排他マーカーで拒否します。
workerも計画消費マーカーと所有中の研究ロックを確認します。既存ロックの自動解除は行いません。
実行される場合、モデルJSONに39列順序・カテゴリ辞書・計算式版・入力形式・較正・選択パラメータ・依存版を保存します。
再読込後の確率一致と同じ評価ロスターを確認し、週クラスタbootstrap1,000回の差を出力します。
二値対照・調整モデル・Hを同じ確率帯で較正表にもまとめます。
実際のlightgbm.train呼出合計時間と探索全体時間を分けて保存します。
既存対照のworker時間はCSV・較正・推論を含むため、その値と純粋な学習速度を直接比較しません。
確率品質の比較でありROI・購入条件・本番モデル昇格は扱いません。

準備完了は実データ学習・性能比較の完了を意味しません。
