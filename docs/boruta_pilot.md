# Boruta：既存ABCD能力モデルの特徴量選択準備

新しい券種研究系統を追加せず、既存win/ABCD39の選択実験として準備します。
実データ選択・RandomForest学習・LightGBM学習はまだ実行しません。
`boruta_pilot.compatibility()` は版・API・予算案を0fitで返します。

計画案はRF64木・深さ5・CPU4、Boruta max_iter=4、実際のforest.fit最大3回。
インストール済Boruta0.4.3はiter=1で始まり `iter < max_iter` のため最大3回です。
失敗・途中停止も実fit枠を消費し、同じ承認の再実行は排他マーカーで拒否します。
後段のLightGBM1fitは別予算・別承認で、本モジュールは学習しません。
3反復は予算を限定した動作確認であり、十分な特徴量選択を保証する設定ではありません。

前処理API：

- `make_preprocessor(train)`：2022-01-01〜2024-08-31の訓練行だけで辞書と中央値を作る。
- `transform(frame,preprocessor)`：辞書を固定してone-hot、数値を訓練中央値で補完。
- `save_preparation(directory,preprocessor,selection)`：辞書・中央値・マスク・元特徴量対応をJSON保存。
- `result_masks(selector,preprocessor)`：confirmedとtentativeを分離。ライブラリrough-fixの重複を昇格しない。
- `selected_matrix(...)`：confirmedのみが既定。空なら停止し、全項目へ戻さない。

全件欠損の元列は除外し、0等を仮定しません。学習・予測の列順序とカテゴリ定義は既存matrixを利用します。
選択はone-hot列単位です。`confirmed_source_features`は「一部の展開列がconfirmed」の対応情報であり、元特徴量全体の自動採用ではありません。
後段をone-hot選択行列で学習する比較は前処理も変わるため、既存の生39列モデルとの差をBorutaだけの効果と断定できません。
公平な選択効果の検証には同じ前処理の対照を用意するか、元特徴量へ戻す方針を実行前に固定します。

`select_training()` は本人の明示承認JSON相当（source/human_user、proposal_sha256、allow_selection=true、actual_forest_fit_max=3）と、
当該runが同一プロセスで保有する `research-training.lock` がなければ選択前に停止します。
コードから承認を自動生成しません。学習時の個別値やエラー本文を出力しません。
既存H、朝7時タスク、Stage7、2026年封印期間は利用・変更しません。
