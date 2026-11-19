# Phase 2 特徴量候補レジストリ

文書日：2026-10-15（Asia/Tokyo）。文書状態：`proposed`。

各候補は物理列・意味・単位・公開時点・履歴の対応が確認されるまで使用不可です。`confirmed` の列は現時点でありません。ここでのテーブル名は調査先の候補であり、物理列の自動対応ではありません。

Phase 2.0更新：実装済みの論理特徴量は [Version 0.1](version_0_1.md) の `ml.features.FEATURES` です。競馬場・馬番を最小構成とし、本人が確認した識別列があれば過去走特徴を追加します。任意の距離・枠・馬齢等はローカル対応表と `pre_race_fields` で採用し、未確認は欠損。実列の確認状態をソースへ書き戻さず、データセットメタデータへ保存します。現在レースの結果・タイム・HR・最終オッズ・IDは特徴行列から除外します。カテゴリ辞書は学習期間でのみ作成・保存し、未知/欠損を分けて予測します。

状態は `confirmed` / `proposed` / `needs_confirmation` / `prohibited_feature`。[data contract](phase2_data_contract.md)で定義します。前日、当日、直前の許可は固定ルールではなく各prediction policyの時刻t以前に公開済みかで決まります。

| 候補ID | 論理特徴量 | 調査先候補 | 状態 | 利用可能時点の条件 | 漏洩リスク・確認項目 |
|---|---|---|---|---|---|
| race_condition | 競馬場、競走条件・クラス | jvd_ra等 | needs_confirmation | 当該条件のt以前の公表版 | 後日の条件訂正・対象レースの結果列の混在 |
| distance_course | 距離、コース、芝/ダート | jvd_ra等 | needs_confirmation | t以前に公表された設定 | コード定義・単位・変更時点 |
| track_condition | 馬場状態など | 対応表未確定 | needs_confirmation | 各policyのt以前の発表 | 当日の最終状態を前日に遡及して使う危険 |
| past_results | 過去走の確定成績・性能指標 | jvd_se / jvd_ra等 | needs_confirmation | 対象レースより前かつ結果版がt以前に利用可能 | 今回の結果、事後訂正、同日公開遅延 |
| rest_period | 過去出走からの休養期間 | 過去走と開催時刻 | needs_confirmation | t以前に把握した出走履歴 | 最新行参照や日付のみ比較の曖昧さ |
| distance_suitability | 過去距離別の性能 | 過去走 | needs_confirmation | t以前の履歴だけで集計 | 対象/未来レースを含む平均・全期間統計 |
| surface_suitability | 過去馬場・芝/ダート適性 | 過去走 | needs_confirmation | t以前の履歴だけで集計 | 当該レースや未来結果の混入 |
| jockey_history | 騎手の過去成績 | 過去走、jvd_ks候補 | needs_confirmation | t以前に確定した結果のみ | 更新済みマスタの累積値・全期間勝率 |
| trainer_history | 調教師の過去成績 | 実対応表・列未確定 | needs_confirmation | t以前に確定した結果のみ | 人物IDの履歴、事後更新、未知カテゴリ |
| assigned_weight | 斤量 | 出走情報候補 | needs_confirmation | t以前に公表された当該馬の版 | 訂正、単位、最終版しかない可能性 |
| draw_runner | 枠順・馬番 | jvd_se等 | needs_confirmation | 確定・発表後のpolicyでのみ使用 | 前日など未発表の時点への遡及、馬個体キーとの混同 |
| training | 調教 | 実表・列未確定 | needs_confirmation | 実施・公開ともt以前の版 | 公開遅延、事後追記、詳細単位・欠損 |
| pre_race_odds | 時点保存されたレース前オッズ | 履歴表未確認 | needs_confirmation | t以前の公開/取得版、締切前 | 最終値での上書き、区分コード・倍率。初期は購入判断の別入力案 |
| current_result | 今回レースの着順、タイム、通過順等 | jvd_se / jvd_ra結果 | prohibited_feature | レース後 | ラベル情報の直接混入 |
| hr_any | HRの金額・的中馬・フラグ・区分等 | jvd_hr | prohibited_feature | 決済/照合専用 | レース後の的中・返還・中止を予測へ混ぜる危険 |
| o1_category5 | O1区分5の確定（月曜）オッズ | jvd_o1 | prohibited_feature | 依頼文では事後確定 | レース前の価格・期待値として利用禁止 |
| final_odds_any | 最終・確定・払戻由来のオッズ | 各オッズ表 | prohibited_feature | 事後 | 同じ列の過去版がなく当時値を再現できない危険 |
| unavailable_time | 時点が未確認の任意の列 | 全表 | prohibited_feature | 確認できるまで禁止 | 作成日や列名だけで入手可能性を推定する危険 |
| future_aggregates | 全期間勝率、未来を含む平均・カテゴリ統計 | 派生処理 | prohibited_feature | 未来結果を含む | split前の一括fit、対象レースを含むrolling集計 |

## 予測policy別の採用ゲート（proposed）

| 情報 | 前日 | 当日 | 発走直前 |
|---|---|---|---|
| レース条件・斤量・枠順等 | t以前に発表済みなら候補 | 最新のt以前の版を候補 | 最新のt以前の版を候補 |
| 過去走・人物の過去成績 | 結果公開/確定がt以前なら候補 | 同条件 | 同条件。今回の結果を含めない |
| 馬場・直前変更・取消 | 未公開の状態を使わない | t以前の発表のみ | t以前の発表のみ。締切変更もtの版 |
| 調教 | 実施/公開がt以前か要確認 | 同条件 | 同条件 |
| レース前オッズ | 時点履歴がある場合だけ別入力候補 | 同条件 | 締切前の取得可能な版だけ |
| HR・今回結果・確定オッズ | 禁止 | 禁止 | 禁止 |

## 候補をconfirmedに昇格する記録

以下は登録項目の案であり、実レコードや秘密を記載しません。実監査結果はOneDrive外で保持します。

| 項目 | 状態 | 必要な記録 |
|---|---|---|
| feature_id / version | proposed | 論理特徴量IDと計算式の版 |
| physical_source | needs_confirmation | 実表・列・型・単位と対応根拠 |
| specification_evidence | needs_confirmation | 資料名・版・項番・確認日 |
| availability_evidence | needs_confirmation | 公開/取得の時点根拠、履歴保持、訂正時の扱い |
| permitted_policies | needs_confirmation | どの予測policyのtで使用できるか |
| computation | proposed | 過去限定の参照、窓、対象除外、集計条件 |
| missing/category_policy | proposed | 初期値・未知・未公開・不正の扱い、辞書のfit範囲 |
| causal_test | proposed | 未来レコード追加でtの値が変わらないこと、ターゲット混入拒否 |
| review_state | proposed | confirmed / needs_confirmation / prohibited_featureを記録 |

仕様・対応・時点が揃わなければ `needs_confirmation` のままとし、モデル採用リストから除外します。理由のない手動overrideや、実行成功だけを根拠とする昇格は行いません。
