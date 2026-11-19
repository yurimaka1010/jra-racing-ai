# Phase 1.3 / 1.4 形式分類と正規化済み架空データの仮想決済API

文書日：2026-10-15（Asia/Tokyo）。API契約は `proposed`。実HRの金額・特殊値・訂正採用は `needs_confirmation` です。

Phase 1.5の [payout_reconciliation.md](payout_reconciliation.md) は独立した仮説照合機能です。照合モジュールは `NormalizedPayout` を生成せず、以下の決済APIを呼びません。一致後も実HR由来の金額/返還決済は停止する契約を維持します。

## 責務と確認ゲート

| モジュール | 今回の責務 | 実データへの経路 |
|---|---|---|
| payout_decoder.py | 列幅2/9/2と最大3/5組の検証、全空白の未使用枠・通常形式候補・00/ゼロ値の要確認枠を分類 | 金額・意味の変換ゲートは閉じたまま |
| refunds.py | 確認済み28文字と単勝/複勝フラグの返還対象判定 | DB取得は行わない。混在フラグは要確認 |
| payout_policy.py | 特殊ケースの将来の実HR決済方針 | 既存のnot_implementedは実HR対応を指す |
| settlement.py | 明示した架空の正規化データを100円単位で決済・合計 | 実HR由来は金額・返還とも停止 |

`parse_payout_layout()` の出力は、意味を持たないa/b/cの文字列です。`ready_for_settlement=False`、`status=needs_confirmation` であり、`settle_virtual_bet()` に渡すと型検証で停止します。

Phase 1.4の `classify_payout_layout()` は、元の枠と9桁の先頭ゼロを保持して形式を分類します。全空白は各幅すべてがASCIIスペースの場合だけ受け付け、NULL/空文字/混在空白は補正しません。a=00かつb/c空白は全空白の未使用枠と区別し、必要な券種別フラグ・成立/発売・レース状態・版の確認を要求します。

`format_status=valid` は形式だけの判定です。分類結果も `ready_for_settlement=False` で、正規化済み入力に変換しません。複数対象を保持しても同着や対象完全性は認定しません。`has_all_slots=True` も全3/5枠を受領した意味だけです。詳細と架空例は [payout_formats.md](payout_formats.md)。この追加は以下の架空決済APIの契約を変更しません。

`decode_hr_payouts()` は構造と重複候補を確認した後、必ず `PayoutDecodingError` で停止します。正式意味・単位・初期値・特殊値・状態/版の規則が未確認で、実HRの正規化結果を生成できないためです。`assume_confirmed` や単位を手動上書きする引数はありません。

## 入力：VirtualBet

| フィールド | API契約（proposed） |
|---|---|
| ticket_ref | 投票を一意にするローカル識別子。1～128文字の空白なしASCII str |
| race_ref | レースのローカル識別子。同じく空白なしASCII str。決済側と完全一致 |
| wager_type | tansho / fukushoのみ |
| horse_number | 厳密な2文字ASCIIの01～28 |
| stake_yen | 正のPython int、100の倍数。bool、float、Decimal、文字列の暗黙変換なし |
| decided_at | timezone付きdatetime。発走時刻より厳密に前であること |

今は架空識別子だけを使います。実用adapterでは、race_refを既存の6列レースキーから衝突なく導出する規則を別途確定し、馬個体IDと当該レース馬番を混同しません。ticket_refの一意性はバッチ全体で保証します。

予測確率・最終オッズ・HR原レコードは購入入力のフィールドではありません。LightGBMの予測・事前判断台帳を作った後、別の決済入力と結びつける設計です。

## 入力：NormalizedPayoutとPayoutTarget

| フィールド | API契約（proposed） |
|---|---|
| race_ref / wager_type | 投票と一致すること。不一致はdata_invalid |
| race_date | JSTの開催日を表すdate。月別帰属の候補で、購入日ではない |
| state | OutcomeStateの列挙値。NORMAL、SPECIAL_PAYMENT、NOT_ESTABLISHED、NOT_OFFERED、CANCELLED、UNCONFIRMED、INVALID |
| origin | 既定はHR_UNCONFIRMED。金額決済は明示したSYNTHETICだけ |
| amount_basis | 架空の正規化データではJPY_PER_100を明示。HRのb列と対応することを意味しない |
| race_start_at / finalized_at | timezone付きdatetime。購入<発走≤確定、JST開催日との整合を検証 |
| revision_state | 架空データのfinalだけ採用。他は停止。実HR区分コードの変換ではない |
| targets | PayoutTargetのtuple。単勝最大3、複勝最大5。馬番の二重登録を禁止 |
| targets_complete | 通常的中/不的中を判定する前にTrueを要求。既定False |
| refund_targets | Phase 1.2の検証済みRefundTargets。Noneを「返還なし」にしない |

PayoutTargetは `horse_number`、正の整数の `payout_yen_per_100`、boolの `is_dead_heat` です。金額は架空の契約上「100円購入時の元本込み総受取額」。HRの9文字の意味や倍率を確定したものではありません。

単勝の複数対象は全対象に同着を明示し、単勝1対象だけの同着宣言は不正とします。複勝では複数対象だけで同着と推定せず、対象ごとのis_dead_heatを使います。複勝対象を3頭に固定せず、明示された完全な対象集合を採用します。このフラグを実HRから生成する規則は未確認です。

originやtargets_completeは呼び出し元の明示契約です。値だけからデータの来歴や真実性を証明する機能ではありません。実レコードをSYNTHETICと付け替えて使用することを禁止します。実データのadapter、確認済みマッピング、正式な決済状態の生成は今回未実装です。

## 出力：SettlementResult

| フィールド | 契約 |
|---|---|
| status | 下表の状態 |
| invested_yen | 入力の投資額 |
| returned_yen | 決済済みなら整数、保留ならNone |
| net_profit_yen | returned_yen−invested_yen。保留ならNone。費用控除はこのAPIに含めない |
| reason_code | 決済済みはNone。保留は固定理由コード。元データを埋め込まない |
| ticket_ref | ローカルな重複防止用。ログ・レポートに出さない |
| race_month_jst | JST開催月。レース不一致・日付不整合等はNoneとして誤った月へ配賦しない |

| 状態 | 払戻額 | 純損益 | 条件/固定理由の例 |
|---|---|---|---|
| normal_hit | 購入単位数×正規化済み公式払戻額 | 払戻−投資 | 完全な通常対象集合に所属 |
| normal_miss | 0 | −投資 | 検証済み・完全・非空の通常対象集合に非所属 |
| dead_heat_hit | 対象固有の公式払戻額×購入単位数 | 払戻−投資 | 対象に同着を明示。別の的中対象を消さない |
| refund | 投資額 | 0 | 架空データの通常状態で検証済み返還対象。既知の払戻対象との重複なし |
| special_payment | None | None | special_payment_rule_unconfirmed |
| not_established | None | None | not_established_rule_unconfirmed |
| not_offered | None | None | not_offered_rule_unconfirmed |
| cancelled | None | None | cancelled_rule_unconfirmed |
| data_unconfirmed | None | None | hr_mapping_unconfirmed / unit_unconfirmed / revision_unconfirmed等 |
| data_invalid | None | None | race_mismatch / duplicate_target / refund_payout_overlap等 |

特払・不成立・発売なし・中止は、返還bitmapがあっても自動的に返還へ置き換えません。通常状態での返還対象は、通常払戻対象の完全性だけが未確認でも、必要な返還・単位・版・時点が確認済みの架空契約なら投資額を返します。通常的中/不的中には対象集合の完全性が必須です。

不正な入力型・賭け金・識別子・時刻などは、入力契約を構成できないため `SettlementValidationError` で停止します。正常な投票契約に対して決済状態が未解決の場合は、固定理由付きの保留結果を返します。例外/保留を非的中や0円へ変換しません。

## 合計と将来の月別評価

`summarize_settlements(tuple_of_results)` は重複ticket_refを拒否し、全票の投資額、決済/保留件数、保留投資額を返します。保留が1件でもあれば、全体のreturned_yen/net_profit_yenはNoneです。空バッチも受取・利益を0とせずNoneにします。

月別集計は、全バッチの一意性を確認した後、race_month_jstでまとめる設計です。Noneの未配賦データを黙って除かず、未解決投資を別に保持します。今回、実レースの月別集計・回収率・資金配分は実行しません。決済入力を特徴量や事前購入判断へ戻してはいけません。

## 架空の使用例

```python
from datetime import date, datetime, timedelta
from jra_racing_ai.refunds import determine_refund_targets
from jra_racing_ai.settlement import (
    JST, JPY_PER_100, DataOrigin, OutcomeState, PayoutTarget,
    VirtualBet, NormalizedPayout, settle_virtual_bet,
)

start = datetime(2025, 1, 5, 12, 0, tzinfo=JST)
no_refunds = determine_refund_targets(
    henkan_umaban_joho="0"*28, henkan_flag_tansho="0", henkan_flag_fukusho="0",
)
ticket = VirtualBet("fake-ticket", "fake-race", "tansho", "01", 300, start-timedelta(minutes=5))
outcome = NormalizedPayout(
    race_ref="fake-race", wager_type="tansho", race_date=date(2025, 1, 5),
    origin=DataOrigin.SYNTHETIC, state=OutcomeState.NORMAL, amount_basis=JPY_PER_100,
    race_start_at=start, finalized_at=start+timedelta(minutes=10), revision_state="final",
    targets=(PayoutTarget("01", 420),), targets_complete=True, refund_targets=no_refunds,
)
result = settle_virtual_bet(ticket, outcome)
assert result.returned_yen == 1260
assert result.net_profit_yen == 960
```

金額・識別子・日付はすべて架空です。実HRのb列を420円と読めることを意味しません。APIは外部入出力を持たず、reprは詳細を隠し、例外診断は固定コードだけです。asdict・属性・tracebackのlocalsを使って実データをOneDrive/ログ/外部へ出さない運用を維持します。

## 次段階のローカル確認と実装手順

1. 公式仕様原本とPC-KEIBA変換資料で、a/b/cの意味・単位・ゼロ埋め・初期値・特殊値を確認する。資料名・版・項番・確認日だけを対応表へ記録する。
2. 全組の物理列、券種フラグ、同着の扱い、レース/馬番対応、取消/除外、混在返還フラグを確認する。
3. 訂正・削除・上書き/履歴の採用方針、決済取消/再計算の一意性を確認する。
4. 確認済み規則ごとの架空テストを追加し、実HR adapterとゲート開放を別途承認する。今回は開放しない。
5. その後に利用者がOneDrive外のローカルで少数の既知ケースを照合する。未対応ケースを含む利益を確定しない。
6. 時点付き特徴量・事前オッズ・ラベルの開始ゲートを満たしてから、別途承認後にLightGBMの実装へ進む。
