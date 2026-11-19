# Phase 1.4 HR払戻フィールドの形式分類

文書日：2026-10-15（Asia/Tokyo）。形式分類は実装済みです。正式な意味・金額単位・特殊値・訂正削除の採用は `needs_confirmation`、実HRの金額変換は実行保留です。

Phase 1.5では、この形式分類器を変更せず [ローカル照合](payout_reconciliation.md) を追加しました。分類に合格した9文字を別モジュール内で金額仮説として比較します。形式分類器・実HRデコーダーの金額変換ゲートと既存の架空決済は維持します。以下の「取得処理なし」「int変換なし」は本形式分類APIの責務です。

## 根拠を分ける

| 区分 | この版で扱う情報 | 制限 |
|---|---|---|
| JRA-VAN公式仕様で確認済み | 新たに原本照合できた事項はなし。Phase 1.2の28文字返還情報と0/1フラグは、利用者による仕様確認結果を維持 | Codexによる独立した原本確認とは区別 |
| PC-KEIBAのローカル集計で確認済み | 利用者から、単勝最大3枠・複勝最大5枠、a/b/c幅2/9/2、bに9桁数字または空白、a=00かつb/c空白のパターン、複数の追加対象の存在が提示された | 特定の年・範囲の観測。全年度の保証や公式の意味付けには使わない。実集計の件数・実レコードは本資料に転記しない |
| 仕様と照合が必要 | a=的中馬番、b=払戻金、c=人気順という対応候補、全空白/全ゼロの初期値、ゼロ埋め規則、00の意味、同着、券種別フラグ、人気順の範囲 | 正式な資料名・版・項番とPC-KEIBA取込規則が必要 |
| 実行保留 | bの通貨・基準購入金額・倍率、特殊払戻、訂正/削除/複数版の採用 | 数字であっても円へ変換せず、実HRの決済を解放しない |

公式原本はローカルに提供されていません。外部通信禁止のため公式URLへアクセスしていません。リンクや列名だけを根拠に正式な意味を認定しません。参照資料は [hr_mapping.md](hr_mapping.md) に整理しています。

## 形式検証と分類

空白は、**各フィールドの幅すべてがASCIIスペースU+0020**である場合だけです。a/cは2文字、bは9文字です。空文字、1文字の空白、TAB、改行、全角空白、NULLとは区別し、trim・桁補完・数値化をしません。

| a | b | c | 分類 / 動作 |
|---|---|---|---|
| 2文字空白 | 9文字空白 | 2文字空白 | `unused_blank`。未使用の空白枠として形式正常。金額0や発売なしとは解釈しない |
| 01～28の2桁ASCII数字 | 9桁ASCII数字（全ゼロ以外） | 2桁ASCII数字（00以外） | `ordinary_candidate`。通常形式の候補。正式な的中対象・円・人気順としての意味変換は保留 |
| 00 | 9文字空白 | 2文字空白 | `zero_horse_requires_confirmation`。未使用の空白枠や通常対象に統合しない |
| 00 | 9桁ASCII数字 | 2桁ASCII数字 | `zero_horse_requires_confirmation`。全ゼロも含め、初期値・特払等の意味を決めない |
| 01～28 | 全ゼロのb、またはc=00 | 両方とも指定幅の数字 | `zero_value_requires_confirmation`。0円・順位0として採用しない |
| 上記以外の空白/数字の混合 | — | — | 固定理由 `inconsistent_slot_fields` で要確認停止 |
| a=29～99（00を除く範囲外） | 数字 | 数字 | `candidate_horse_out_of_range` で要確認停止。特殊コードへも自動変換しない |
| NULL、型違い、幅違い、数字と空白の混在、不正文字 | — | — | 固定理由で要確認停止。元の値を例外に含めない |

cの01～99は数字形式としてだけ受け入れます。正式な人気順の許容範囲・特殊コードを確定したものではありません。bは先頭ゼロを保ったstrのまま保持し、`int`・通貨・倍率への変換は実装しません。

異なるa候補は元の順序と枠番号を保ち、複勝を3対象に固定しません。複数対象だけから同着を推定しません。同一の通常候補aは、b/cが同じなら `duplicate_raw_slot`、異なれば `duplicate_a_requires_confirmation` で停止します。空白の未使用枠や、要確認の00枠の反復を通常的中馬の二重登録とは扱いません。00は反復しても未解決のままです。

## 00の確認ゲート

`ClassifiedPayoutSlot.required_context` は、券種別の返還・特払・不成立・発売状態、レース状態、訂正削除の採用という**必要な意味情報**を示します。実列名の確定やフラグ値の自動採用ではありません。

Phase 1.2の検証済み返還対象や返還なしフラグだけでは00を確定できません。不足する券種別フラグの正式定義・優先関係・他の条件が未確認なので、この版はフラグを渡して00を解決する引数を提供しません。00を返還・特払・不成立・発売なし・未使用と決める処理は、資料照合後に別途実装します。

## APIの境界

1. `RawPayoutSlot(a,b,c)`：外部入出力のない入力コンテナ。実レコードの取得機能ではない。
2. `parse_payout_layout(wager_type=..., slots=tuple)`：型・2/9/2幅・数字/全空白・最大3/5枠を検証し、原文字列を保持する。フィールド間の意味や対象集合の完全性は認定しない。
3. `classify_payout_layout(parsed)`：枠の組合せ・範囲・重複を検証して分類する。`slots`は全入力枠を保持し、`ordinary_candidates`は通常形式の候補だけを保持する。
4. `decode_hr_payouts(parsed)`：分類の検証後も**必ず停止**する。実HRの金額や `NormalizedPayout` は生成しない。

形式だけの `format_status` は正常なら `valid`、00/ゼロ値があれば `needs_confirmation` です。不正入力は `PayoutDecodingError` で停止します。全体の `status` は常に `needs_confirmation`、`ready_for_settlement` は常にFalseです。正常形式と決済可能を混同しません。

後方互換性のため1枠から最大枠数までの部分入力も受け付けます。全枠を受け取ったかは `has_all_slots` で区別します。Trueでも、通常対象集合の完全性・レコードの正当性・券種の成立を証明しません。候補だけを取り出して不的中ラベルや決済入力に変換するadapterはありません。

型は不変、reprは詳細非表示、例外診断は固定コードのみです。属性/asdict/tracebackのlocalsによる実データ出力は認めません。分類結果を既存監査レポートへ自動追加しません。実データ取得・DB接続・SQL・CLIは変更しません。

### 架空例

```python
from jra_racing_ai.payout_decoder import (
    RawPayoutSlot, SlotKind, parse_payout_layout, classify_payout_layout,
    PayoutDecodingError, decode_hr_payouts,
)

# 全文字列が架空。bを円や購入100円あたりの金額と認定する例ではない。
unused = RawPayoutSlot("  ", " " * 9, "  ")
parsed = parse_payout_layout(wager_type="tansho", slots=(
    RawPayoutSlot("04", "000000730", "02"), unused, unused,
))
classified = classify_payout_layout(parsed)
assert classified.has_all_slots
assert classified.format_status == "valid"
assert classified.slots[1].kind is SlotKind.UNUSED_BLANK
assert classified.ordinary_candidates[0].raw.b == "000000730"
assert not classified.ready_for_settlement
try:
    decode_hr_payouts(parsed)
except PayoutDecodingError as error:
    assert error.reason == "payout_semantics_unconfirmed"
else:
    raise AssertionError("monetary_gate_must_remain_closed")
```

将来の他券種には、組番の幅・組合せ・繰返し数・特殊値を確認した専用定義が必要です。この版の形式定義は単勝・複勝だけを登録し、他券種は拒否します。最大枠数だけを変更して他券種に流用しません。

## 次のローカル照合

1. 利用者が公式原本とPC-KEIBA資料でa/b/cの正式意味、各幅、未使用枠の初期値、空白の長さ/取込時のtrim有無、数字のゼロ埋めを確認する。
2. a=00かつb/c空白、全空白、全ゼロ、通常対象、同着の各条件を、券種別フラグ・頭数・発売/成立/レース状態と照合する。00の意味は条件ごとに根拠を分ける。
3. bの通貨・基準購入額・倍率、特払等の特殊決済、訂正/削除/履歴の採用を確認する。HR区分2や数字形式だけで最終版・金額を確定しない。
4. 資料名・版・ページ/項番・確認日・非機密の変換規則だけを本対応表へ追記する。実レコードや実集計はOneDrive外の採用済み保存先だけで扱い、チャットへ送らない。
5. Phase 1.5の専用手順で、本人が少数ケースの仮説照合を行える。形式分類API自体はSQL・レコード取得を持たない。実HR adapter設計・金額変換ゲートの開放は別段階で承認する。
