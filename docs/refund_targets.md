# Phase 1.2 単勝・複勝の返還対象馬番判定

文書日：2026-10-15（Asia/Tokyo）。金額・的中・通常払戻の決済処理は対象外です。

## 確認根拠と範囲

利用者が確認済みと提示した次の対応を `confirmed` として実装しました。Codex自身はDBや実レコードを読み取っていません。

| 対応 | 状態 | 根拠・実装範囲 |
|---|---|---|
| jvd_hr.henkan_umaban_joho | confirmed | 28文字。左端が馬番01、右端が28 |
| 位置ごとの値 | confirmed | ASCIIの0=返還なし、1=返還あり |
| henkan_flag_tansho / henkan_flag_fukusho | confirmed | 単勝/複勝ごとの返還有無を示す0/1 |
| フラグが異なる組合せ | needs_confirmation | 利用者回答により未確認。共通bitmapの券種別適用を推測せず停止 |
| 金額単位・返還金額・返還理由 | needs_confirmation | 今回は解釈・計算しない |
| 通常払戻・特払・不成立・発売なし | needs_confirmation | この返還対象判定からは分類・決済しない |

参照：[利用者提示のJV-Data4901.pdf](https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf)。外部通信禁止のため今回未アクセスです。提示された確認内容を根拠とし、独立した仕様原本の照合とは区別します。実DB由来の件数・レコードはこの文書へ保存しません。

## 公開API

`src/jra_racing_ai/refunds.py` の `determine_refund_targets()` に3項目を明示します。正常時は不変の `RefundTargets` を返します。

```python
from jra_racing_ai.refunds import determine_refund_targets

# 架空の入力。実データや馬IDを例示しているものではありません。
fake_bitmap = "".join("1" if n in {1, 12, 28} else "0" for n in range(1, 29))
targets = determine_refund_targets(
    henkan_umaban_joho=fake_bitmap,
    henkan_flag_tansho="1",
    henkan_flag_fukusho="1",
)
assert targets.tansho_umaban == ("01", "12", "28")
assert targets.fukusho_umaban == ("01", "12", "28")
assert targets.is_refunded(wager_type="tansho", horse_number="12") is True
assert targets.is_refunded(wager_type="fukusho", horse_number="02") is False
```

馬番集合は昇順のtupleです。単勝と複勝の両フラグが1の確認済み経路では、共通の28位置の対象を両方へ対応させます。結果に馬ID・レースキー・金額・的中ラベルは含めません。照会のFalseは当該券種の返還対象ではないという意味だけで、非的中・払戻なし・0円を意味しません。

判定関数はDB、ファイル、ロガー、ネットワークを使用しません。既存の `payouts` CLIは引き続き匿名の集計だけを行い、この関数のために実レコードを取得する機能は追加しません。

## 入力の厳密な扱い

- bitmapはPythonの通常のstrだけを受け付け、長さが厳密に28、すべての文字がASCIIの0/1であることを要求します。
- NULL、空文字、空白、制御文字、全角数字、長さ不一致、bytes・数値・配列は拒否します。trim・桁補完・切り詰め・型変換はしません。
- フラグのAPI表現は1文字strの"0"/"1"、または厳密なPython intの0/1だけです。これは明示したコード表現の許可です。bool、float、Decimal、bytes、空白付き文字、"01"などを暗黙変換しません。実列から別の型が返る場合は別途対応確認が必要です。
- 照会馬番は厳密に2文字のASCII str "01"～"28"。整数や"1"、"001"、"00"、"29"などを補正しません。
- 照会券種は `tansho` / `fukusho` だけ。別表記や他券種を自動対応しません。

## フラグとbitmapの整合性

| 単勝/複勝フラグ | bitmapの対象 | 結果 |
|---|---|---|
| 0 / 0 | 対象なし、28文字すべて0 | 正常。両券種の対象集合が空 |
| 1 / 1 | 1頭以上 | 正常。複数頭・28頭すべての対象を保持 |
| 0 / 0 | 1頭以上 | 要確認で停止：targets_without_flag |
| いずれかが1 | 対象なし | 要確認で停止：flag_without_targets |
| 0 / 1 または1 / 0 | 1頭以上 | 要確認で停止：mixed_wager_flags_unconfirmed |
| NULL・不正なフラグ/bitmap | 任意 | 要確認で停止。返還なしへ置換しない |

フラグが異なるケースを破損と断定していません。共通情報のどの馬をどちらの券種に適用するか、追加確認が必要なため処理を止めます。通常払戻・特払等の状態から返還対象を推測しません。

## エラーと情報保護

不正・矛盾・未確認時には `RefundValidationError` を送出します。既存の `SafeAuditError` のサブクラスで、固定コード `refund_needs_confirmation`、状態 `needs_confirmation` を持ちます。例外が発生した入力の処理を中止し、対象の空集合・False・非的中・損失に置換しないでください。

```python
from jra_racing_ai.refunds import RefundValidationError, determine_refund_targets

try:
    determine_refund_targets(
        henkan_umaban_joho="0" * 28,  # 架空の不整合ケース
        henkan_flag_tansho="1",
        henkan_flag_fukusho="1",
    )
except RefundValidationError as error:
    safe_diagnostic = error.diagnostic()
    assert safe_diagnostic["status"] == "needs_confirmation"
    assert safe_diagnostic["reason"] == "flag_without_targets"
    # 入力を再解釈・補正せず、この入力の処理を終了する。
```

diagnosticには固定のstatus/code/reason/fieldだけを含め、入力bitmap・馬番・フラグ値・レース情報は含めません。結果のreprも対象馬番を表示しません。実結果はローカルのメモリ内で扱い、対象tupleやasdict結果、元の入力、tracebackのlocalsをログ・レポート・OneDrive・外部へ出さないでください。既存レポート生成器はこの結果オブジェクトを自動展開しません。

## 今後の確認

1. フラグ混在時の共通bitmapの適用規則。確認までは停止を維持する。
2. 特払・不成立・発売なし・中止との併存/優先関係、返還理由と投票対象の有効性。
3. 返還金額・金額単位・購入単位、通常払戻との決済手順。
4. 実データの列型、版/訂正の選択、開催頭数と馬番の関係。28位置の仕様だけから開催頭数を推定しない。

上記の追加確認と別途承認後に、実データの読み取り・決済・損益処理との統合を検討します。現在の判定結果を予測特徴量に使いません。
