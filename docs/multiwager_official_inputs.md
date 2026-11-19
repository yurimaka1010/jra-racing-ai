# 複数券種研究：正規化済み公式情報のローカル入力

2026-10-24追補：`multiwager/lab.py` と `preparation.py` に、通常HRの完全な的中対象だけを接続する経路を追加しました。各券種の金額9文字は元のまま保持し、金額単位・ROIは未確認のままです。下記の金額付き `OfficialOutcome` 受入口を緩めたり、未確認単位へverifiedを付けたりしていません。詳細は [multiwager_lab.md](multiwager_lab.md)。

`multiwager/official_io.py` は、別途ローカルで照合済みの公式情報を受け取る小さなJSON入力口です。
HR生レコードのデコーダー、物理列の自動採用、金額倍率の推測、訂正・削除の自動選択は実装していません。
公式ラベル・公式払戻の入力準備用であり、実データのROIや実馬券購入を有効化しません。

実ファイルはOneDrive外の `%LOCALAPPDATA%\JraRacingAI` 配下に置きます。
例：`multiwager-research\inputs\official_normalized.local.json`。
実レコード・生オッズ・馬名・認証情報をチャットや同期フォルダーへ保存しません。
この文書のレースID・馬番・金額・ハッシュはすべて架空です。

## JSON形式

```json
{
  "schema_version": "jra.multiwager.official.normalized.v1",
  "records": [
    {
      "race_id": "2025030105010101",
      "date": "2025-03-01",
      "wager": "place",
      "state": "normal",
      "payouts": {"01": 120, "02": 150},
      "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "complete_winning_set": true,
      "flags_verified": true,
      "units": "JPY_PER_100",
      "revision_policy_verified": true
    },
    {
      "race_id": "2025030105010101",
      "date": "2025-03-01",
      "wager": "wide",
      "state": "unconfirmed",
      "payouts": null,
      "source_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "complete_winning_set": null,
      "flags_verified": null,
      "units": null,
      "revision_policy_verified": null
    }
  ]
}
```

`wager` は `place`（複勝）・`wide`（ワイド）・`umaren`（馬連）です。
複勝の対象は2桁の馬番、ワイド・馬連は2桁馬番を昇順で結合した `01-02` 形式です。
`02-01`・同一馬2頭・存在しない馬番を補正しません。
`payouts` は通常の的中対象全体と、100円馬券あたりの確認済み円額です。0・負値・小数・数字文字列を受け付けません。
額はJavaScriptで整数を正確に表現できる `2^53−1` 以下に制限します。これは保存形式上の上限であり、JRA払戻仕様の断定ではありません。
`source_sha256` は照合対象となった公式出典・ローカル原資料のSHA-256です。ハッシュ文字列だけで出典の真正性や単位を検証したことにはなりません。

必須識別情報は `race_id`・`date`・`wager`・`source_sha256` です。
その他の確認項目が未入力・NULLなら、欠損のまま `official_context_missing` で保留します。
型の違う値を埋めたり変換したりせず、余分なフィールドも拒否します。
フラグはJSONの `true` / `false` / `null` のみで、`"true"` や `1` は拒否します。
特殊払戻・返還・発売なし・不成立・中止・未知状態は、通常不的中や0円へ置き換えません。

## 呼出方法と確認ゲート

```python
from pathlib import Path
from jra_racing_ai.multiwager.official_io import load_normalized_outcomes

# 同じ評価対象・同じ馬番集合を別途用意。元データにない馬は追加しない。
rosters = {"2025030105010101": ["01", "02", "03", "04"]}  # 架空例

ready, summary = load_normalized_outcomes(
    Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\multiwager-research\inputs\official_normalized.local.json"),
    rosters,
    # 既存のローカル照合を完了した出典のみ。JSON内の宣言から自動生成しない。
    locally_verified_source_sha256=(),
)
```

初期値の空集合では、JSONに `flags_verified=true` などと書かれていても通常結果を使用可能と認定しません。
別途ローカルで照合を完了した出典SHA-256を、呼出側が明示した場合だけ受け付けます。
この引数は照合済みであるという呼出側の明示的な確認であり、このモジュール自体による事実の独立検証ではありません。
既存の `validate_official` と `official_labels` により通常状態・単位・フラグ・訂正方針・全的中集合を確認した結果のみ、
`ready[(race_id, wager)]` に `OfficialOutcome` として返します。
保留レコードから公式0/1ラベルを生成しません。

`summary` は件数・固定保留理由・ファイルSHA-256だけです。個別レースID・馬番・払戻額は含みません。
`ready` は個別情報を保持するため、標準出力・ログ・チャットへ表示しません。

制限は16MiB/ファイル、10,000レース、30,000レコード、1レース3〜28頭です。
重複JSONキー、同じレース・券種の複数レコード、完全な馬番集合との不整合を拒否します。
既存の保存先検証で同期先・リパースポイント等を拒否します。
現行研究契約どおり対象日を2023-01-01〜2025-09-30に限定し、ラベル解釈前に検証します。
2026年封印期間・Stage 7の期間は読み込めません。2025年10〜12月も現行契約の対象外です。

## HR等から用意する前に必要な別途確認

- HRの券種別物理列・繰返し枠・対象組合せの桁数と格納形式。
- 単勝で確認した対応を他券種へ無条件に流用せず、各券種の金額単位・100円基準・特殊値を照合。
- 複勝2着払い／3着払いを同時点の確認済み発売情報と対応付ける。
- 不成立・特払・返還・発売なし・中止の各フラグ。返還馬や特殊ケースを通常の不的中にしない。
- 同着時の全的中対象。ワイドは着順上位馬の単純な組合せだけで公式対象を推測しない。
- 同一完全キーの欠損・複数版・訂正・削除の採用方針と出典版。
- 予測時点の出走馬集合との完全対応、対象外データの除外数と理由。
- 原資料のローカルSHA-256、照合完了の根拠と対象範囲。未確認範囲は保留のまま残す。

これらを満たす実データの抽出・正規化をこのモジュールは代行しません。
将来の仮想収益計算にも、判断時点価格・固定した研究票・公式払戻との対応が別途必要です。
