# 市場優位条件・第1段階（追加学習0fit）

研究コードは`market_advantage.py`、追補2は`market_advantage_update.py`に独立配置。
予測器はStage 3のABCD39項目の年次時系列OOFだけを使う。
2023・2024・2025年各1〜9月。前年までの学習・早期終了・較正を確認し、保存確率T=0.75を再較正しない。
現行control39・pedigree45・reference47を学習期間内へ適用した再予測は行わない。

## 事前固定と追補

元の研究ID：`18cfe6c38b9f421f9ec3e33004aedc63`。
5条件は本命一致、本命不一致、EV最大候補の参考人気4〜6位、同候補の過去3走以上、同候補のT30→T10市場支持上昇。
元のEV5%・1点ルールは追補2前に集計済み。その結果を開示し、旧成果物は保持する。

追補2の研究ID：`69f7df87de2d412aa17a2805570833c5`。
元の**レース条件**を変更せず、各条件に該当したレースの全馬から`p × T10単勝オッズ >= 1.0`の馬を各100円購入する。
レースごとの1点制限はない。市場本命は同じ条件の全対象レースで1頭100円、同率最低オッズは馬番昇順。
投資額が違うため、損益だけを同額投資の比較とは解釈しない。

T10は既存の中間O1、分終端が判断時刻以前、鮮度300秒以下、同一馬番集合。
発走時刻変更の当時状態を再現できないレースは除外。PC実取得・当時の予定時刻のas-of性は未検証で、公表時刻による事後分析。
確定オッズは買い目のハッシュ固定後に、既存のデコーダーで区分4/5だけを読む。値動き診断専用であり、購入判断や払戻の代替にしない。
公式払戻は保存済み確認済みHR。未精算を0円と扱わない。

回収率95%区間は既存WeekBootstrap、月曜始まりの117週、5,000回、seed20261019。
週合計の投資・払戻を再標本化し、その総額の比率を取る。週回収率の単純平均ではない。
条件の探索に対する名目区間であり、独立検証・採用認定ではない。

## 実行経路

新しい事前登録済みIDに対してのみ、次の順序で実行する。完了済みIDを再実行して上書きしない。

```powershell
$researchPython = 'C:\Users\acbd5\AppData\Local\JraRacingAI\venv\Scripts\python.exe'
& $researchPython -B '.\scripts\research_market_advantage_update.py' prepare --run-id <新しい事前登録済みID>
# 次の段階だけ、従来のPC内DPAPI資格情報をプロセス環境へ安全に読み込む。
# PGPASSWORDをコードやチャットに記入せず、終了後に必ず消去する。
& $researchPython -B '.\scripts\research_market_advantage_update.py' final-odds --run-id <同じID>
& $researchPython -B '.\scripts\research_market_advantage_update.py' analyze --run-id <同じID>
```

成果物は`C:\Users\acbd5\AppData\Local\JraRacingAI\market-advantage\evaluations\<ID>`。
`plan.json`、買い目のコミット、匿名の研究レポート・CSV、個別非公開台帳を分ける。
当該研究にタスク登録や本番接続はない。H・Stage 7・21週の途中成績を開かない。

架空回帰テストは`tests/run_tests.py`。ネットワークと実DBドライバーは禁止したまま実施。

## 追補1：前向き記録の履歴修正は別研究

修正対象は`pedigree_forward/history.py`の履歴母集団確認。
7＝確定を対象とし、9＝中止、0＝削除を別集計して履歴から外す。
1〜6・未知区分・通常レースの欠落・矛盾する重複キーでは停止を維持する。
20日間の月跨ぎ取得と対象日7日前の切断をDBモックで回帰確認。

新しい前向き研究IDは`0fd856ad15aa428d92fa9a60514f553a`。
研究専用タスク`JraRacingAI_Pedigree3_Research_T10`だけを新IDへ切替。
10/31 07:00 JST、Ready、初回対象開催は10/31、21週終了2027/03/21。
旧研究のモデル・温度・閾値・窓・保存成果物は保持。その他のタスクは変更しない。
10/26時点の10/24・25はRA区分6（72レース）で、追加履歴の実読込確認は正規の区分7更新待ち。
