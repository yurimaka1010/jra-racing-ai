# KEIBA LABの起動と復旧

閲覧URLは `http://127.0.0.1:8767/` です。新しい研究や予測記録を開始せず、既存の表示サーバーだけを起動します。

## 再起動後

`KEIBA_LAB_Local_Viewer` が本人ユーザーのWindowsログインから約5秒後に起動します。画面ロックは可能ですが、ログオフ中やPC停止中は利用できません。ブラウザは自動では開きません。

異常終了時は1分間隔で最大3回再起動します。時間制限とバッテリー移行による自動終了は設定しません。既存のPostgreSQL起動タスク・研究記録器・朝7時タスクは変更しません。

## ワンクリックで開く・復旧する

デスクトップの `KEIBA LAB` ショートカットを開きます。表示サーバーが正常ならそのままブラウザを開き、停止していれば既存の起動処理で復旧してから開きます。既存ショートカットと名前が衝突した場合は `KEIBA LAB Local` を使用します。

ショートカットを使わない場合は次を実行します。

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai\scripts\keiba_lab_live.ps1'
```

## 設定の再登録

```powershell
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -NoProfile -NonInteractive -ExecutionPolicy Bypass -File 'C:\Users\acbd5\OneDrive\デスクトップ\Codex 練習用\jra-racing-ai\scripts\register_keiba_lab_viewer.ps1'
```

`-CheckOnly` を付ければ、依存ファイルと表示サーバーの応答だけを確認し、タスクやファイルを変更しません。

## 確認とログ

- `http://127.0.0.1:8767/api/health` は表示サーバーの生存確認です。DB接続の成功まで保証するものではありません。
- 起動ログは `C:\Users\acbd5\AppData\Local\JraRacingAI\keiba-lab-live\service-*.log` です。
- 登録記録は同フォルダーの `startup-registrations` 配下に新しいIDで保存します。
- 最新の出馬表・結果を取得するにはPostgreSQLとPC-KEIBAの通常更新が必要です。DBの停止と表示サーバーの停止は別々に確認します。
- 資格情報は従来の本人ユーザー用DPAPIファイルを読み、パスワードをタスク引数・コードへ保存しません。
- 公開範囲はループバック `127.0.0.1:8767` のみです。
