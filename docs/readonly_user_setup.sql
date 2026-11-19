-- 管理者が内容を確認して後日実行する専用読取ロールの手順用SQL。
-- 監査プログラムはこのファイルを読み込まず、実行しません。
-- 今回は未実行です。pckeibaへ管理者として接続した状態を想定します。
-- 同名ロールが既に存在する場合は実行を止め、権限・所有権を先に調査します。
-- パスワードはこのファイルに書きません。psqlの \password を別途使います。

CREATE ROLE jra_audit_reader LOGIN
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 1;

GRANT CONNECT ON DATABASE pckeiba TO jra_audit_reader;
GRANT USAGE ON SCHEMA public TO jra_audit_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO jra_audit_reader;

-- 新しいテーブルは後日の管理者確認後にSELECT権限を追加してください。
-- DEFAULT PRIVILEGES、PUBLIC権限、既存ロール、所有者は変更しません。
