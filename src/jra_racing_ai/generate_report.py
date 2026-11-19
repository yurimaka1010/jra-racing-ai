"""Local aggregate reports with an explicit separation of synthetic outputs."""
from __future__ import annotations

import ctypes
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .core import SafeAuditError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_REPORT_DIR = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\audit-reports")


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _has_link(path: Path) -> bool:
    # Inspect ancestors first, before a leaf probe could traverse a remote link.
    for part in reversed((path, *path.parents)):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            return True
    return False


def _local_disk(path: Path) -> bool:
    if str(path).startswith(("\\\\", "//")):
        return False
    if os.name == "nt":
        # DRIVE_FIXED=3. Exclude SMB-mapped drives and removable/network storage.
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor))
        return drive_type == 3
    return True  # Used by synthetic unit tests on non-Windows hosts.


def validate_output_dir(path: Path, *, real: bool,
                        approved_root: Path = REAL_REPORT_DIR,
                        project_root: Path = PROJECT_ROOT,
                        env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    original = path.expanduser()
    # Reject remote paths before stat/resolve can touch a network filesystem.
    if not original.is_absolute() or not _local_disk(original):
        raise SafeAuditError("unsafe_output")
    if _has_link(original):
        raise SafeAuditError("unsafe_output")
    resolved = original.resolve()
    if real:
        if (_has_link(approved_root) or resolved != approved_root.resolve()
                or not _local_disk(resolved) or _within(resolved, project_root.resolve())):
            raise SafeAuditError("unsafe_output")
        sync_roots = []
        for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            if value := env.get(key):
                sync_path = Path(value)
                if not sync_path.is_absolute() or not _local_disk(sync_path):
                    raise SafeAuditError("unsafe_output")
                if _has_link(sync_path):
                    raise SafeAuditError("unsafe_output")
                sync_roots.append(sync_path.resolve())
        if (any(_within(resolved, root) for root in sync_roots)
                or any("onedrive" in part.lower() for part in resolved.parts)):
            raise SafeAuditError("unsafe_output")
    else:
        if not _within(resolved, (project_root / "reports").resolve()):
            raise SafeAuditError("unsafe_output")
    return resolved


def _json_default(value):
    # Decimal/date values only come from approved aggregates or catalog timestamps.
    from decimal import Decimal
    from datetime import date
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise SafeAuditError("internal_error")


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=_json_default)
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def _render_value(value: Any, depth: int = 0) -> list[str]:
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.extend(["", f"{'#' * min(6, depth + 3)} {_cell(key)}", ""])
                lines.extend(_render_value(item, depth + 1))
            else:
                lines.append(f"- **{_cell(key)}**: {_cell(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return ["該当なし／未実行。"]
        if all(isinstance(item, dict) for item in value):
            columns = list(dict.fromkeys(key for item in value for key in item))
            return (["| " + " | ".join(map(_cell, columns)) + " |",
                     "| " + " | ".join("---" for _ in columns) + " |"]
                    + ["| " + " | ".join(_cell(item.get(key, "")) for key in columns) + " |" for item in value])
        return [f"- {_cell(item)}" for item in value]
    return [_cell(value)]


def ml_readiness(schema: dict) -> dict:
    items = []
    for table in schema.get("tables", []):
        name = table["name"]
        specific = {
            "jvd_ra": ("レース条件の候補。条件変更の反映時点は要確認。", "確定成績・中止区分の候補。", "実施判定とラベル対象の絞り込み候補。"),
            "jvd_se": ("出走馬情報の候補。登録・取消・変更の時点は要確認。", "着順等の結果情報の候補。", "単勝・複勝ラベル候補。結果列と特殊ケースの仕様は要確認。"),
            "jvd_o1": ("予測時点の保存履歴は要確認。区分5は依頼文では確定（月曜）オッズで、レース前特徴量には使用禁止。", "最終・確定オッズ。事前の購入判断には利用しない。", "ラベルや払戻の代替にしない。予測時点オッズと評価を分ける。"),
            "jvd_hr": ("prohibited_feature：今回レースの払戻情報を予測特徴量に使用しない。", "決済・的中対象の公式照合用。列・単位・特殊値の対応は要確認。", "通常の単勝・複勝的中対象集合の照合候補。同着・返還・特払等を分離する。"),
            "jvd_ks": ("騎手マスタ候補。過去時点の履歴と更新方式は要確認。", "後日の更新値・累積値の遡及利用に注意。", "通常は直接の結果ラベルにしない候補。"),
        }.get(name, ("仕様・履歴・更新方式の確認が必要。", "要確認。", "要確認。"))
        items.append({"table": name, "classification": "要確認",
                      "feature_status": "prohibited_feature" if name == "jvd_hr" else "needs_confirmation",
                      "pre_race_information": specific[0], "post_race_information": specific[1],
                      "label_candidates": specific[2],
                      "feature_candidates": ("使用禁止。決済・ラベル照合専用。" if name == "jvd_hr" else
                                              "列の意味と予測時点の入手可能性を仕様と照合して選定。"),
                      "leakage_risks": "結果・最終オッズ・事後更新マスタ・未来を含む累積値。作成日だけでは時点を判定しない。",
                      "additional_investigation": "JRA-VAN仕様とPC-KEIBA列定義、更新履歴、利用可能時点。"})
    return {"status": "要確認", "tables": items,
            "next_phase_tasks": [
                "6列レースキー・馬単位キー・区分列の型と意味を仕様で確定する。",
                "未対応レース、重複、欠損、中止・取消・除外・同着の扱いを決める。",
                "docs/hr_mapping.mdでHRの区分・金額単位・繰返し・特殊ケースの実列対応を確定する。",
                "docs/phase2_data_contract.md・phase2_feature_registry.md・backtest_policy.mdの確認ゲートを満たす。",
                "予測時刻・オッズ取得時刻・払戻時刻を区別し、時点履歴の保有状況を確認する。",
                "最終オッズしかない場合に、予測時点の期待値検証が可能か見直す。",
                "過去時点の特徴量だけで時系列分割し、単勝・複勝ラベルの定義を確認する。",
                "確率校正、控除・購入制約・投資上限を含む月間純利益評価の方法を決める。",
                "実監査結果の利用者確認後、Phase 2のLightGBM基本モデル計画を作成する。",
            ]}


def build_report(schema: dict, sections: dict, *, source: str, scope=None) -> dict:
    if source not in ("synthetic", "local_database"):
        raise SafeAuditError("invalid_configuration")
    return {"format_version": 1, "source": source,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": None if scope is None else {"year": scope.year, "venues": list(scope.venues),
                                                "global_summary": scope.global_summary},
            "snapshot": "架空データ" if source == "synthetic" else "REPEATABLE READ / READ ONLY（同一接続）",
            "disclosure": "ローカル集計。公開・外部共有の判断は利用者が行う。個別レコードは含めない。",
            "schema": schema, "audits": sections, "ml_readiness": ml_readiness(schema)}


def schema_markdown(schema: dict) -> str:
    lines = ["# テーブル構造一覧", ""]
    lines.extend(_render_value({"status": schema.get("status"),
                                "catalog_counts": schema.get("catalog_counts", {}),
                                "warnings": schema.get("warnings", [])}))
    overview = [{key: table.get(key) for key in (
        "name", "kind", "auditable", "estimated_rows", "total_size_bytes")}
        for table in schema.get("tables", [])]
    lines.extend(["", "## 一覧", "", *_render_value(overview)])
    for table in schema.get("tables", []):
        lines.extend(["", f"## public.{_cell(table['name'])}", ""])
        details = {key: value for key, value in table.items() if key not in ("name", "columns", "indexes", "foreign_keys")}
        lines.extend(_render_value(details))
        for label, key in (("カラム", "columns"), ("外部キー", "foreign_keys"), ("インデックス", "indexes")):
            lines.extend(["", f"### {label}", "", *_render_value(table.get(key, []))])
    return "\n".join(lines) + "\n"


def report_markdown(report: dict) -> str:
    summary = {key: value for key, value in report.items() if key not in ("schema", "ml_readiness")}
    schema = report["schema"]
    summary["schema_summary"] = {"listed_relations": len(schema.get("tables", [])),
                                 "status": schema.get("status"), "warnings": schema.get("warnings", [])}
    return ("# JRA Phase 1 監査レポート\n\n"
            "構造詳細は [schema.md](schema.md)、次Phaseの課題は [lightgbm_tasks.md](lightgbm_tasks.md)。\n\n"
            + "\n".join(_render_value(summary)) + "\n")


def write_report(report: dict, output_dir: Path) -> dict[str, Path]:
    real = report.get("source") == "local_database"
    if report.get("source") not in ("synthetic", "local_database"):
        raise SafeAuditError("invalid_configuration")
    root = validate_output_dir(output_dir, real=real)
    # Serialize first: no partial files if a payload is unsupported.
    payloads = {
        "audit.json": json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        "schema.json": json.dumps(report["schema"], ensure_ascii=False, indent=2, default=_json_default) + "\n",
        "anomalies.json": json.dumps(report["audits"], ensure_ascii=False, indent=2, default=_json_default) + "\n",
        "audit.md": report_markdown(report),
        "schema.md": schema_markdown(report["schema"]),
        "lightgbm_tasks.md": "# 次Phaseに向けた確認事項\n\n" + "\n".join(_render_value(report["ml_readiness"])) + "\n",
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        # Recheck after mkdir; refuse symlink/reparse paths and overwrite.
        validate_output_dir(root, real=real)
        run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ("_synthetic" if not real else "_local"))
        run.mkdir(exist_ok=False)
        if _has_link(run):
            raise SafeAuditError("unsafe_output")
        for name, content in payloads.items():
            validate_output_dir(root, real=real)
            target = run / name
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        return {name: run / name for name in payloads}
    except SafeAuditError:
        raise
    except Exception:
        raise SafeAuditError("output_failed") from None
