"""Small, fully fictitious aggregates for a local output demonstration."""
from collections import Counter

from .core import RACE_KEY, Scope
from .generate_report import build_report
from .payout_policy import policy_summary


def demonstration_report() -> dict:
    # Deliberately fictitious keys, kept in memory; none are exported as rows.
    races = [("2025", "0101", "01", "01", "01", "01"),
             ("2025", "0101", "01", "01", "01", "02"),
             ("2025", "0102", "02", "01", "01", "01")]
    horse_races = [races[0], races[0], races[1],
                   ("2025", "0103", "03", "01", "01", "01")]
    histogram = Counter(Counter(horse_races).values())
    tables = []
    for name in ("jvd_ra", "jvd_se", "jvd_o1", "jvd_hr"):
        tables.append({"name": name, "kind": "table", "auditable": True,
                       "table_access_method": "heap", "row_security": False, "force_row_security": False,
                       "estimated_rows": None, "estimated_rows_source": "架空構造・統計なし",
                       "columns": [{"name": col, "data_type": "text", "type_is_builtin": True,
                                    "nullable": False} for col in (*RACE_KEY, "data_kubun")],
                       "primary_key": [], "foreign_keys": [], "indexes": []})
    schema = {"status": "synthetic", "tables": tables,
              "warnings": ["実DBから取得した構造ではありません。区分列も架空設定です。"]}
    sections = {
        "races": {"status": "synthetic", "record_count": len(races),
                  "duplicate_race_groups": len(races) - len(set(races)),
                  "null_key_records": 0, "invalid_key_records": 0},
        "horses": {"status": "synthetic", "record_count": len(horse_races),
                   "distinct_races": len(set(horse_races)),
                   "only_ra_races": len(set(races) - set(horse_races)),
                   "only_se_races": len(set(horse_races) - set(races)),
                   "horse_records_per_race_distribution": [
                       {"horse_records": count, "races": number} for count, number in sorted(histogram.items())],
                   "horse_identity": "要確認：架空デモでは馬単位キーを設定していません。"},
        "odds": {"status": "needs_confirmation", "time_semantics": "要確認",
                 "prediction_vs_final_odds": "要確認", "raw_odds_exported": False},
    }
    fictitious_hr = [races[0], races[0], races[2]]
    hr_groups = Counter(fictitious_hr)
    sections["payouts"] = {
        "status": "synthetic", "record_count": len(fictitious_hr),
        "distinct_races": len(hr_groups), "duplicate_key_groups": sum(n > 1 for n in hr_groups.values()),
        "races_only_in_ra": len(set(races) - set(fictitious_hr)),
        "races_only_in_hr": len(set(fictitious_hr) - set(races)),
        "payload_decoding": "not_implemented", "profit_calculation": "not_implemented",
        "special_case_policy": policy_summary(),
        "note": "説明用の架空集計。実SQLやJV-Data payloadの展開結果ではありません。",
    }
    return build_report(schema, sections, source="synthetic", scope=Scope(2025))
