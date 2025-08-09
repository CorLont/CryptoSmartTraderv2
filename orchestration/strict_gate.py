# orchestration/strict_gate.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass
class GateResult:
    passed: pd.DataFrame
    total_candidates: int
    passed_count: int
    status: str  # "OK" | "EMPTY" | "INVALID"

def strict_toplist(pred_df: pd.DataFrame, pred_col="pred_720h", conf_col="conf_720h", thr=0.80) -> GateResult:
    if pred_df is None or pred_df.empty or pred_col not in pred_df or conf_col not in pred_df:
        return GateResult(pd.DataFrame(), 0, 0, "INVALID")
    df = pred_df.dropna(subset=[pred_col, conf_col]).copy()
    total = len(df)
    df = df[df[conf_col] >= thr].sort_values(pred_col, ascending=False)
    status = "EMPTY" if df.empty else "OK"
    return GateResult(df.reset_index(drop=True), total, len(df), status)

def apply_strict_gate_orchestration(all_preds: dict[str, pd.DataFrame], thr=0.80) -> dict:
    results = {h: strict_toplist(df, f"pred_{h}", f"conf_{h}", thr) for h, df in all_preds.items()}
    total = sum(r.total_candidates for r in results.values())
    passed = sum(r.passed_count for r in results.values())
    any_ok = any(r.status == "OK" for r in results.values())
    return {
        "gate_status": "OK" if any_ok else "EMPTY",
        "total_candidates": total,
        "total_passed": passed,
        "per_horizon": {h: r.passed for h, r in results.items()},
    }