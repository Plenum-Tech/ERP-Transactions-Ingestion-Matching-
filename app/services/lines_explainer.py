from __future__ import annotations
import pandas as pd
import re
from typing import List, Dict, Any, Tuple

# configurable tolerances
AMOUNT_TOL_ABS = 1.00         # AED
AMOUNT_TOL_PCT = 0.01         # 1%
DATE_TOL_DAYS  = 2            # for timing hints

# simple column detection (case-insensitive)
CAND_COLS = {
    "date": ["posting date","transaction date","date"],
    "amount": ["amount","statement amount","txn amount"],
    "currency": ["currency","curr","ccy"],
    "desc": ["description","narration","payee","beneficiary","details","memo"],
    "diff": ["difference","residual","variance","unapplied amount"]
}

def _find_col(df: pd.DataFrame, keys: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for pattern in keys:
        for name_lc, orig in cols.items():
            if pattern == name_lc or name_lc.startswith(pattern) or pattern in name_lc:
                return orig
    return None

def _pct(x: float, base: float) -> float:
    return 0.0 if base == 0 else abs(x) / abs(base)

def _is_fee(desc: str) -> bool:
    return bool(re.search(r"\b(fee|charges?|commission|bank\s?fee|service\s?charge|maintenance)\b", desc, re.I))

def _is_fx(desc: str) -> bool:
    return bool(re.search(r"\b(fx|forex|exchange|conversion|rate|intl|international)\b", desc, re.I))

def _is_atm(desc: str) -> bool:
    return bool(re.search(r"\b(atm|cash\s?withdrawal)\b", desc, re.I))

def _dup_key(date_val, amount_val, desc: str) -> Tuple[str, float, str]:
    # coarsened key for duplicate hints (same amount + similar desc same day)
    desc_norm = re.sub(r"[^a-z0-9 ]","", str(desc).lower())
    desc_norm = re.sub(r"\s+"," ", desc_norm).strip()[:40]
    dstr = ""
    try:
        dstr = pd.to_datetime(date_val).date().isoformat()
    except Exception:
        dstr = str(date_val)
    return (dstr, round(float(amount_val or 0.0), 2), desc_norm)

def explain_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    df = lines_df.copy()

    # locate columns
    col_diff = _find_col(df, CAND_COLS["diff"])
    col_amt  = _find_col(df, CAND_COLS["amount"])
    col_date = _find_col(df, CAND_COLS["date"])
    col_curr = _find_col(df, CAND_COLS["currency"])  # optional, not required for this function
    col_desc = _find_col(df, CAND_COLS["desc"])

    missing = [("Difference", col_diff), ("Amount", col_amt), ("Date", col_date), ("Description", col_desc)]
    missing = [name for name, found in missing if not found]
    if missing:
        raise ValueError(f"Lines.xlsx missing required columns (case-insensitive): {', '.join(missing)}")

    # coerce types on the full sheet
    df[col_diff] = pd.to_numeric(df[col_diff], errors="coerce").fillna(0.0)
    df[col_amt]  = pd.to_numeric(df[col_amt],  errors="coerce")
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_desc] = df[col_desc].astype(str).fillna("")

    # --- NEW: keep ONLY exception rows (Difference != 0)
    mask_ex = df[col_diff].abs() > 1e-9
    df = df.loc[mask_ex].copy()

    # if no exceptions, return an empty frame with the two new columns
    if df.empty:
        out = lines_df.iloc[0:0].copy()
        out["ReasonCodes"] = pd.Series(dtype="string")
        out["Suggestion"]  = pd.Series(dtype="string")
        return out

    # prepare duplicate detection on the filtered (exception) set
    keys = df.apply(lambda r: _dup_key(r[col_date], r[col_amt], r[col_desc]), axis=1)
    dup_counts = keys.value_counts()

    reason_codes_list: List[str] = []
    suggestion_list: List[str] = []

    for idx, row in df.iterrows():
        diff = float(row[col_diff] or 0.0)

        amount = float(row[col_amt] or 0.0)
        desc   = str(row[col_desc] or "")
        rcodes: List[str] = []
        notes:  List[str] = []

        # 1) rounding / small variance?
        tol_abs = AMOUNT_TOL_ABS
        tol_pct = AMOUNT_TOL_PCT
        if abs(diff) <= max(tol_abs, tol_pct * abs(amount)):
            rcodes.append("ROUNDING")
            notes.append(f"Small residual {diff:+.2f} within tolerance (≤ max(AED {tol_abs:.2f}, {tol_pct*100:.0f}% of amount)).")

        # 2) bank fee pattern?
        if _is_fee(desc):
            rcodes.append("POSSIBLE_FEE")
            notes.append("Description indicates bank/service fee. Consider recording separate fee line.")

        # 3) FX variance?
        if _is_fx(desc):
            rcodes.append("FX_RATE_VARIANCE")
            notes.append("Foreign exchange/conversion keywords detected—variance may be due to rate or FX fee.")

        # 4) ATM / cash?
        if _is_atm(desc):
            rcodes.append("CASH_WITHDRAWAL")
            notes.append("ATM/cash withdrawal text found—ensure corresponding cash entry is posted.")

        # 5) possible duplicate?
        if dup_counts.get(_dup_key(row[col_date], row[col_amt], row[col_desc]), 0) > 1:
            rcodes.append("POSSIBLE_DUPLICATE")
            notes.append("Similar amount/description appears multiple times on same date—review for duplicates.")

        # 6) generic amount variance if nothing else flagged
        if not rcodes:
            rcodes.append("AMOUNT_VARIANCE")
            notes.append(f"Residual {diff:+.2f} suggests ledger amount/date/fee mismatch. Check VAT, tips, or split postings.")

        # Build concise human suggestion (top 1–2 notes)
        suggestion = "; ".join(notes[:2])

        reason_codes_list.append(",".join(dict.fromkeys(rcodes)))  # de-dupe, keep order
        suggestion_list.append(suggestion)

    # append columns to the filtered exceptions only
    df["ReasonCodes"] = reason_codes_list
    df["Suggestion"]  = suggestion_list

    return df
