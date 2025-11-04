# app/services/bank_reco_evaluator.py
from __future__ import annotations
import os, json, unicodedata, re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from openai import OpenAI

# -------------------- Config --------------------
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AUTO_THRESHOLD     = float(os.getenv("AUTO_THRESHOLD", "0.90"))
REVIEW_THRESHOLD   = float(os.getenv("REVIEW_THRESHOLD", "0.75"))
DATE_WINDOW_DAYS   = int(os.getenv("DATE_WINDOW_DAYS", "5"))
AMOUNT_WINDOW_ABS  = float(os.getenv("AMOUNT_WINDOW_ABS", "5"))
AMOUNT_WINDOW_PCT  = float(os.getenv("AMOUNT_WINDOW_PCT", "0.05"))
CANDIDATE_LIMIT    = int(os.getenv("CANDIDATE_LIMIT", "8"))
MAX_ROWS           = int(os.getenv("LLM_EVAL_MAX_ROWS", "200"))
RECO_K             = int(os.getenv("RECO_K", "3"))
DEFAULT_CCY        = os.getenv("DEFAULT_CCY", "AED")

client = OpenAI()

# -------------------- Canonicalization & helpers --------------------
def _canon(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper()
    s = re.sub(r"\b(LLC|LTD|CO|INC|PVT|PTE|LLP|EST|COMPANY|TRD|TRADING)\b", "", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _days(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(abs((a.normalize() - b.normalize()).days))

def _amount_band(amount: float) -> float:
    return max(AMOUNT_WINDOW_ABS, AMOUNT_WINDOW_PCT * abs(amount))

def _merchant_sim(a: str, b: str) -> float:
    a = a or ""; b = b or ""
    return max(
        fuzz.token_set_ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b),
    ) / 100.0

def _canon_ccy(x: Any, default: str = DEFAULT_CCY) -> str:
    s = ("" if x is None else str(x)).strip().upper()
    return s or default

# -------------------- Reasons & humanization --------------------
_REASON_MAP = {
    "auto_match": "AUTO_MATCH",
    "low_confidence": "LOW_CONFIDENCE",
    "multiple_possible_matches": "MULTIPLE_POSSIBLE_MATCHES",
    "amount_delta_exceeds_auto_threshold": "AMOUNT_DELTA_EXCEEDS_GUARD",
    "date_delta_days_exceeds_auto_threshold": "DATE_DELTA_EXCEEDS_GUARD",
    "amount_delta_high": "AMOUNT_DELTA_EXCEEDS_GUARD",
    "date_delta_days_high": "DATE_DELTA_EXCEEDS_GUARD",
    "missing_ledger": "MISSING_LEDGER",
    "missing_candidate": "MISSING_CANDIDATE",
}

_ALLOWED_REASON_CODES = list({v for v in _REASON_MAP.values()} | {
    "AUTO_MATCH", "LOW_CONFIDENCE", "MULTIPLE_POSSIBLE_MATCHES",
    "AMOUNT_DELTA_EXCEEDS_GUARD", "DATE_DELTA_EXCEEDS_GUARD",
    "MISSING_LEDGER", "MISSING_CANDIDATE",
})

def _norm_reasons(reasons: List[str]) -> List[str]:
    out: List[str] = []
    for r in reasons or []:
        key = r.strip().lower()
        out.append(_REASON_MAP.get(key, r.upper()))
    seen = set(); uniq = []
    for r in out:
        if r not in seen:
            seen.add(r); uniq.append(r)
    return [r for r in uniq if r in _ALLOWED_REASON_CODES] or ["LOW_CONFIDENCE"]

def _fmt_ccy(amount: float, ccy: str = DEFAULT_CCY) -> str:
    return f"{ccy} {amount:,.2f}"

def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")

def _humanize_reasons(
    decision: str,
    reason_codes: List[str],
    tx_amount: float,
    selected: Optional[Dict[str, Any]],
    amt_guard_abs: float,
    date_guard_days: int,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    amt_delta = selected.get("amount_delta") if selected else None
    date_delta = selected.get("date_delta_days") if selected else None
    merch_sim = selected.get("merchant_sim") if selected else None

    for code in reason_codes or []:
        if code == "AUTO_MATCH":
            if selected is not None:
                msg = (
                    f"Matched: amount difference {_fmt_ccy(0 if amt_delta is None else amt_delta)} "
                    f"(allowed up to {_fmt_ccy(max(1.0, 0.01*tx_amount))}), "
                    f"date difference {0 if date_delta is None else date_delta} day(s) "
                    f"(policy ≤ {date_guard_days}), merchant similarity {merch_sim:.2f}."
                )
            else:
                msg = "Amounts and date align within policy, with a clear description match."
        elif code == "LOW_CONFIDENCE":
            msg = "Candidate exists but confidence is low; needs a quick human check."
        elif code == "MULTIPLE_POSSIBLE_MATCHES":
            msg = "More than one plausible ledger entry; please pick the correct one."
        elif code == "AMOUNT_DELTA_EXCEEDS_GUARD":
            msg = (
                "Amount difference exceeds the auto-match tolerance "
                f"(delta {_fmt_ccy(amt_delta or 0)}, policy ≤ {_fmt_ccy(max(1.0, 0.01*tx_amount))})."
            )
        elif code == "DATE_DELTA_EXCEEDS_GUARD":
            msg = (
                "Date difference exceeds the auto-match tolerance "
                f"({_plural(date_delta or 0, 'day')}, policy ≤ {date_guard_days} days)."
            )
        elif code in ("MISSING_LEDGER", "MISSING_CANDIDATE"):
            msg = "No ledger candidate found in the allowed date/amount window."
        else:
            msg = "Needs review due to policy checks."
        msgs.append({"code": code, "message": msg})

    if not msgs:
        msgs.append({"code": "LOW_CONFIDENCE", "message": "Candidate exists but confidence is low; needs review."})
    return msgs

# -------------------- Core shortlist/widen --------------------
def _block_candidates(tx: Dict[str, Any], ledger_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shortlist by currency (if present), date±D, amount±band, then sort by description similarity.
    Expected ledger_df columns: ledger_id, tx_date, total_amount, currency, description_canonical
    """
    amt = float(tx["amount"])
    band = _amount_band(amt)
    post_date = pd.to_datetime(tx["post_date"], errors="coerce")
    dcanon = _canon(tx.get("merchant_desc") or tx.get("merchant_canonical") or tx.get("description") or "")
    cur = _canon_ccy(tx.get("currency", ""))

    lo = post_date - pd.Timedelta(days=DATE_WINDOW_DAYS)
    hi = post_date + pd.Timedelta(days=DATE_WINDOW_DAYS)

    df = ledger_df.copy()
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    df["currency"] = df["currency"].map(_canon_ccy)
    df["description_canonical"] = df["description_canonical"].astype(str).fillna("")

    if cur:
        df = df[df["currency"].eq(cur)]

    df = df[
        df["tx_date"].between(lo, hi) &
        df["total_amount"].between(amt - band, amt + band)
    ].copy()

    if df.empty:
        return df

    df["merchant_sim"] = df["description_canonical"].apply(lambda s: _merchant_sim(dcanon, s))
    df["amount_delta"] = pd.to_numeric((df["total_amount"] - amt).abs(), errors="coerce").fillna(np.inf)
    df["date_delta_days"] = pd.to_numeric(df["tx_date"].apply(lambda d: _days(post_date, d)), errors="coerce").fillna(999999).astype(int)
    return df.sort_values(["merchant_sim"], ascending=False).head(CANDIDATE_LIMIT).reset_index(drop=True)

def _widen_once(tx: Dict[str, Any], ledger_df: pd.DataFrame, date_days: int, extra_pct: float) -> pd.DataFrame:
    amt = float(tx["amount"])
    post_date = pd.to_datetime(tx["post_date"], errors="coerce")
    dcanon = _canon(tx.get("merchant_desc") or tx.get("merchant_canonical") or tx.get("description") or "")
    cur = _canon_ccy(tx.get("currency", ""))

    base = _amount_band(amt)
    band = base + extra_pct * abs(amt)
    lo = post_date - pd.Timedelta(days=date_days)
    hi = post_date + pd.Timedelta(days=date_days)

    df = ledger_df.copy()
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    df["currency"] = df["currency"].map(_canon_ccy)
    df["description_canonical"] = df["description_canonical"].astype(str).fillna("")

    if cur:
        df = df[df["currency"].eq(cur)]
    df = df[
        df["tx_date"].between(lo, hi) &
        df["total_amount"].between(amt - band, amt + band)
    ].copy()
    if df.empty:
        return df

    df["merchant_sim"] = df["description_canonical"].apply(lambda s: _merchant_sim(dcanon, s))
    df["amount_delta"] = pd.to_numeric((df["total_amount"] - amt).abs(), errors="coerce").fillna(np.inf)
    df["date_delta_days"] = pd.to_numeric(df["tx_date"].apply(lambda d: _days(post_date, d)), errors="coerce").fillna(999999).astype(int)
    return df.sort_values(["merchant_sim"], ascending=False).head(CANDIDATE_LIMIT).reset_index(drop=True)

def _recommend_candidates(tx: Dict[str, Any], ledger_df: pd.DataFrame, k: int = RECO_K) -> List[Dict[str, Any]]:
    for dwin in (10, 15, 30):
        for extra in (0.02, 0.05, 0.10):
            df = _widen_once(tx, ledger_df, date_days=dwin, extra_pct=extra)
            if not df.empty:
                out = df.head(k).to_dict(orient="records")
                for r in out:
                    r["why"] = f"Within ±{dwin}d and widened amount band (+{int(extra*100)}%)."
                return out

    df = _widen_once({**tx, "currency": ""}, ledger_df, date_days=30, extra_pct=0.10)
    if not df.empty:
        out = df.head(k).to_dict(orient="records")
        for r in out:
            r["why"] = "Currency ignored; ±30d and widened amount band (+10%)."
        return out

    amt = float(tx["amount"])
    post_date = pd.to_datetime(tx["post_date"], errors="coerce")
    dcanon = _canon(tx.get("merchant_desc") or tx.get("merchant_canonical") or tx.get("description") or "")

    df2 = ledger_df.copy()
    df2["tx_date"] = pd.to_datetime(df2["tx_date"], errors="coerce")
    df2["total_amount"] = pd.to_numeric(df2["total_amount"], errors="coerce")
    df2["description_canonical"] = df2["description_canonical"].astype(str).fillna("")

    df2["amount_delta"] = pd.to_numeric((df2["total_amount"] - amt).abs(), errors="coerce").fillna(np.inf)
    df2["date_delta_days"] = pd.to_numeric(df2["tx_date"].apply(lambda d: _days(post_date, d)), errors="coerce").fillna(999999).astype(float)
    df2["merchant_sim"] = df2["description_canonical"].apply(lambda s: _merchant_sim(dcanon, s))

    denom_amt = max(abs(amt), 1e-6)
    df2["nn_score"] = (
        0.6 * (1.0 - (df2["amount_delta"] / denom_amt).clip(0, 1)) +
        0.25 * (1.0 - (df2["date_delta_days"] / 30.0).clip(0, 1)) +
        0.15 * df2["merchant_sim"]
    )
    out = df2.sort_values("nn_score", ascending=False).head(k).to_dict(orient="records")
    for r in out:
        r["why"] = "Closest by amount/date with description similarity considered."
    return out

# -------------------- LLM prompt & call --------------------
def _build_prompt(tx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    schema = {
        "type": "object",
        "required": ["decision", "reason_codes", "explanation"],
        "properties": {
            "decision": {"enum": ["AUTO", "REVIEW", "NO_MATCH"]},
            "selected_index": {"type": ["integer", "null"]},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"}
        },
        "additionalProperties": False
    }

    payload = {
        "transaction": {
            "tx_id": tx["tx_id"],
            "post_date": str(pd.to_datetime(tx["post_date"]).date()),
            "amount": float(tx["amount"]),
            "currency": tx.get("currency") or "",
            "description_canonical": _canon(tx.get("merchant_desc") or tx.get("merchant_canonical") or tx.get("description") or "")
        },
        "candidates": [
            {
                "index": i,
                "ledger_id": c["ledger_id"],
                "tx_date": str(pd.to_datetime(c["tx_date"]).date()),
                "total_amount": float(c["total_amount"]),
                "description_canonical": c["description_canonical"],
                "merchant_sim": round(float(c["merchant_sim"]), 3),
                "amount_delta": round(float(c["amount_delta"]), 2),
                "date_delta_days": int(c["date_delta_days"])
            }
            for i, c in enumerate(candidates)
        ],
        "rules_summary": {
            "auto_threshold": AUTO_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD,
            "guards": {"amount_delta_max_auto": "max(1.0, 1%)", "date_delta_days_max_auto": 2},
            "policy": "Prefer smallest amount_delta, then date_delta_days, then highest merchant_sim"
        },
        "allowed_reason_codes": _ALLOWED_REASON_CODES,
        "schema": schema,
        "instruction": (
            "Decide AUTO if a clearly best candidate satisfies guards and is above auto_threshold; "
            "REVIEW if ambiguous or between thresholds; NO_MATCH if none qualify. "
            "If decision is AUTO, you MUST include a valid integer selected_index (0-based) pointing to candidates[]. "
            "If candidates has length 1 and decision is AUTO, selected_index MUST be 0. "
            "Choose reason_codes ONLY from allowed_reason_codes. "
            "Respond STRICT JSON matching the schema—no extra keys."
        )
    }
    return json.dumps(payload, ensure_ascii=False)

def _call_llm(prompt_json: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a finance reconciliation judge. Respond ONLY with the requested JSON."},
            {"role": "user", "content": prompt_json}
        ]
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        data = {}

    decision = data.get("decision", "REVIEW")
    selected_index = data.get("selected_index", None)
    reason_codes = data.get("reason_codes") or (["AUTO_MATCH"] if decision == "AUTO" else ["LOW_CONFIDENCE"])
    explanation = data.get("explanation") or ("Auto-selected because only one candidate" if decision == "AUTO" else "")

    return {"decision": decision, "selected_index": selected_index, "reason_codes": reason_codes, "explanation": explanation}

# -------------------- Normalizers & evaluator --------------------
def _normalize_ledger_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ledger candidate dataframe to:
      ledger_id, tx_date, total_amount, currency, description_canonical
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for lc, orig in cols.items():
                if n == lc or lc.startswith(n) or n in lc:
                    return orig
        return None

    cid = pick("ledger_id", "document no.", "document no", "external doc no", "entry no", "id", "no.", "doc no", "document number")
    cdt = pick("tx_date", "posting date", "date", "document date", "entry date", "value date")
    cam = pick("total_amount", "remaining amount", "applied amount", "amount", "statement amount")
    ccy = pick("currency", "currency code", "curr", "ccy")
    dsc = pick("description_canonical", "description", "posting description", "narration", "memo", "details")

    out = df.copy()
    if cid is None:
        out["ledger_id"] = out.reset_index().index.astype(str)
    else:
        out = out.rename(columns={cid: "ledger_id"})
    if cdt and cdt != "tx_date":
        out = out.rename(columns={cdt: "tx_date"})
    if cam and cam != "total_amount":
        out = out.rename(columns={cam: "total_amount"})
    if ccy and ccy != "currency":
        out = out.rename(columns={ccy: "currency"})
    if dsc and dsc != "description_canonical":
        out = out.rename(columns={dsc: "description_canonical"})

    # coerce & clean
    out["tx_date"] = pd.to_datetime(out["tx_date"], errors="coerce")
    out["total_amount"] = pd.to_numeric(out["total_amount"], errors="coerce")
    out["currency"] = out["currency"].map(_canon_ccy) if "currency" in out.columns else DEFAULT_CCY
    out["description_canonical"] = out.get("description_canonical", "").astype(str).fillna("")
    out = out.dropna(subset=["tx_date", "total_amount"]).reset_index(drop=True)
    return out[["ledger_id", "tx_date", "total_amount", "currency", "description_canonical"]]

def evaluate_llm(
    statement_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    *,
    max_rows: int = MAX_ROWS
) -> Dict[str, Any]:
    """
    statement_df columns (flexible): tx_id, post_date, amount, currency, merchant_desc/description
    ledger_df columns (flexible): ledger_id, tx_date, total_amount, currency, description_canonical/description
    """
    # normalize inputs
    st = statement_df.rename(columns=str.lower).copy()
    lg = _normalize_ledger_df(ledger_df)

    # required statement columns
    req_st = {"tx_id", "post_date", "amount", "currency"}
    missing_st = [c for c in req_st if c not in st.columns]
    if missing_st:
        raise ValueError(f"Missing columns in statement: {missing_st}")

    # coerce types
    st["post_date"] = pd.to_datetime(st["post_date"], errors="coerce")
    st["amount"]    = pd.to_numeric(st["amount"], errors="coerce")

    # drop bad rows
    st = st.dropna(subset=["post_date", "amount", "currency"])

    # canonicalize description & currency
    desc_col = "merchant_desc" if "merchant_desc" in st.columns else ("description" if "description" in st.columns else None)
    st["merchant_canonical"] = st[desc_col].fillna("").map(_canon) if desc_col else ""
    st["currency"] = st["currency"].map(_canon_ccy)

    results: List[Dict[str, Any]] = []
    auto_ct = review_ct = nomatch_ct = 0

    for _, tx in st.head(max_rows).iterrows():
        tx_dict = tx.to_dict()

        # shortlist & widen
        cands_df = _block_candidates(tx_dict, lg)
        if cands_df.empty:
            widened = None
            for dwin in (10, 15, 30):
                for extra in (0.02, 0.05, 0.10):
                    widened = _widen_once(tx_dict, lg, date_days=dwin, extra_pct=extra)
                    if not widened.empty:
                        cands_df = widened
                        break
                if not cands_df.empty:
                    break

        cands = cands_df.to_dict(orient="records")

        # LLM judge
        if cands:
            prompt = _build_prompt(tx_dict, cands)
            decision = _call_llm(prompt)
        else:
            decision = {
                "decision": "NO_MATCH",
                "selected_index": None,
                "reason_codes": ["MISSING_LEDGER"],
                "explanation": "No candidates available.",
            }

        # safety nets
        if decision["decision"] == "AUTO" and decision.get("selected_index") is None and len(cands) == 1:
            decision["selected_index"] = 0
            decision["reason_codes"] = decision.get("reason_codes") or ["AUTO_MATCH"]
            decision["explanation"] = decision.get("explanation") or "Single candidate satisfies rules; selecting index 0."

        if decision["decision"] == "AUTO" and (
            decision.get("selected_index") is None or not (0 <= int(decision["selected_index"]) < len(cands))
        ):
            decision["decision"] = "REVIEW"
            decision["reason_codes"] = list(set((decision.get("reason_codes") or []) + ["MULTIPLE_POSSIBLE_MATCHES"]))
            decision["explanation"] = (decision.get("explanation") or "") + " (Invalid or missing selected_index)"

        # post-LLM guard (single candidate)
        if len(cands) == 1:
            c = cands[0]
            amt = float(tx_dict["amount"])
            amt_guard = max(1.0, 0.01 * amt)
            date_guard = 2
            within = (c["amount_delta"] <= amt_guard) and (c["date_delta_days"] <= date_guard)
            if within and decision["decision"] != "AUTO":
                decision["decision"] = "AUTO"
                decision["selected_index"] = 0
                decision["reason_codes"] = ["AUTO_MATCH"]
                decision["explanation"] = "Single candidate within guards; auto-upgraded."
            elif (not within) and decision["decision"] == "AUTO":
                decision["decision"] = "REVIEW"
                rcodes = []
                if c["amount_delta"] > amt_guard: rcodes.append("AMOUNT_DELTA_EXCEEDS_GUARD")
                if c["date_delta_days"] > date_guard: rcodes.append("DATE_DELTA_EXCEEDS_GUARD")
                decision["reason_codes"] = rcodes or ["LOW_CONFIDENCE"]
                decision["selected_index"] = None
                decision["explanation"] = "Auto downgraded; guard check failed."

        # finalize
        selected = None
        if decision["decision"] == "AUTO" and decision.get("selected_index") is not None:
            si = int(decision["selected_index"])
            if 0 <= si < len(cands):
                selected = cands[si]
                auto_ct += 1
            else:
                decision["decision"] = "REVIEW"
                decision["reason_codes"] = list(set((decision.get("reason_codes") or []) + ["MULTIPLE_POSSIBLE_MATCHES"]))
                decision["explanation"] = (decision.get("explanation") or "") + " (selected_index out of range)"
                review_ct += 1
        elif decision["decision"] == "REVIEW":
            review_ct += 1
        else:
            nomatch_ct += 1

        # reasons & proxy
        decision["reason_codes"] = _norm_reasons(decision.get("reason_codes"))

        if selected is None:
            proxy = None
        else:
            amt = float(tx_dict["amount"])
            band = max(5.0, 0.02 * amt)
            denom = max(band, 1e-6)
            amt_comp = max(0.0, 1.0 - selected["amount_delta"] / denom)
            date_comp = max(0.0, 1.0 - float(selected["date_delta_days"]) / 5.0)
            merch_comp = float(selected["merchant_sim"])
            proxy = round(0.55 * amt_comp + 0.25 * date_comp + 0.20 * merch_comp, 3)

        amt_guard = max(1.0, 0.01 * float(tx_dict["amount"]))
        date_guard = 2
        human_reasons = _humanize_reasons(
            decision=decision["decision"],
            reason_codes=decision["reason_codes"],
            tx_amount=float(tx_dict["amount"]),
            selected=selected,
            amt_guard_abs=amt_guard,
            date_guard_days=date_guard,
        )

        # recommendations for NO_MATCH
        recommendations: List[Dict[str, Any]] = []
        if decision["decision"] == "NO_MATCH":
            recs = _recommend_candidates(tx_dict, lg, k=RECO_K)
            for r in recs:
                recommendations.append({
                    "ledger_id": r.get("ledger_id"),
                    "tx_date": (str(pd.to_datetime(r.get("tx_date")).date())
                                if r.get("tx_date") is not None else None),
                    "total_amount": (float(r.get("total_amount"))
                                     if r.get("total_amount") is not None else None),
                    "description_canonical": r.get("description_canonical"),
                    "merchant_sim": round(float(r.get("merchant_sim", 0.0)), 3),
                    "amount_delta": round(float(r.get("amount_delta", 0.0)), 2),
                    "date_delta_days": int(r.get("date_delta_days", 0)),
                    "why": r.get("why"),
                })

        results.append({
            "tx_id": tx_dict["tx_id"],
            "decision": decision["decision"],
            "reason_codes": decision["reason_codes"],
            "human_reasons": human_reasons,
            "explanation": decision["explanation"],
            "selected_ledger_id": selected["ledger_id"] if selected else None,
            "selected_score_proxy": proxy,
            "cand_count": len(cands),
            "recommendations": recommendations or None
        })

    return {
        "summary": {"processed": len(results), "auto": auto_ct, "review": review_ct, "no_match": nomatch_ct},
        "results": results,
    }

# -------------------- Builders (Statement + Ledger) --------------------
def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in cands:
        cand = cand.strip().lower()
        for lc, orig in cols.items():
            if cand == lc or lc.startswith(cand) or cand in lc:
                return orig
    return None

def build_inputs_from_statement_and_ledger(
    statement_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    *,
    filter_zero_diff: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (statement_norm, ledger_norm)

    statement_norm: tx_id, post_date, amount, currency, description
    ledger_norm:    ledger_id, tx_date, total_amount, currency, description_canonical
    """
    # ----- Statement (bank lines) -----
    S_DATE = _pick(statement_df, "transaction date", "posting date", "date", "statement date", "value date")
    S_AMT  = _pick(statement_df, "statement amount", "amount", "txn amount", "debit/credit amount")
    S_DESC = _pick(statement_df, "description", "narration", "payee", "beneficiary", "details", "memo")
    S_CCY  = _pick(statement_df, "currency", "curr", "ccy", "currency code")
    S_DIFF = _pick(statement_df, "difference", "unapplied amount", "residual", "variance")
    if not all([S_DATE, S_AMT, S_DESC]):
        raise ValueError("Statement file: required columns not found (date/amount/description).")

    lines = statement_df.copy()
    if filter_zero_diff and S_DIFF and S_DIFF in lines.columns:
        lines = lines[pd.to_numeric(lines[S_DIFF], errors="coerce").fillna(0.0) != 0.0]

    statement_norm = pd.DataFrame({
        "tx_id": np.arange(len(lines)).astype(str),
        "post_date": pd.to_datetime(lines[S_DATE], errors="coerce"),
        "amount": pd.to_numeric(lines[S_AMT], errors="coerce"),
        "currency": lines[S_CCY].map(_canon_ccy) if S_CCY in lines.columns else DEFAULT_CCY,
        "description": lines[S_DESC].astype(str),
    }).dropna(subset=["post_date", "amount"]).reset_index(drop=True)

    # ----- Ledger (ERP entries) -----
    L_DATE = _pick(ledger_df, "posting date", "date", "document date", "entry date", "postingdate", "value date")
    L_DESC = _pick(ledger_df, "description", "narration", "memo", "details", "posting description")
    L_CCY  = _pick(ledger_df, "currency code", "currency", "curr", "ccy")
    L_ID   = _pick(ledger_df, "ledger_id", "document no.", "document no", "external doc no", "entry no", "id", "no.", "doc no", "document number")

    avail = {str(c).lower(): c for c in ledger_df.columns}
    L_AMT_REM = next((orig for lc, orig in avail.items() if "remaining amount" in lc), None)
    L_AMT_RAW = next((orig for lc, orig in avail.items()
                      if re.search(r"(^|[^a-z])amount([^a-z]|$)", lc) and "remaining" not in lc and "applied" not in lc), None)
    L_DEBIT   = _pick(ledger_df, "debit amount", "debit", "dr amount")
    L_CREDIT  = _pick(ledger_df, "credit amount", "credit", "cr amount")

    if not all([L_DATE, L_DESC]):
        raise ValueError("Ledger file: required columns not found (date/description).")

    led = ledger_df.copy()
    amt = pd.to_numeric(led.get(L_AMT_REM), errors="coerce") if L_AMT_REM else pd.Series(np.nan, index=led.index)
    if amt.isna().all() and L_AMT_RAW:
        amt = pd.to_numeric(led.get(L_AMT_RAW), errors="coerce")
    if (amt.isna().all() or (amt.fillna(0) == 0).all()) and (L_DEBIT or L_CREDIT):
        dr = pd.to_numeric(led.get(L_DEBIT), errors="coerce").fillna(0.0) if L_DEBIT else 0.0
        cr = pd.to_numeric(led.get(L_CREDIT), errors="coerce").fillna(0.0) if L_CREDIT else 0.0
        amt = (dr - cr).abs()

    ledger_norm = pd.DataFrame({
        "ledger_id": (led[L_ID].astype(str) if L_ID else led.reset_index().index.astype(str)),
        "tx_date": pd.to_datetime(led[L_DATE], errors="coerce"),
        "total_amount": pd.to_numeric(amt, errors="coerce"),
        "currency": (led[L_CCY].map(_canon_ccy) if L_CCY in led.columns else DEFAULT_CCY),
        "description_canonical": led[L_DESC].astype(str),
    }).dropna(subset=["tx_date", "total_amount"]).reset_index(drop=True)

    return statement_norm, ledger_norm

# -------- Convenience (local tests) ----------
def _read_any(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=None) if p.endswith((".xlsx", ".xls")) else pd.read_csv(path)

def evaluate_from_files(
    statement_path: str,
    ledger_path: str,
    *,
    filter_zero_diff: bool = False,
    max_rows: int = MAX_ROWS
) -> Dict[str, Any]:
    st_obj = _read_any(statement_path)
    lg_obj = _read_any(ledger_path)

    def _to_df(x):
        if isinstance(x, dict):  # excel sheets
            for _, df in x.items():
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    return df
            return list(x.values())[0]
        return x

    st_df = _to_df(st_obj)
    lg_df = _to_df(lg_obj)

    st_norm, lg_norm = build_inputs_from_statement_and_ledger(st_df, lg_df, filter_zero_diff=filter_zero_diff)
    return evaluate_llm(st_norm, lg_norm, max_rows=max_rows)
