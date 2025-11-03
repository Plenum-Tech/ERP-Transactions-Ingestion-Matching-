from __future__ import annotations
import os, json, unicodedata, re
from typing import List, Dict, Any, Optional
import pandas as pd
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
MAX_ROWS           = int(os.getenv("LLM_EVAL_MAX_ROWS", "200"))  # guardrails

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

def _block_candidates(tx: Dict[str, Any], receipts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shortlist by currency, date±D, amount±band, then sort by coarse merchant similarity.
    """
    amt = float(tx["amount"])
    band = _amount_band(amt)
    post_date = pd.to_datetime(tx["post_date"], errors="coerce")
    mcanon = _canon(tx.get("merchant_desc") or tx.get("merchant_canonical") or "")

    lo = post_date - pd.Timedelta(days=DATE_WINDOW_DAYS)
    hi = post_date + pd.Timedelta(days=DATE_WINDOW_DAYS)
    cur = tx["currency"]

    df = receipts_df[
        (receipts_df["currency"] == cur)
        & (receipts_df["tx_date"].between(lo, hi))
        & (receipts_df["total_amount"].between(amt - band, amt + band))
    ].copy()

    df["merchant_sim"] = df["merchant_canonical"].apply(lambda s: fuzz.token_set_ratio(mcanon, s) / 100.0)
    df = df.sort_values("merchant_sim", ascending=False).head(CANDIDATE_LIMIT)

    # helpful deltas
    df["amount_delta"] = (df["total_amount"] - amt).abs()
    df["date_delta_days"] = df["tx_date"].apply(lambda d: _days(post_date, d))
    return df

# -------------------- Reasons: enums & humanization --------------------
_REASON_MAP = {
    "auto_match": "AUTO_MATCH",
    "low_confidence": "LOW_CONFIDENCE",
    "multiple_possible_matches": "MULTIPLE_POSSIBLE_MATCHES",
    "amount_delta_exceeds_auto_threshold": "AMOUNT_DELTA_EXCEEDS_GUARD",
    "date_delta_days_exceeds_auto_threshold": "DATE_DELTA_EXCEEDS_GUARD",
    "amount_delta_high": "AMOUNT_DELTA_EXCEEDS_GUARD",
    "date_delta_days_high": "DATE_DELTA_EXCEEDS_GUARD",
    "missing_receipt": "MISSING_RECEIPT",
}

_ALLOWED_REASON_CODES = list({v for v in _REASON_MAP.values()} | {
    "AUTO_MATCH", "LOW_CONFIDENCE", "MULTIPLE_POSSIBLE_MATCHES",
    "AMOUNT_DELTA_EXCEEDS_GUARD", "DATE_DELTA_EXCEEDS_GUARD", "MISSING_RECEIPT"
})

def _norm_reasons(reasons: List[str]) -> List[str]:
    out: List[str] = []
    for r in reasons or []:
        key = r.strip().lower()
        out.append(_REASON_MAP.get(key, r.upper()))
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    # clamp to allowed set
    return [r for r in uniq if r in _ALLOWED_REASON_CODES] or ["LOW_CONFIDENCE"]

def _fmt_aed(x: float) -> str:
    return f"AED {x:,.2f}"

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
                    f"Matched: amount difference {_fmt_aed(0 if amt_delta is None else amt_delta)} "
                    f"(allowed up to {_fmt_aed(max(1.0, 0.01*tx_amount))}), "
                    f"date difference {0 if date_delta is None else date_delta} day(s) "
                    f"(policy ≤ {date_guard_days}), "
                    f"merchant similarity {merch_sim:.2f}."
                )
            else:
                msg = "Amounts and date align within policy, and the merchant name is a clear match."
        elif code == "LOW_CONFIDENCE":
            msg = "Candidate exists but confidence is low; needs a quick human check."
        elif code == "MULTIPLE_POSSIBLE_MATCHES":
            msg = "More than one plausible receipt; please pick the correct one."
        elif code == "AMOUNT_DELTA_EXCEEDS_GUARD":
            msg = (
                "Amount difference exceeds the auto-match tolerance "
                f"(delta {_fmt_aed(amt_delta or 0)}, policy ≤ {_fmt_aed(max(1.0, 0.01*tx_amount))})."
            )
        elif code == "DATE_DELTA_EXCEEDS_GUARD":
            msg = (
                "Date difference exceeds the auto-match tolerance "
                f"({_plural(date_delta or 0, 'day')}, policy ≤ {date_guard_days} days)."
            )
        elif code == "MISSING_RECEIPT":
            msg = "No receipt candidate found in the allowed date/amount window."
        else:
            msg = "Needs review due to policy checks."
        msgs.append({"code": code, "message": msg})

    if not msgs:
        msgs.append({"code": "LOW_CONFIDENCE", "message": "Candidate exists but confidence is low; needs review."})
    return msgs

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
            "currency": tx["currency"],
            "merchant_canonical": _canon(tx.get("merchant_desc") or tx.get("merchant_canonical") or "")
        },
        "candidates": [
            {
                "index": i,
                "receipt_id": c["receipt_id"],
                "tx_date": str(pd.to_datetime(c["tx_date"]).date()),
                "total_amount": float(c["total_amount"]),
                "merchant_canonical": c["merchant_canonical"],
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
        temperature=0.0,  # prefer determinism
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

# -------------------- Main entry --------------------
def evaluate_llm(mismatches_df: pd.DataFrame, receipts_df: pd.DataFrame, max_rows: int = MAX_ROWS) -> Dict[str, Any]:
    # normalize columns
    mm = mismatches_df.rename(columns=str.lower).copy()
    rc = receipts_df.rename(columns=str.lower).copy()

    # required columns
    req_mm = {"tx_id", "post_date", "amount", "currency", "merchant_desc"}
    req_rc = {"receipt_id", "tx_date", "total_amount", "currency", "merchant_canonical"}
    missing_mm = [c for c in req_mm if c not in mm.columns]
    missing_rc = [c for c in req_rc if c not in rc.columns]
    if missing_mm or missing_rc:
        raise ValueError(f"Missing columns — mismatches:{missing_mm} receipts:{missing_rc}")

    # coerce types
    mm["post_date"] = pd.to_datetime(mm["post_date"], errors="coerce")
    rc["tx_date"] = pd.to_datetime(rc["tx_date"], errors="coerce")
    rc["total_amount"] = pd.to_numeric(rc["total_amount"], errors="coerce")
    mm["amount"] = pd.to_numeric(mm["amount"], errors="coerce")

    # drop bad rows
    mm = mm.dropna(subset=["post_date", "amount", "currency"])
    rc = rc.dropna(subset=["tx_date", "total_amount", "currency"])

    # canonicalize merchant columns
    mm["merchant_canonical"] = mm["merchant_desc"].fillna("").map(_canon)
    rc["merchant_canonical"] = rc["merchant_canonical"].fillna("").map(_canon)

    results: List[Dict[str, Any]] = []
    auto_ct = review_ct = nomatch_ct = 0

    for _, tx in mm.head(max_rows).iterrows():
        tx_dict = tx.to_dict()
        cands_df = _block_candidates(tx_dict, rc)
        cands = cands_df.to_dict(orient="records")

        prompt = _build_prompt(tx_dict, cands)
        decision = _call_llm(prompt)

        # ---------- SAFETY NETS ----------
        # If AUTO but no index, and exactly one candidate, pick 0
        if decision["decision"] == "AUTO" and decision.get("selected_index") is None and len(cands) == 1:
            decision["selected_index"] = 0
            if not decision.get("reason_codes"):
                decision["reason_codes"] = ["AUTO_MATCH"]
            if not decision.get("explanation"):
                decision["explanation"] = "Single candidate satisfies rules; selecting index 0."

        # If AUTO but selected_index out of range, downgrade to REVIEW
        if decision["decision"] == "AUTO" and (
            decision.get("selected_index") is None or not (0 <= int(decision["selected_index"]) < len(cands))
        ):
            decision["decision"] = "REVIEW"
            decision["reason_codes"] = list(set((decision.get("reason_codes") or []) + ["MULTIPLE_POSSIBLE_MATCHES"]))
            decision["explanation"] = (decision.get("explanation") or "") + " (Invalid or missing selected_index)"

        # ---------- POST-LLM GUARD OVERRIDE (single candidate only) ----------
        if len(cands) == 1:
            c = cands[0]
            amt = float(tx_dict["amount"])
            amt_guard = max(1.0, 0.01 * amt)  # max(AED 1, 1%)
            date_guard = 2
            within = (c["amount_delta"] <= amt_guard) and (c["date_delta_days"] <= date_guard)
            if within and decision["decision"] != "AUTO":
                decision["decision"] = "AUTO"
                decision["selected_index"] = 0
                decision["reason_codes"] = ["AUTO_MATCH"]
                decision["explanation"] = "Single candidate within guards; auto-upgraded."
            elif (not within) and decision["decision"] == "AUTO":
                # downgrade if LLM was too optimistic
                decision["decision"] = "REVIEW"
                rcodes = []
                if c["amount_delta"] > amt_guard:
                    rcodes.append("AMOUNT_DELTA_EXCEEDS_GUARD")
                if c["date_delta_days"] > date_guard:
                    rcodes.append("DATE_DELTA_EXCEEDS_GUARD")
                decision["reason_codes"] = rcodes or ["LOW_CONFIDENCE"]
                decision["selected_index"] = None
                decision["explanation"] = "Auto downgraded; guard check failed."

        # ---------- Finalize selection & counts ----------
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
            # NO_MATCH
            nomatch_ct += 1

        # ---------- Normalize reasons, compute proxy, humanize ----------
        decision["reason_codes"] = _norm_reasons(decision.get("reason_codes"))

        if selected is None:
            proxy = None
        else:
            amt = float(tx_dict["amount"])
            band = max(5.0, 0.02 * amt)
            amt_comp = max(0.0, 1.0 - selected["amount_delta"] / max(1e-9, band))
            date_comp = max(0.0, 1.0 - selected["date_delta_days"] / 5.0)
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

        results.append({
            "tx_id": tx_dict["tx_id"],
            "decision": decision["decision"],
            "reason_codes": decision["reason_codes"],   # machine-friendly
            "human_reasons": human_reasons,             # [{code, message}]
            "explanation": decision["explanation"],     # LLM one-liner (optional)
            "selected_receipt_id": selected["receipt_id"] if selected else None,
            "selected_score_proxy": proxy,
            "cand_count": len(cands),
        })

    return {
        "summary": {"processed": len(results), "auto": auto_ct, "review": review_ct, "no_match": nomatch_ct},
        "results": results,
    }




