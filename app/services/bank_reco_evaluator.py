from __future__ import annotations

import io
import os
import re
import json
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable

import pandas as pd
import numpy as np

# --- Logger -------------------------------------------------------------------
try:
    from app.utils.logger import get_logger  # your project logger
    log = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

# --- Optional OpenAI (LLM rationale + embeddings) -----------------------------
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
_DISABLE_EMBEDDINGS = os.getenv("DISABLE_EMBEDDINGS", "false").lower() == "true"

_OPENAI_ENABLED = True if os.getenv("OPENAI_API_KEY","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") else False
_openai_client = None
if _OPENAI_ENABLED:
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except Exception as e:
        log.warning("OpenAI client not available: %s", e)
        _OPENAI_ENABLED = False

_EMBED_ENABLED = _OPENAI_ENABLED and not _DISABLE_EMBEDDINGS

# --- Tunables (env overrides) -------------------------------------------------
RECO_K_PER_LINE = int(os.getenv("RECO_K_PER_LINE", "3"))
RECO_TOL_MULT = float(os.getenv("RECO_TOL_MULT", "5.0"))
RECO_MIN_SCORE = float(os.getenv("RECO_MIN_SCORE", "0.3"))
RECO_MAX_CANDIDATES_PER_LINE = int(os.getenv("RECO_MAX_CANDS", "50"))
DATE_WINDOW_DAYS = int(os.getenv("DATE_WINDOW_DAYS", "10"))
DATE_BONUS = float(os.getenv("DATE_BONUS", "0.08"))
TEXT_TOKEN_WEIGHT = float(os.getenv("TEXT_TOKEN_WEIGHT", "0.15"))
TEXT_SEM_WEIGHT = float(os.getenv("TEXT_SEM_WEIGHT", "0.20"))

# ==============================================================================
# Parsing & Normalization
# ==============================================================================

_AMOUNT_COL_ALIASES = [
    "amount", "statement amount", "txn amount", "value", "value amount",
    "amt", "statementamount", "statement_amount",
]
_DEBIT_COL_ALIASES = [
    "debit", "debit amount", "dr", "withdrawal", "withdrawal amount",
    "debitamount", "debit_amount",
]
_CREDIT_COL_ALIASES = [
    "credit", "credit amount", "cr", "deposit", "deposit amount",
    "creditamount", "credit_amount",
]
_DESC_ALIASES = [
    "description", "narration", "memo", "details", "posting description",
    "postingdescription", "posting_description",
]
_DATE_ALIASES = [
    "transaction date", "document date", "value date", "posting date",
    "entry date", "date", "statement date",
]
_APPLIED_ALIASES = ["applied amount", "applied", "applied_amount", "appliedamount"]
_DIFF_ALIASES = ["difference", "diff", "delta", "variance", "difference_amount"]

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _find_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    cols = {c: _canon(c) for c in df.columns}
    a_norm = [a.strip().lower() for a in aliases]
    for c, cn in cols.items():
        for a in a_norm:
            if a == cn or a in cn:
                return c
    return None

def _parse_amount(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    negative = False
    if "(" in s and ")" in s:
        negative = True
        s = s.replace("(", "").replace(")", "")
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return -abs(v) if negative else v

def _signed_from_debit_credit(debit: Any, credit: Any) -> Optional[float]:
    d = _parse_amount(debit) or 0.0
    c = _parse_amount(credit) or 0.0
    if d and not c:
        return -abs(d)
    if c and not d:
        return abs(c)
    if d and c:
        return abs(c) if abs(c) >= abs(d) else -abs(d)
    return None

def _classify_intent(text: str) -> str:
    t = _canon(text)
    tokens = set(t.split())
    if "salary" in tokens or "payroll" in tokens:
        return "salary"
    if "atm" in tokens or "withdrawal" in tokens:
        return "atm"
    if "interest" in tokens:
        return "interest"
    if re.search(r"\bcheque\b|\bchq\b|\bcheck\b(?!ing\b)", t):
        return "cheque"
    if "deposit" in tokens:
        return "deposit"
    if {"purchase", "pos", "store", "merchant"} & tokens:
        return "purchase"
    return "other"

def _intent_bonus(line_intent: str, ledger_intent: str) -> float:
    if line_intent == ledger_intent and line_intent != "other":
        return 0.12
    if line_intent in {"atm", "purchase"} and ledger_intent in {"atm", "purchase"}:
        return 0.07
    if line_intent in {"salary", "deposit"} and ledger_intent in {"salary", "deposit"}:
        return 0.08
    if line_intent in {"interest"} and ledger_intent in {"interest"}:
        return 0.1
    return 0.0

def _token_overlap(a: str, b: str) -> float:
    A = set(_canon(a).split())
    B = set(_canon(b).split())
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union

def _direction_bonus(line_signed: float, ledger_side: str) -> float:
    if line_signed is None:
        return 0.0
    if line_signed < 0 and ledger_side == "credit":
        return 0.08
    if line_signed > 0 and ledger_side == "debit":
        return 0.08
    return 0.0

def _abs_equal(a: float, b: float, tol_abs: float, tol_pct: float) -> bool:
    if a is None or b is None:
        return False
    da = abs(abs(a) - abs(b))
    if da <= tol_abs:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return (da / denom) <= tol_pct

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[\uFEFF\u200B\u00A0]", "", str(c)).strip()
        for c in df.columns
    ]
    return df

def _try_parse_date(val: Any) -> Optional[pd.Timestamp]:
    try:
        dt = pd.to_datetime(val, errors="coerce", dayfirst=True, utc=False)
        if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
            return dt.normalize()
    except Exception:
        pass
    return None

def _read_any_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(bio)
            return _clean_headers(df)
        except Exception as e:
            raise ValueError(f"Failed reading Excel: {e}")
    attempts = [
        {"sep": None, "engine": "python"},
        {"delim_whitespace": True, "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": ",", "engine": "python"},
    ]
    for kw in attempts:
        bio.seek(0)
        try:
            df = pd.read_csv(bio, **kw)
            return _clean_headers(df)
        except Exception:
            continue
    raise ValueError("Failed to read as CSV (tried multiple delimiters).")

# ==============================================================================
# Public API: normalize tables
# ==============================================================================

def normalize_statement_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty statement file")

    col_amount = _find_col(df, _AMOUNT_COL_ALIASES)
    col_debit  = _find_col(df, _DEBIT_COL_ALIASES)
    col_credit = _find_col(df, _CREDIT_COL_ALIASES)
    col_desc   = _find_col(df, _DESC_ALIASES) or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    col_applied = _find_col(df, _APPLIED_ALIASES)
    col_diff    = _find_col(df, _DIFF_ALIASES)
    col_date    = _find_col(df, _DATE_ALIASES)

    out_rows = []
    for idx, row in df.iterrows():
        raw_desc = str(row.get(col_desc, "")) if col_desc else ""
        applied = _parse_amount(row.get(col_applied)) if col_applied else None
        diffv   = _parse_amount(row.get(col_diff)) if col_diff else None
        dt = _try_parse_date(row.get(col_date)) if col_date else None

        if col_amount:
            signed = _parse_amount(row.get(col_amount))
        else:
            signed = _signed_from_debit_credit(row.get(col_debit), row.get(col_credit))

        if signed is None:
            for c in df.columns:
                if c in {col_desc, col_applied, col_diff, col_debit, col_credit, col_amount}:
                    continue
                v = _parse_amount(row.get(c))
                if v is not None:
                    signed = v
                    break
        if signed is None:
            continue

        out_rows.append({
            "row_id": int(idx),
            "raw_desc": raw_desc,
            "signed_amount": float(signed),
            "intent": _classify_intent(raw_desc),
            "date": dt,
            "applied_amount": applied,
            "difference": diffv,
            "raw": {c: (None if (isinstance(row.get(c), float) and math.isnan(row.get(c))) else row.get(c)) for c in df.columns}
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("No parseable rows in statement file")
    return out.reset_index(drop=True)

def _col_norm(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = (name or "").replace("\u00A0", " ").replace("\u200B", "")
    name = re.sub(r"\s+", " ", name).strip().lower()
    return re.sub(r"[^a-z0-9 ]+", "", name)

def _lookup_col_by_tokens(df: pd.DataFrame, tokens: List[str]) -> Optional[str]:
    toks = [t.lower() for t in tokens]
    for c in df.columns:
        cn = _col_norm(c)
        if all(t in cn for t in toks):
            return c
    return None

def normalize_ledger_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty ledger file")

    preferred = [
        _find_col(df, _DESC_ALIASES),
        "Bal. Account Type", "Bal. Account No.", "External Document No.", "Statement Status",
        "G/L Account", "Customer", "Vendor", "Bal Account Type", "Bal Account No", "Document No."
    ]
    desc_cols = [c for c in preferred if c and c in df.columns]
    if not desc_cols:
        nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        desc_cols = nonnum[:3] if nonnum else [df.columns[0]]

    col_date = _lookup_col_by_tokens(df, ["posting", "date"]) or _find_col(df, _DATE_ALIASES)

    candidates_map: List[Tuple[str, str]] = []
    debit_lcy = _lookup_col_by_tokens(df, ["debit", "amount", "lcy"])
    credit_lcy = _lookup_col_by_tokens(df, ["credit", "amount", "lcy"])
    debit_std = _lookup_col_by_tokens(df, ["debit", "amount"]) or _find_col(df, _DEBIT_COL_ALIASES) or "Debit Amount"
    credit_std = _lookup_col_by_tokens(df, ["credit", "amount"]) or _find_col(df, _CREDIT_COL_ALIASES) or "Credit Amount"

    for side, col in [
        ("debit", debit_lcy),
        ("credit", credit_lcy),
        ("debit", debit_std if debit_std in df.columns else None),
        ("credit", credit_std if credit_std in df.columns else None),
    ]:
        if col and col in df.columns and (side, col) not in candidates_map:
            candidates_map.append((side, col))

    if not candidates_map:
        for c in df.columns:
            cn = _col_norm(c)
            if "debit" in cn and "amount" in cn:
                candidates_map.append(("debit", c))
            elif "credit" in cn and "amount" in cn:
                candidates_map.append(("credit", c))

    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        parts = [str(row.get(c, "")) for c in desc_cols]
        if "Document No." in df.columns:
            parts.append(f"Document No.: {row.get('Document No.', '')}")
        if col_date:
            parts.append(f"Posting Date: {row.get(col_date, '')}")
        raw_desc = " | ".join([p for p in parts if p]).strip()
        intent = _classify_intent(raw_desc)
        dt = _try_parse_date(row.get(col_date)) if col_date else None

        found_any = False
        for side, col in candidates_map:
            amt = _parse_amount(row.get(col))
            if amt is None or abs(amt) < 1e-12:
                continue
            rows.append({
                "row_id": int(idx),
                "parent_row_id": int(idx),
                "side": side,
                "amount": float(abs(amt)),
                "intent": intent,
                "raw_desc": raw_desc,
                "date": dt,
                "raw": {c: (None if (isinstance(row.get(c), float) and math.isnan(row.get(c))) else row.get(c)) for c in df.columns}
            })
            found_any = True

        if not found_any:
            numeric_cells = []
            for c in df.columns:
                v = _parse_amount(row.get(c))
                if v is not None and abs(v) > 0:
                    numeric_cells.append((c, v))
            if len(numeric_cells) >= 1:
                for c, v in numeric_cells:
                    cn = _col_norm(c)
                    if "debit" in cn:
                        rows.append({
                            "row_id": int(idx),
                            "parent_row_id": int(idx),
                            "side": "debit",
                            "amount": float(abs(v)),
                            "intent": intent,
                            "raw_desc": raw_desc,
                            "date": dt,
                            "raw": {cc: (None if (isinstance(row.get(cc), float) and math.isnan(row.get(cc))) else row.get(cc)) for cc in df.columns}
                        })
                    elif "credit" in cn:
                        rows.append({
                            "row_id": int(idx),
                            "parent_row_id": int(idx),
                            "side": "credit",
                            "amount": float(abs(v)),
                            "intent": intent,
                            "raw_desc": raw_desc,
                            "date": dt,
                            "raw": {cc: (None if (isinstance(row.get(cc), float) and math.isnan(row.get(cc))) else row.get(cc)) for cc in df.columns}
                        })

    if not rows:
        headers_debug = [f"{c} -> {_col_norm(c)}" for c in df.columns]
        raise ValueError(f"No usable debit/credit amounts in ledger file. Headers seen: {headers_debug}")

    out = pd.DataFrame(rows).drop_duplicates(subset=["parent_row_id", "side", "amount", "raw_desc"]).reset_index(drop=True)
    return out

# ==============================================================================
# Embeddings helpers (optional)
# ==============================================================================

def _embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    if not _EMBED_ENABLED or not texts:
        return None
    try:
        resp = _openai_client.embeddings.create(model=_OPENAI_EMBED_MODEL, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)
    except Exception as e:
        log.warning("Embeddings call failed; continuing without semantic sim: %s", e)
        return None

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)

# ==============================================================================
# Matching Logic
# ==============================================================================

def _date_bonus(ldt: Optional[pd.Timestamp], cdt: Optional[pd.Timestamp]) -> float:
    if ldt is None or cdt is None:
        return 0.0
    try:
        gap = abs((ldt - cdt).days)
        return DATE_BONUS if gap <= DATE_WINDOW_DAYS else 0.0
    except Exception:
        return 0.0

def _score_candidate(
    line_row: Dict[str, Any],
    cand: Dict[str, Any],
    tol_abs: float,
    tol_pct: float,
    line_vec: Optional[np.ndarray] = None,
    ledger_vec: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, Any]]:
    amt_line = abs(line_row["signed_amount"])
    amt_ledg = abs(cand["amount"])
    within = _abs_equal(amt_line, amt_ledg, tol_abs, tol_pct)
    diff = abs(amt_line - amt_ledg)
    denom = max(amt_line, amt_ledg, 1.0)
    amount_score = max(0.0, 1.0 - (diff / denom)) if within else max(0.0, 0.85 - (diff / denom))

    dir_bonus = _direction_bonus(line_row["signed_amount"], cand["side"])
    intent_bonus = _intent_bonus(line_row.get("intent", "other"), cand.get("intent", "other"))
    token_sim = _token_overlap(line_row.get("raw_desc", ""), cand.get("raw_desc", ""))
    token_bonus = TEXT_TOKEN_WEIGHT * token_sim

    sem_sim = 0.0
    if line_vec is not None and ledger_vec is not None:
        sem_sim = _cosine_sim(line_vec, ledger_vec)
    sem_bonus = TEXT_SEM_WEIGHT * sem_sim
    d_bonus = _date_bonus(line_row.get("date"), cand.get("date"))

    score = amount_score + dir_bonus + intent_bonus + token_bonus + sem_bonus + d_bonus
    score = max(0.0, min(1.0, score))
    reasons = {
        "amount_score": round(amount_score, 4),
        "dir_bonus": round(dir_bonus, 4),
        "intent_bonus": round(intent_bonus, 4),
        "text_token_bonus": round(token_bonus, 4),
        "semantic_sim": round(sem_sim, 3),
        "date_bonus": round(d_bonus, 4),
        "diff": diff,
    }
    return score, reasons

def _greedy_one_to_one(
    lines: pd.DataFrame,
    ledger: pd.DataFrame,
    tol_abs: float,
    tol_pct: float,
    min_score: float,
    line_embs: Optional[np.ndarray],
    ledger_embs: Optional[np.ndarray],
) -> Tuple[List[Dict[str, Any]], set, set]:
    matches: List[Dict[str, Any]] = []
    used_lines: set = set()
    used_ledgers: set = set()

    ledger_records = [ledger.iloc[i].to_dict() for i in range(len(ledger))]
    for i in range(len(lines)):
        line = lines.iloc[i].to_dict()
        best: Tuple[float, Dict[str, Any], int] = (0.0, {}, -1)

        for j, cand in enumerate(ledger_records):
            if j in used_ledgers:
                continue
            lv = line_embs[i] if line_embs is not None and i < len(line_embs) else None
            cv = ledger_embs[j] if ledger_embs is not None and j < len(ledger_embs) else None
            score, reasons = _score_candidate(line, cand, tol_abs, tol_pct, lv, cv)
            if score > best[0]:
                best = (score, reasons, j)

        if best[0] >= min_score and best[2] >= 0:
            j = best[2]
            used_lines.add(i)
            used_ledgers.add(j)
            cand = ledger_records[j]
            matches.append({
                "type": "1to1",
                "line_index": int(i),
                "ledger_index": int(j),
                "line_amount": float(lines.iloc[i]["signed_amount"]),
                "ledger_amount": float(cand["amount"]),
                "ledger_side": cand["side"],
                "score": round(best[0], 4),
                "intent_line": lines.iloc[i].get("intent", "other"),
                "intent_ledger": cand.get("intent", "other"),
                "rationale": best[1],
            })

    return matches, used_lines, used_ledgers

def _two_sum_pairs(
    targets: List[Tuple[int, float]],
    tol_abs: float,
    tol_pct: float,
    target_value: float
) -> Optional[Tuple[Tuple[int, float], Tuple[int, float]]]:
    T = abs(target_value)
    n = len(targets)
    for i in range(n):
        idx1, a1 = targets[i]
        for j in range(i + 1, n):
            idx2, a2 = targets[j]
            if _abs_equal(a1 + a2, T, tol_abs, tol_pct):
                return (targets[i], targets[j])
    return None

def _composites_line_from_ledger(
    lines: pd.DataFrame,
    ledger: pd.DataFrame,
    unused_line_ids: set,
    unused_ledger_ids: set,
    tol_abs: float,
    tol_pct: float,
    prefer_fee_cap: float = 600.0,
    min_score: float = 0.76
) -> Tuple[List[Dict[str, Any]], set, set]:
    matches: List[Dict[str, Any]] = []
    ledger_targets = []
    for j in sorted(list(unused_ledger_ids)):
        amt = float(abs(ledger.iloc[j]["amount"]))
        ledger_targets.append((j, amt))

    for i in sorted(list(unused_line_ids)):
        line_amt = float(abs(lines.iloc[i]["signed_amount"]))
        result = _two_sum_pairs(ledger_targets, tol_abs, tol_pct, line_amt)
        if not result:
            continue
        (j1, a1), (j2, a2) = result
        if j1 not in unused_ledger_ids or j2 not in unused_ledger_ids:
            continue

        fee_bonus = 0.08 if (a1 <= prefer_fee_cap or a2 <= prefer_fee_cap) else 0.0
        total = a1 + a2
        diff = abs(total - line_amt)
        denom = max(total, line_amt, 1.0)
        base = max(0.0, 1.0 - (diff / denom))
        score = min(1.0, base + fee_bonus)

        if score >= min_score:
            matches.append({
                "type": "composite_line=ledger1+ledger2",
                "line_index": int(i),
                "ledger_indices": [int(j1), int(j2)],
                "line_amount": float(lines.iloc[i]["signed_amount"]),
                "ledger_amounts": [float(ledger.iloc[j1]["amount"]), float(ledger.iloc[j2]["amount"])],
                "score": round(score, 4),
                "rationale": {"diff": diff, "fee_bonus": fee_bonus}
            })
            unused_line_ids.remove(i)
            unused_ledger_ids.remove(j1)
            unused_ledger_ids.remove(j2)

    return matches, unused_line_ids, unused_ledger_ids

def _composites_ledger_from_lines(
    lines: pd.DataFrame,
    ledger: pd.DataFrame,
    unused_line_ids: set,
    unused_ledger_ids: set,
    tol_abs: float,
    tol_pct: float,
    min_score: float = 0.76
) -> Tuple[List[Dict[str, Any]], set, set]:
    matches: List[Dict[str, Any]] = []
    line_targets = []
    for i in sorted(list(unused_line_ids)):
        amt = float(abs(lines.iloc[i]["signed_amount"]))
        line_targets.append((i, amt))

    for j in sorted(list(unused_ledger_ids)):
        led_amt = float(abs(ledger.iloc[j]["amount"]))
        result = _two_sum_pairs(line_targets, tol_abs, tol_pct, led_amt)
        if not result:
            continue
        (i1, a1), (i2, a2) = result
        if i1 not in unused_line_ids or i2 not in unused_line_ids:
            continue

        total = a1 + a2
        diff = abs(total - led_amt)
        denom = max(total, led_amt, 1.0)
        base = max(0.0, 1.0 - (diff / denom))
        score = min(1.0, base + 0.02)

        if score >= min_score:
            matches.append({
                "type": "composite_ledger=line1+line2",
                "ledger_index": int(j),
                "line_indices": [int(i1), int(i2)],
                "ledger_amount": float(ledger.iloc[j]["amount"]),
                "line_amounts": [float(lines.iloc[i1]["signed_amount"]), float(lines.iloc[i2]["signed_amount"])],
                "score": round(score, 4),
                "rationale": {"diff": diff}
            })
            unused_ledger_ids.remove(j)
            unused_line_ids.remove(i1)
            unused_line_ids.remove(i2)

    return matches, unused_line_ids, unused_ledger_ids

# ==============================================================================
# Recommendations
# ==============================================================================

def _notes_for_reco(amt_line: float, amt_ledg: float, reasons: Dict[str, Any], date_gap: Optional[int]) -> List[str]:
    notes = []
    diff = abs(amt_line - amt_ledg)
    if diff > 0:
        notes.append(f"Amount near-miss (Δ={round(diff, 2)})")
    if reasons.get("semantic_sim", 0) >= 0.6:
        notes.append("High semantic similarity (possible naming drift)")
    elif reasons.get("text_token_bonus", 0) >= 0.08:
        notes.append("High token overlap (similar wording)")
    if reasons.get("intent_bonus", 0) > 0:
        notes.append("Intent aligns")
    if reasons.get("dir_bonus", 0) > 0:
        notes.append("Direction consistent (statement vs ledger)")
    if date_gap is not None:
        if date_gap <= DATE_WINDOW_DAYS:
            notes.append(f"Dates close (±{DATE_WINDOW_DAYS}d window)")
        else:
            notes.append(f"Date gap {date_gap}d (still consider if other signals strong)")
    return notes

def _recommendations_for_lines(
    lines: pd.DataFrame,
    ledger: pd.DataFrame,
    remaining_lines: set,
    remaining_ledgers: set,
    tol_abs: float,
    tol_pct: float,
    line_embs: Optional[np.ndarray],
    ledger_embs: Optional[np.ndarray],
    k_per_line: int = RECO_K_PER_LINE,
    widen_mult: float = RECO_TOL_MULT,
    min_reco_score: float = RECO_MIN_SCORE,
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if not remaining_lines or not remaining_ledgers:
        return recs

    ledger_records = [ledger.iloc[j].to_dict() for j in range(len(ledger))]

    for i in sorted(list(remaining_lines)):
        line = lines.iloc[i].to_dict()
        lv = line_embs[i] if line_embs is not None and i < len(line_embs) else None

        candidates: List[Tuple[float, int, Dict[str, Any]]] = []
        scan_count = 0
        for j in sorted(list(remaining_ledgers)):
            scan_count += 1
            if scan_count > RECO_MAX_CANDIDATES_PER_LINE:
                break
            cand = ledger_records[j]
            cv = ledger_embs[j] if ledger_embs is not None and j < len(ledger_embs) else None

            score, reasons = _score_candidate(line, cand, tol_abs, tol_pct, lv, cv)

            amt_line = abs(line["signed_amount"])
            amt_ledg = abs(cand["amount"])
            within_strict = _abs_equal(amt_line, amt_ledg, tol_abs, tol_pct)
            within_wide = _abs_equal(amt_line, amt_ledg, tol_abs * widen_mult, tol_pct * widen_mult)

            # keep a bit strict but not too strict
            if not within_strict and not within_wide and reasons.get("semantic_sim", 0) < 0.3 and reasons.get("text_token_bonus", 0) < 0.05:
                continue

            candidates.append((score, j, reasons))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        filtered = [c for c in candidates[:max(k_per_line, 1)] if c[0] >= min_reco_score]
        if not filtered:
            continue

        rec_entry = {
            "line_index": int(i),
            "line_amount": float(line["signed_amount"]),
            "line_intent": line.get("intent", "other"),
            "line_desc": line.get("raw_desc", ""),
            "candidates": []
        }
        for score, j, reasons in filtered:
            cand = ledger_records[j]
            amt_line = abs(line["signed_amount"])
            amt_ledg = abs(cand["amount"])
            amt_diff = round(abs(amt_line - amt_ledg), 4)
            date_gap = None
            if isinstance(line.get("date"), pd.Timestamp) and isinstance(cand.get("date"), pd.Timestamp):
                date_gap = abs((line["date"] - cand["date"]).days)
            rec_entry["candidates"].append({
                "ledger_index": int(j),
                "ledger_amount": float(cand["amount"]),
                "ledger_side": cand["side"],
                "ledger_intent": cand.get("intent", "other"),
                "ledger_desc": cand.get("raw_desc", ""),
                "score": round(float(score), 4),
                "reasoning": {
                    "amount_diff": amt_diff,
                    "direction_hint": reasons.get("dir_bonus", 0.0),
                    "intent_bonus": reasons.get("intent_bonus", 0.0),
                    "token_overlap_bonus": reasons.get("text_token_bonus", 0.0),
                    "semantic_sim": reasons.get("semantic_sim", 0.0),
                    "date_bonus": reasons.get("date_bonus", 0.0),
                    "date_gap_days": date_gap
                },
                "notes": _notes_for_reco(amt_line, amt_ledg, reasons, date_gap)
            })
        recs.append(rec_entry)
    return recs

# ==============================================================================
# Orchestrator
# ==============================================================================

def run_matcher(
    statement_bytes: bytes,
    statement_filename: str,
    ledger_bytes: bytes,
    ledger_filename: str,
    tol_abs: float = 2.0,
    tol_pct: float = 0.005,
    enable_llm_summary: bool = False
) -> Dict[str, Any]:
    # --- Load & normalize ------------------------------------------------------
    st_raw = _read_any_table(statement_bytes, statement_filename)
    lg_raw = _read_any_table(ledger_bytes, ledger_filename)
    log.info("Loaded statement rows=%s, ledger rows=%s", len(st_raw), len(lg_raw))

    lines = normalize_statement_df(st_raw)
    ledger_flat = normalize_ledger_df(lg_raw)

    # --- Embeddings (optional) -------------------------------------------------
    line_texts = [f"{r.get('raw_desc','')} [intent:{r.get('intent','other')}]"
                  for r in lines.to_dict(orient="records")]
    ledger_texts = [f"{r.get('raw_desc','')} [intent:{r.get('intent','other')}]"
                    for r in ledger_flat.to_dict(orient="records")]
    line_embs = _embed_texts(line_texts)
    ledger_embs = _embed_texts(ledger_texts)

    emb_active = bool(_EMBED_ENABLED and line_embs is not None and ledger_embs is not None)
    emb_shape_lines = tuple(line_embs.shape) if line_embs is not None else None
    emb_shape_ledger = tuple(ledger_embs.shape) if ledger_embs is not None else None
    log.info("Embeddings active=%s | line_embs=%s | ledger_embs=%s",
             emb_active, emb_shape_lines, emb_shape_ledger)

    # --- Matching --------------------------------------------------------------
    greedy_min_score = 0.82
    m1, used_lines, used_ledgers = _greedy_one_to_one(
        lines, ledger_flat, tol_abs, tol_pct, min_score=greedy_min_score,
        line_embs=line_embs, ledger_embs=ledger_embs
    )

    remaining_lines = set(range(len(lines))) - used_lines
    remaining_ledgers = set(range(len(ledger_flat))) - used_ledgers

    m2, remaining_lines, remaining_ledgers = _composites_line_from_ledger(
        lines, ledger_flat, remaining_lines, remaining_ledgers, tol_abs, tol_pct
    )
    m3, remaining_lines, remaining_ledgers = _composites_ledger_from_lines(
        lines, ledger_flat, remaining_lines, remaining_ledgers, tol_abs, tol_pct
    )

    recs = _recommendations_for_lines(
        lines, ledger_flat, remaining_lines, remaining_ledgers,
        tol_abs, tol_pct, line_embs, ledger_embs,
        k_per_line=RECO_K_PER_LINE, widen_mult=RECO_TOL_MULT, min_reco_score=RECO_MIN_SCORE
    )

    # --- Diagnostics: semantic influence counts --------------------------------
    sem_hits_m1 = sum(
        1 for m in m1
        if float((m.get("rationale", {}) or {}).get("semantic_sim", 0.0)) > 0.0
    )
    rec_cands_total = sum(len(r.get("candidates", [])) for r in recs)
    rec_cands_sem_over_55 = sum(
        1 for r in recs for c in r.get("candidates", [])
        if float((c.get("reasoning", {}) or {}).get("semantic_sim", 0.0)) >= 0.3
    )

    # --- Unmatched -------------------------------------------------------------
    unmatched_lines = lines.iloc[sorted(list(remaining_lines))].to_dict(orient="records")
    unmatched_ledgers = ledger_flat.iloc[sorted(list(remaining_ledgers))].to_dict(orient="records")

    # --- Summary ---------------------------------------------------------------
    summary = {
        "statement_lines": int(len(lines)),
        "ledger_items": int(len(ledger_flat)),
        "matches_1to1": int(len(m1)),
        "matches_composite_line_from_ledger": int(len(m2)),
        "matches_composite_ledger_from_lines": int(len(m3)),
        "matched_total": int(len(m1) + len(m2) + len(m3)),
        "coverage_statement_pct": round(100.0 * (len(lines) - len(remaining_lines)) / max(len(lines), 1), 2),
        "coverage_ledger_pct": round(100.0 * (len(ledger_flat) - len(remaining_ledgers)) / max(len(ledger_flat), 1), 2),
        "recommendations_count": int(rec_cands_total),
        "recommendations_lines": int(len(recs)),

        # New diagnostics
        "embeddings_active": emb_active,
        "embed_model": _OPENAI_EMBED_MODEL if emb_active else None,
        "line_embs_shape": emb_shape_lines,
        "ledger_embs_shape": emb_shape_ledger,
        "matches_with_semantic_signal": int(sem_hits_m1),
        "recommendation_candidates_with_semantic_sim_ge_0_55": int(rec_cands_sem_over_55),

        # Scoring/matching parameters for traceability
        "amount_tolerance": {"abs": float(tol_abs), "pct": float(tol_pct)},
        "greedy_min_score": float(greedy_min_score),
        "weights": {
            "text_token_weight": float(TEXT_TOKEN_WEIGHT),
            "text_sem_weight": float(TEXT_SEM_WEIGHT),
            "date_bonus": float(DATE_BONUS),
        },
        "windows": {
            "date_window_days": int(DATE_WINDOW_DAYS),
            "reco_k_per_line": int(RECO_K_PER_LINE),
            "reco_tol_mult": float(RECO_TOL_MULT),
            "reco_min_score": float(RECO_MIN_SCORE),
        },
    }

    # --- Result payload --------------------------------------------------------
    result: Dict[str, Any] = {
        "summary": summary,
        "matches": {
            "one_to_one": m1,
            "composite_line_from_ledger": m2,
            "composite_ledger_from_lines": m3
        },
        "recommendations": recs,
        "unmatched": {
            "statement_lines": unmatched_lines,
            "ledger_items": unmatched_ledgers
        },
        "normalized_preview": {
            "statement_sample": lines.head(5).to_dict(orient="records"),
            "ledger_sample": ledger_flat.head(5).to_dict(orient="records"),
        },
        # full normalized tables to power Excel side-by-side sheets
        "normalized_full": {
            "statement": lines.to_dict(orient="records"),
            "ledger": ledger_flat.to_dict(orient="records"),
        }
    }

    # --- Optional LLM summary --------------------------------------------------
    if enable_llm_summary and _OPENAI_ENABLED and _openai_client:
        try:
            result["llm_summary"] = _llm_summarize_match(result)
        except Exception as e:
            log.warning("LLM summary failed: %s", e)

    return result


def _llm_summarize_match(match_result: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "You are an auditor. Explain the reconciliation outcome in simple terms.\n"
        "Focus on: how many 1:1 matches, notable composite matches (e.g., fees), "
        "and top 3 reasons some items remain unmatched.\n\n"
        f"DATA(JSON):\n{json.dumps(match_result['summary'])}\n"
        f"EXAMPLES(JSON):\n{json.dumps(match_result['matches']['one_to_one'][:5])}\n"
    )
    resp = _openai_client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content if resp and resp.choices else ""
    return {"model": _OPENAI_MODEL, "text": text}

# ==============================================================================
# Excel Export
# ==============================================================================

def to_excel_bytes(result: Dict[str, Any]) -> bytes:
    """
    Build an Excel workbook with multiple sheets:
      Summary, Matches_1to1 (side-by-side), Composite_LedgerFromLines,
      Composite_LineFromLedgers, Recommendations, Unmatched_Statement,
      Unmatched_Ledger, Normalized_Statement, Normalized_Ledger
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        # 1) Summary
        pd.DataFrame([result.get("summary", {})]).to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]
        ws.freeze_panes(1, 0)
        ws.set_column(0, 20, 20)

        # helpers
        stmt_all = result.get("normalized_full", {}).get("statement", [])
        ledg_all = result.get("normalized_full", {}).get("ledger", [])
        stmt_df = pd.DataFrame(stmt_all)
        ledg_df = pd.DataFrame(ledg_all)

        def _safe_get(arr, idx, default=None):
            try:
                return arr[idx]
            except Exception:
                return default

        # 2) 1:1 matches (side-by-side)
        rows = []
        for m in result.get("matches", {}).get("one_to_one", []):
            li, lj = m["line_index"], m["ledger_index"]
            lrow = _safe_get(stmt_all, li, {})
            jrow = _safe_get(ledg_all, lj, {})
            r = m.get("rationale", {})
            rows.append({
                "Line.RowID": lrow.get("row_id"),
                "Line.Date": lrow.get("date"),
                "Line.Intent": lrow.get("intent"),
                "Line.Desc": lrow.get("raw_desc"),
                "Line.Amount": lrow.get("signed_amount"),
                "Ledger.RowID": jrow.get("row_id"),
                "Ledger.Date": jrow.get("date"),
                "Ledger.Side": jrow.get("side"),
                "Ledger.Intent": jrow.get("intent"),
                "Ledger.Desc": jrow.get("raw_desc"),
                "Ledger.Amount": jrow.get("amount"),
                "Score": m.get("score"),
                "Amount.Diff(abs)": r.get("diff"),
                "Token.Bonus": r.get("text_token_bonus"),
                "SemSim": r.get("semantic_sim"),
                "Dir.Bonus": r.get("dir_bonus"),
                "Intent.Bonus": r.get("intent_bonus"),
                "Date.Bonus": r.get("date_bonus"),
            })
        df_m1 = pd.DataFrame(rows)
        if not df_m1.empty:
            df_m1.to_excel(writer, sheet_name="Matches_1to1", index=False)
            ws = writer.sheets["Matches_1to1"]
            ws.freeze_panes(1, 0)
            ws.set_column(0, len(df_m1.columns)-1, 24)
            # Conditional format on Score and Amount.Diff
            if "Score" in df_m1.columns:
                col = df_m1.columns.get_loc("Score")
                ws.conditional_format(1, col, len(df_m1)+1, col, {"type": "3_color_scale"})
            if "Amount.Diff(abs)" in df_m1.columns:
                col = df_m1.columns.get_loc("Amount.Diff(abs)")
                ws.conditional_format(1, col, len(df_m1)+1, col, {"type": "3_color_scale"})
        else:
            # still create the sheet for consistency
            pd.DataFrame(columns=[
                "Line.RowID","Line.Date","Line.Intent","Line.Desc","Line.Amount",
                "Ledger.RowID","Ledger.Date","Ledger.Side","Ledger.Intent","Ledger.Desc","Ledger.Amount",
                "Score","Amount.Diff(abs)","Token.Bonus","SemSim","Dir.Bonus","Intent.Bonus","Date.Bonus"
            ]).to_excel(writer, sheet_name="Matches_1to1", index=False)
            ws = writer.sheets["Matches_1to1"]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 17, 20)

        # 3) Composite: ledger = line1 + line2
        rows_c1 = []
        for m in result.get("matches", {}).get("composite_ledger_from_lines", []):
            j = m.get("ledger_index")
            lidxs = m.get("line_indices", [])
            jrow = _safe_get(ledg_all, j, {})
            lrow1 = _safe_get(stmt_all, _safe_get(lidxs, 0), {})
            lrow2 = _safe_get(stmt_all, _safe_get(lidxs, 1), {})
            rows_c1.append({
                "Ledger.RowID": jrow.get("row_id"),
                "Ledger.Date": jrow.get("date"),
                "Ledger.Side": jrow.get("side"),
                "Ledger.Intent": jrow.get("intent"),
                "Ledger.Desc": jrow.get("raw_desc"),
                "Ledger.Amount": jrow.get("amount"),
                "Line1.RowID": lrow1.get("row_id"),
                "Line1.Date": lrow1.get("date"),
                "Line1.Intent": lrow1.get("intent"),
                "Line1.Desc": lrow1.get("raw_desc"),
                "Line1.Amount": lrow1.get("signed_amount"),
                "Line2.RowID": lrow2.get("row_id"),
                "Line2.Date": lrow2.get("date"),
                "Line2.Intent": lrow2.get("intent"),
                "Line2.Desc": lrow2.get("raw_desc"),
                "Line2.Amount": lrow2.get("signed_amount"),
                "Score": m.get("score"),
                "Diff(abs)": (abs(abs(jrow.get("amount", 0)) - abs((lrow1.get("signed_amount") or 0) + (lrow2.get("signed_amount") or 0))))
            })
        df_c1 = pd.DataFrame(rows_c1)
        df_c1.to_excel(writer, sheet_name="Composite_LedgerFromLines", index=False)
        ws = writer.sheets["Composite_LedgerFromLines"]
        ws.freeze_panes(1, 0)
        ws.set_column(0, max(0, len(df_c1.columns)-1), 24)

        # 4) Composite: line = ledger1 + ledger2
        rows_c2 = []
        for m in result.get("matches", {}).get("composite_line_from_ledger", []):
            i = m.get("line_index")
            jidxs = m.get("ledger_indices", [])
            lrow = _safe_get(stmt_all, i, {})
            jrow1 = _safe_get(ledg_all, _safe_get(jidxs, 0), {})
            jrow2 = _safe_get(ledg_all, _safe_get(jidxs, 1), {})
            rows_c2.append({
                "Line.RowID": lrow.get("row_id"),
                "Line.Date": lrow.get("date"),
                "Line.Intent": lrow.get("intent"),
                "Line.Desc": lrow.get("raw_desc"),
                "Line.Amount": lrow.get("signed_amount"),
                "Ledger1.RowID": jrow1.get("row_id"),
                "Ledger1.Date": jrow1.get("date"),
                "Ledger1.Side": jrow1.get("side"),
                "Ledger1.Intent": jrow1.get("intent"),
                "Ledger1.Desc": jrow1.get("raw_desc"),
                "Ledger1.Amount": jrow1.get("amount"),
                "Ledger2.RowID": jrow2.get("row_id"),
                "Ledger2.Date": jrow2.get("date"),
                "Ledger2.Side": jrow2.get("side"),
                "Ledger2.Intent": jrow2.get("intent"),
                "Ledger2.Desc": jrow2.get("raw_desc"),
                "Ledger2.Amount": jrow2.get("amount"),
                "Score": m.get("score"),
                "Diff(abs)": (abs(abs(lrow.get("signed_amount", 0)) - abs((jrow1.get("amount") or 0) + (jrow2.get("amount") or 0))))
            })
        df_c2 = pd.DataFrame(rows_c2)
        df_c2.to_excel(writer, sheet_name="Composite_LineFromLedgers", index=False)
        ws = writer.sheets["Composite_LineFromLedgers"]
        ws.freeze_panes(1, 0)
        ws.set_column(0, max(0, len(df_c2.columns)-1), 24)

        # 5) Recommendations (flatten each candidate as a row)
        rec_rows = []
        for rec in result.get("recommendations", []):
            li = rec.get("line_index")
            lrow = _safe_get(stmt_all, li, {})
            for cand in rec.get("candidates", []):
                lj = cand.get("ledger_index")
                jrow = _safe_get(ledg_all, lj, {})
                reason = cand.get("reasoning", {}) or {}
                rec_rows.append({
                    "Line.RowID": lrow.get("row_id"),
                    "Line.Date": lrow.get("date"),
                    "Line.Intent": lrow.get("intent"),
                    "Line.Desc": lrow.get("raw_desc"),
                    "Line.Amount": lrow.get("signed_amount"),
                    "Cand.Ledger.RowID": jrow.get("row_id"),
                    "Cand.Ledger.Date": jrow.get("date"),
                    "Cand.Ledger.Side": jrow.get("side"),
                    "Cand.Ledger.Intent": jrow.get("intent"),
                    "Cand.Ledger.Desc": jrow.get("raw_desc"),
                    "Cand.Ledger.Amount": jrow.get("amount"),
                    "Reco.Score": cand.get("score"),
                    "Amt.Diff(abs)": reason.get("amount_diff"),
                    "Token.Bonus": reason.get("token_overlap_bonus"),
                    "SemSim": reason.get("semantic_sim"),
                    "Dir.Bonus": reason.get("direction_hint"),
                    "Intent.Bonus": reason.get("intent_bonus"),
                    "Date.Bonus": reason.get("date_bonus"),
                    "Date.Gap(days)": reason.get("date_gap_days"),
                    "Notes": "; ".join(cand.get("notes", [])) if cand.get("notes") else None
                })
        df_rec = pd.DataFrame(rec_rows)
        df_rec.to_excel(writer, sheet_name="Recommendations", index=False)
        ws = writer.sheets["Recommendations"]
        ws.freeze_panes(1, 0)
        ws.set_column(0, max(0, len(df_rec.columns)-1), 24)
        if not df_rec.empty and "Reco.Score" in df_rec.columns:
            col = df_rec.columns.get_loc("Reco.Score")
            ws.conditional_format(1, col, len(df_rec)+1, col, {"type": "3_color_scale"})

        # 6) Unmatched (Statement / Ledger)
        df_u_stmt = pd.DataFrame(result.get("unmatched", {}).get("statement_lines", []))
        df_u_ledg = pd.DataFrame(result.get("unmatched", {}).get("ledger_items", []))
        df_u_stmt.to_excel(writer, sheet_name="Unmatched_Statement", index=False)
        df_u_ledg.to_excel(writer, sheet_name="Unmatched_Ledger", index=False)
        ws = writer.sheets["Unmatched_Statement"]; ws.freeze_panes(1, 0); ws.set_column(0, max(0, len(df_u_stmt.columns)-1), 24)
        ws = writer.sheets["Unmatched_Ledger"]; ws.freeze_panes(1, 0); ws.set_column(0, max(0, len(df_u_ledg.columns)-1), 24)

        # 7) Normalized full tables (with safe fallbacks if not present)
        def _collect_stmt_full() -> pd.DataFrame:
            if stmt_df is not None and not stmt_df.empty:
                return stmt_df
            # fallback: build from matches + unmatched
            rows = []
            # from matches
            for m in result.get("matches", {}).get("one_to_one", []):
                rows.append(_safe_get(stmt_all, m.get("line_index"), {}))
            for m in result.get("matches", {}).get("composite_ledger_from_lines", []):
                for idx in m.get("line_indices", []):
                    rows.append(_safe_get(stmt_all, idx, {}))
            for r in result.get("recommendations", []):
                rows.append(_safe_get(stmt_all, r.get("line_index"), {}))
            # unmatched
            rows.extend(result.get("unmatched", {}).get("statement_lines", []))
            # dedupe by row_id if present
            df = pd.DataFrame([r for r in rows if r]) if rows else pd.DataFrame()
            if "row_id" in df.columns:
                df = df.drop_duplicates(subset=["row_id"])
            return df

        def _collect_ledg_full() -> pd.DataFrame:
            if ledg_df is not None and not ledg_df.empty:
                return ledg_df
            rows = []
            for m in result.get("matches", {}).get("one_to_one", []):
                rows.append(_safe_get(ledg_all, m.get("ledger_index"), {}))
            for m in result.get("matches", {}).get("composite_line_from_ledger", []):
                for idx in m.get("ledger_indices", []):
                    rows.append(_safe_get(ledg_all, idx, {}))
            for rec in result.get("recommendations", []):
                for cand in rec.get("candidates", []):
                    rows.append(_safe_get(ledg_all, cand.get("ledger_index"), {}))
            rows.extend(result.get("unmatched", {}).get("ledger_items", []))
            df = pd.DataFrame([r for r in rows if r]) if rows else pd.DataFrame()
            if "row_id" in df.columns:
                df = df.drop_duplicates(subset=["row_id"])
            return df

        df_stmt_full = _collect_stmt_full()
        df_ledg_full = _collect_ledg_full()

        df_stmt_full.to_excel(writer, sheet_name="Normalized_Statement", index=False)
        df_ledg_full.to_excel(writer, sheet_name="Normalized_Ledger", index=False)
        writer.sheets["Normalized_Statement"].freeze_panes(1, 0)
        writer.sheets["Normalized_Ledger"].freeze_panes(1, 0)
        writer.sheets["Normalized_Statement"].set_column(0, max(0, len(df_stmt_full.columns)-1), 24)
        writer.sheets["Normalized_Ledger"].set_column(0, max(0, len(df_ledg_full.columns)-1), 24)

        # 8) AutoFilter on all data tabs for convenience
        for name in ["Matches_1to1","Composite_LedgerFromLines","Composite_LineFromLedgers",
                     "Recommendations","Unmatched_Statement","Unmatched_Ledger",
                     "Normalized_Statement","Normalized_Ledger"]:
            ws = writer.sheets.get(name)
            if ws:
                # determine last row/col if we can
                try:
                    df = {
                        "Matches_1to1": df_m1,
                        "Composite_LedgerFromLines": df_c1,
                        "Composite_LineFromLedgers": df_c2,
                        "Recommendations": df_rec,
                        "Unmatched_Statement": df_u_stmt,
                        "Unmatched_Ledger": df_u_ledg,
                        "Normalized_Statement": df_stmt_full,
                        "Normalized_Ledger": df_ledg_full
                    }[name]
                    nrows = len(df.index) + 1
                    ncols = len(df.columns) - 1
                    if nrows > 1 and ncols >= 0:
                        ws.autofilter(0, 0, nrows-1, ncols)
                except Exception:
                    pass

    return buf.getvalue()

