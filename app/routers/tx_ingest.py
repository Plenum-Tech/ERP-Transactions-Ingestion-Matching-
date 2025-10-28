
# app/routers/tx_ingest.py
from __future__ import annotations

import os, re, json, hashlib, unicodedata
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, date
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

from app.db import db_conn
from app.di_extractors import bank_statement as di_bank
from app.di_extractors import receipt as di_receipt
from app.utils.logger import get_logger, log_duration

# ==== Router & config =========================================================
router = APIRouter(prefix="/tx", tags=["transactions-ingest"])
log = get_logger(__name__)

DEFAULT_CCY = os.getenv("DEFAULT_CURRENCY", "AED")
RECO_STATEMENT_TYPE = os.getenv("RECO_STATEMENT_TYPE", "Bank Reconciliation")
AI_TX_BATCH_SIZE = int(os.getenv("AI_TX_BATCH_SIZE", "40"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ==== OpenAI client (official SDK v1) ========================================
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
except Exception as e:
    _openai_client = None
    _openai_client_error = e

# ==== Minimal helpers =========================================================
def _as_date(v: Any) -> Optional[date]:
    if v is None: return None
    if isinstance(v, date) and not isinstance(v, datetime): return v
    if isinstance(v, datetime): return v.date()
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except Exception:
            try:
                return date.fromisoformat(s[:10])
            except Exception:
                return None
    return None

def _num_soft(v: Any) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v)
    neg = "(" in s and ")" in s
    s2 = re.sub(r"[^0-9.\-]", "", s)
    if s2 in ("", ".", "-", "-."): return None
    try:
        n = float(s2)
        return -abs(n) if neg and n > 0 else n
    except Exception:
        return None

def _norm_merchant(s: Optional[str]) -> Optional[str]:
    if not s: return None
    x = s.upper().strip()
    x = re.sub(r"[^A-Z0-9 &.\-]", " ", x)
    x = re.sub(r"\b(LLC|LTD|CO|INC|PJSC|PVT)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x or None

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _normalize_bank_name(s: Optional[str]) -> Optional[str]:
    if not s: return None
    raw = _strip_accents(s).upper()
    if "CITI" in raw or "CITIBANK" in raw:
        return "Citibank"
    if "FIRST ABU DHABI" in raw or " FAB " in f" {raw} ":
        return "First Abu Dhabi Bank"
    if "EMIRATES NBD" in raw:
        return "Emirates NBD"
    if "ADCB" in raw or "ABU DHABI COMMERCIAL" in raw:
        return "Abu Dhabi Commercial Bank"
    if "MASHREQ" in raw:
        return "Mashreq Bank"
    if "DUBAI ISLAMIC" in raw or "DIB" in raw:
        return "Dubai Islamic Bank"
    if "RAKBANK" in raw or "RAK BANK" in raw:
        return "RAKBANK"
    if "STANDARD CHARTERED" in raw:
        return "Standard Chartered"
    if "HSBC" in raw:
        return "HSBC"
    return " ".join(w.capitalize() for w in raw.split())

def _fp(parts: List[str]) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _infer_currency_from_result(di_result: Any) -> Optional[str]:
    hints = {"£": "GBP", "GBP": "GBP", "€": "EUR", "EUR": "EUR", "$": "USD", "USD": "USD",
             "AED": "AED", "SAR": "SAR", "QAR": "QAR", "OMR": "OMR", "BHD": "BHD"}
    try:
        for p in getattr(di_result, "pages", []) or []:
            for ln in getattr(p, "lines", []) or []:
                txt = (getattr(ln, "content", "") or "").upper()
                for token, code in hints.items():
                    if token in txt: return code
        for tbl in getattr(di_result, "tables", []) or []:
            for c in getattr(tbl, "cells", []) or []:
                txt = (getattr(c, "content", "") or "").upper()
                for token, code in hints.items():
                    if token in txt: return code
    except Exception:
        pass
    return None

def _getv(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            obj = d[k]
            if isinstance(obj, dict) and "value" in obj: return obj["value"]
            if obj is not None: return obj
    return None

def _gather_text(di_result: Any) -> str:
    try:
        return "\n".join((ln.content or "") for p in (getattr(di_result, "pages", []) or []) for ln in (p.lines or []))
    except Exception:
        return ""

# ==== Bank/statement fallbacks ===============================================
_BANK_PATTERNS = [
    (re.compile(r"\bFIRST\s+ABU\s+DHABI\s+BANK\b", re.I), "First Abu Dhabi Bank"),
    (re.compile(r"\bEMIRATES\s+NBD\b", re.I),            "Emirates NBD"),
    (re.compile(r"\bABU\s+DHABI\s+COMMERCIAL\s+BANK\b|\bADCB\b", re.I), "Abu Dhabi Commercial Bank"),
    (re.compile(r"\bMASHREQ\b", re.I),                   "Mashreq Bank"),
    (re.compile(r"\bDUBAI\s+ISLAMIC\s+BANK\b|\bDIB\b", re.I), "Dubai Islamic Bank"),
    (re.compile(r"\bRAK\s*BANK\b|\bRAKBANK\b", re.I),    "RAKBANK"),
    (re.compile(r"\bSTANDARD\s+CHARTERED\b", re.I),      "Standard Chartered"),
    (re.compile(r"\bHSBC\b", re.I),                      "HSBC"),
    (re.compile(r"\bCITI(?:BANK)?\b", re.I),             "Citibank"),
]

def _infer_bank_name(fields: Dict[str, Any], di_result: Any, filename_hint: Optional[str]) -> Optional[str]:
    bank_name = (fields.get("BankName") or {}).get("value")
    if bank_name: return bank_name
    text = _gather_text(di_result)
    for rx, name in _BANK_PATTERNS:
        if rx.search(text): return name
    if filename_hint:
        up = filename_hint.upper()
        for rx, name in _BANK_PATTERNS:
            if rx.search(up): return name
        if " BANKFAB " in f" {up} " or " FAB " in f" {up} ":
            return "First Abu Dhabi Bank"
    return None

def _infer_statement_date(fields: Dict[str, Any], di_result: Any, accounts: List[Dict[str, Any]]) -> Optional[date]:
    d = _as_date((fields.get("StatementEndDate") or {}).get("value"))
    if d: return d
    text = _gather_text(di_result)

    # "Statement generated on: 11-Nov-2024"
    m = re.search(r"Statement\s+generated\s+on:\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", text, re.I)
    if m:
        d = _as_date(m.group(1))
        if d: return d

    # "Account statement: ... to 11-Nov-2024"
    m2 = re.search(r"Account\s+statement:\s*.*?\bto\b\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", text, re.I | re.S)
    if m2:
        d = _as_date(m2.group(1))
        if d: return d

    # "FOR THE PERIOD OF 11-10-2024 TO 11-11-2024"
    m3 = re.search(r"FOR\s+THE\s+PERIOD\s+OF\s+[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}\s+TO\s+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})", text, re.I)
    if m3:
        d = _as_date(m3.group(1))
        if d: return d

    # Fallback: latest tx date
    max_tx: Optional[date] = None
    for acc in accounts or []:
        for t in (acc.get("Transactions") or []):
            cand = _as_date(_getv(t, "TransactionDate", "PostingDate", "PostDate", "Date", "BookingDate", "ValueDate"))
            if cand and (max_tx is None or cand > max_tx):
                max_tx = cand
    return max_tx

# ==== Account number selection (robust, non-IBAN, line-aware) ================
# Label must be on the same line; value may be same line or the next line.
_ACCT_LABEL_LINE_RX = re.compile(
    r"\b(?:ACCOUNT\s*(?:NO\.?|NUMBER)|A/C\s*NO\.?|ACC(?:OUNT)?\s*#?)\b", re.I
)
# A plausible account value token on the value line (digits only)  << FIXED
_ACCT_LINE_VALUE_RX = re.compile(r"\b([0-9][0-9\-\s]{6,30})\b")

_IBAN_FULL_RX = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")
_DIGIT_BLOCK_RX = re.compile(r"\d{8,20}")

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _looks_like_iban(s: Optional[str]) -> bool:
    if not s:
        return False
    raw = re.sub(r"\s+", "", s.strip())
    return bool(_IBAN_FULL_RX.match(raw))

def _find_account_no_in_text(di_result: Any) -> Optional[str]:
    """
    Walk the statement line-by-line and only extract digits from:
      - the same line as 'Account No'  OR
      - the very next line
    Never scan globally (prevents IBAN bleed-through).
    """
    lines: List[str] = []
    try:
        for p in getattr(di_result, "pages", []) or []:
            for ln in (p.lines or []):
                lines.append((ln.content or "").strip())
    except Exception:
        return None

    for i, line in enumerate(lines):
        if _ACCT_LABEL_LINE_RX.search(line):
            # Same-line value
            m_same = _ACCT_LINE_VALUE_RX.search(line)
            if m_same:
                d = _digits_only(m_same.group(1))
                if 8 <= len(d) <= 20:
                    return d
            # Next-line value
            if i + 1 < len(lines):
                m_next = _ACCT_LINE_VALUE_RX.search(lines[i + 1])
                if m_next:
                    d = _digits_only(m_next.group(1))
                    if 8 <= len(d) <= 20:
                        return d
    return None

def _acct_from_iban(iban: Optional[str]) -> Optional[str]:
    """
    If we only have IBAN, extract the trailing account core (digits).
    Prefer 10–16 digits; else choose the longest digit block.
    Never return the IBAN itself.
    """
    if not iban:
        return None
    raw = re.sub(r"\s+", "", str(iban).strip())
    if not _looks_like_iban(raw):
        return None
    blocks = list(_DIGIT_BLOCK_RX.finditer(raw))
    if not blocks:
        return None
    pref = [b.group(0) for b in blocks if 10 <= len(b.group(0)) <= 16]
    return (pref[-1] if pref else max(blocks, key=lambda mm: len(mm.group(0))).group(0))

def _pick_account_no(acc: Dict[str, Any], di_result: Any) -> Optional[str]:
    """
    Priority:
      1) Structured DI 'AccountNumber' (digits-only, not IBAN)
      2) Labelled text line ('Account No.' line or the following line)
      3) IBAN tail (digits-only core)
    """
    # 1) Structured DI value
    acct_raw = _getv(acc, "AccountNumber", "Account No", "AccountNo")
    if acct_raw and not _looks_like_iban(str(acct_raw)):
        d = _digits_only(str(acct_raw))
        if 8 <= len(d) <= 20:
            return d

    # 2) Label line(s)
    from_label = _find_account_no_in_text(di_result)
    if from_label:
        return from_label

    # 3) IBAN tail
    iban_raw = _getv(acc, "IBAN", "Iban", "IbanNumber")
    from_iban = _acct_from_iban(iban_raw)
    if from_iban:
        return from_iban

    return None

# ==== Amount + continuation utilities ========================================
def _signed_amount_from_tx(tx: Dict[str, Any]) -> Optional[float]:
    raw_amount = _getv(tx, "Amount", "TransactionAmount")
    amt = _num_soft(raw_amount)
    if amt is not None:
        tval = str(_getv(tx, "Type", "CreditDebit", "TransactionType") or "").strip().lower()
        if tval in ("debit", "dr", "withdrawal", "payment", "out", "card debit", "debit card transaction"):
            return -abs(amt)
        if tval in ("credit", "cr", "deposit", "in", "salary transfer", "salary", "paid in"):
            return abs(amt)
        s = str(raw_amount or "")
        if s.strip().startswith("-") or ("(" in s and ")" in s):
            return -abs(amt)
        return amt

    dep = _num_soft(_getv(tx, "DepositAmount", "Credit", "PaidIn", "In"))
    wdr = _num_soft(_getv(tx, "WithdrawalAmount", "Debit", "PaidOut", "Out"))
    if dep is not None or wdr is not None:
        dep = dep or 0.0
        wdr = wdr or 0.0
        return dep - abs(wdr)

    credit = _num_soft(_getv(tx, "Credit"))
    debit = _num_soft(_getv(tx, "Debit"))
    if credit is not None: return abs(credit)
    if debit is not None: return -abs(debit)
    return None

def _desc_of(tx: Dict[str, Any]) -> Optional[str]:
    return _getv(tx, "Description")

def _candidate_dates(tx: Dict[str, Any]) -> Set[date]:
    ks = ("TransactionDate", "PostingDate", "PostDate", "Date", "BookingDate", "ValueDate", "ValDate", "Value Date")
    out: Set[date] = set()
    for k in ks:
        d = _as_date(_getv(tx, k))
        if d: out.add(d)
    return out

def _is_continuation(prev_anchor: Dict[str, Any], curr: Dict[str, Any]) -> bool:
    if not prev_anchor or not curr: return False
    if not ((_desc_of(curr) or "").strip()): return False
    if _signed_amount_from_tx(curr) is not None:
        return False
    curr_dates = _candidate_dates(curr)
    if not curr_dates:
        return True
    prev_dates = _candidate_dates(prev_anchor)
    return bool(prev_dates and curr_dates and curr_dates.issubset(prev_dates))

def _stitch_wrapped_lines(txs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    stitched: List[Dict[str, Any]] = []
    appended = 0
    for tx in txs or []:
        if stitched and _is_continuation(stitched[-1], tx):
            d_next = (_desc_of(tx) or "").strip()
            if d_next:
                prev = stitched[-1]
                prev_desc = (_getv(prev, "Description") or "")
                prev.setdefault("Description", (prev_desc + " " + d_next).strip())
                appended += 1
            continue
        stitched.append(tx)
    return stitched, appended

# ---- description composer ----------------------------------------------------
_GENERIC_KINDS = {
    "debit card transaction", "atm cash withdrawal", "salary transfer",
    "ipp transfer", "ipp charges", "cash chq withdrawal",
    "guarantee issue commission", "vat charges", "monthly charge",
}
def _compose_description(kind: Optional[str], tx_text: Optional[str]) -> Optional[str]:
    if not kind and not tx_text: return None
    if kind and tx_text:
        if kind.strip().lower() in _GENERIC_KINDS and tx_text.strip():
            if tx_text.lower() not in kind.lower():
                return f"{kind.strip()} - {tx_text.strip()}"
    return kind or tx_text

# ===== Document No. extraction (deterministic) ================================
_MONTH_TRIGGERS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_LABEL_RX = re.compile(r"\b(?:ref(?:erence)?|rrn|approval|auth(?:[.\s]*code)?|txn(?:\s*id)?|transaction\s*id|arn)\b[:\-\s]*", re.I)
_CHECK_RX = re.compile(r"\b(?:cheque|chq)\s*no\.?\s*[:\-]?\s*(\d{4,12})\b", re.I)

_STRONG_LABEL_RX = re.compile(
    r"\b(?:ref(?:\.|erence)?(?:\s*no\.?)?|rrn|txn(?:\s*id)?|transaction\s*id|auth(?:\s*code)?|approval(?:\s*code)?)\b[:\-–—]?\s*",
    re.I,
)
_NUM_TINY_RX = re.compile(r"\b\d{3,5}\b")
_TIME_LIKE_RX = re.compile(r"\b([01]\d|2[0-3])[0-5]\d\b")
_BRANCH_CONTEXT_RX = re.compile(r"\b(?:BRANCH|MG\s*ROAD|NO\s*\d{1,2}|KAIN|LAYOUT|BANGALORE|VIVEKANANDA|OFFBANGALORE)\b", re.I)

_NUM_PRIMARY_RX = re.compile(r"\b\d{9,15}\b")
_NUM_SECONDARY_RX = re.compile(r"\b\d{6,8}\b")
_ALNUM_RX = re.compile(r"\b[A-Z0-9]{8,20}\b")

def _looks_like_date_token(tok: str) -> bool:
    u = tok.upper()
    if any(m in u for m in _MONTH_TRIGGERS):
        return True
    datey = [
        r"^\d{1,2}[-/][A-Z]{3}[-/]\d{2,4}$",
        r"^\d{1,2}[A-Z]{3}\d{2,4}$",
        r"^\d{4}[-/]\d{2}[-/]\d{2}$",
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
        r"^\d{1,2}[A-Z]{3}\d{4}$",
        r"^\d{1,2}[A-Z]{3}\d{4,}$",
    ]
    return any(re.match(p, u) for p in datey)

def _bad_token(tok: str) -> bool:
    if not tok: return True
    t = tok.strip()
    if any(ch in t for ch in ("/", ":", "-", " ", ".")): return True
    if "X" in t.upper(): return True
    if _looks_like_date_token(t): return True
    return False

def _nearest_label_bonus(desc: str, span_start: int, window1: int = 40, window2: int = 80) -> int:
    bonus = 0
    for m in _LABEL_RX.finditer(desc):
        d = abs(span_start - m.end())
        if d <= window1: bonus = max(bonus, 60)
        elif d <= window2: bonus = max(bonus, 30)
    return bonus

def extract_check_no_from_text(desc: str) -> Optional[str]:
    m = _CHECK_RX.search(desc or "")
    if m:
        return m.group(1)
    return None

def _has_long_candidate_near(text: str, pivot_pos: int, radius: int = 25) -> bool:
    left = max(0, pivot_pos - radius)
    right = min(len(text), pivot_pos + radius)
    window = text[left:right]
    return bool(_NUM_PRIMARY_RX.search(window) or _NUM_SECONDARY_RX.search(window) or _ALNUM_RX.search(window))

def _in_branchy_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 25):min(len(text), end + 25)]
    return bool(_BRANCH_CONTEXT_RX.search(window))

def _nearest_strong_label_distance(text: str, span_start: int) -> int:
    best = 10 ** 9
    for m in _STRONG_LABEL_RX.finditer(text):
        d = abs(span_start - m.end())
        if d < best: best = d
    return best

def extract_document_no_from_text(desc: Optional[str]) -> Optional[str]:
    if not desc:
        return None
    text = desc.strip()
    if not text:
        return None

    candidates: List[Tuple[int, str]] = []

    for m in _NUM_PRIMARY_RX.finditer(text):
        tok = m.group(0)
        if _bad_token(tok): continue
        score = 80 + _nearest_label_bonus(text, m.start())
        candidates.append((score, tok))

    for m in _NUM_SECONDARY_RX.finditer(text):
        tok = m.group(0)
        if _bad_token(tok): continue
        bonus = _nearest_label_bonus(text, m.start())
        if bonus >= 60:
            candidates.append((40 + bonus, tok))

    for m in _ALNUM_RX.finditer(text):
        tok = m.group(0)
        if not tok.isalnum() or tok != tok.upper() or _bad_token(tok): continue
        if not any(ch.isdigit() for ch in tok): continue
        bonus = _nearest_label_bonus(text, m.start())
        if bonus >= 60:
            candidates.append((30 + bonus, tok))

    for m in _NUM_TINY_RX.finditer(text):
        tok = m.group(0)
        if _bad_token(tok) or _TIME_LIKE_RX.match(tok): continue
        dist = _nearest_strong_label_distance(text, m.start())
        if dist > 8: continue
        if _has_long_candidate_near(text, m.start(), radius=25): continue
        if _in_branchy_context(text, m.start(), m.end()): continue
        score = 20 + max(0, 60 - dist * 5)
        candidates.append((score, tok))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return candidates[0][1]

# ==== AI-assisted helpers =====================================================
_LLM_REF_SYSTEM = """You extract the single best banking reference/transaction/document number from a bank transaction DESCRIPTION.

Output STRICT JSON: {"document_no": string|null}

Selection rules (must-follow):
- Prefer 10–15 digit numerics, then 6–8 digit numerics, then 8–20 char uppercase alphanumerics.
- You MAY return a 3–5 digit numeric ONLY if it appears immediately after a strong label (e.g., "Ref", "Ref No", "RRN", "Txn ID", "Auth Code", "Approval Code") and there is no longer candidate (>= 6 digits or alphanumerics) near the same label.
- HARD EXCLUDE any token that is a date/time or contains month text/format (JAN..DEC, DD-MMM-YYYY, DDMMMYYYY, YYYY-MM-DD, DD/MM/YYYY) or masked card numbers.
- EXCLUDE amounts/currencies and known branch/location codes (e.g., codes next to words like BRANCH, MG ROAD).
- If multiple candidates exist, choose the one nearest to a strong label.
- If nothing suitable, return {"document_no": null}.
"""

_LLM_REF_USER_TEMPLATE = """Description:
{desc}

Context:
- Bank: {bank}
- Transaction date: {tx_date}
- Value date: {val_date}
- Statement end date: {stmt_date}
- Amount: {amount}

DO NOT return any of the dates above or tokens derived from them.
Return only JSON with 'document_no'.
"""

def _openai_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if _openai_client is None:
        raise RuntimeError(f"OpenAI client not available: {_openai_client_error}")
    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt or "{}")
    except Exception as e:
        raise RuntimeError(f"LLM returned non-JSON: {e}: {txt[:400]}")

def _ai_extract_document_no(
    desc: str,
    bank: Optional[str],
    tx_date: Optional[date],
    amount: Optional[float],
    val_date: Optional[date] = None,
    stmt_date: Optional[date] = None,
) -> Optional[str]:
    if not desc or _openai_client is None:
        return None
    user = _LLM_REF_USER_TEMPLATE.format(
        desc=desc,
        bank=bank or "Unknown",
        tx_date=(tx_date.isoformat() if isinstance(tx_date, date) else "Unknown"),
        val_date=(val_date.isoformat() if isinstance(val_date, date) else "Unknown"),
        stmt_date=(stmt_date.isoformat() if isinstance(stmt_date, date) else "Unknown"),
        amount=(f"{amount:.2f}" if isinstance(amount, (int, float)) else "Unknown"),
    )
    try:
        out = _openai_json([
            {"role": "system", "content": _LLM_REF_SYSTEM},
            {"role": "user", "content": user},
        ])
        doc_no = out.get("document_no")
        if isinstance(doc_no, str) and doc_no.strip():
            return doc_no.strip()
        return None
    except Exception:
        return None

# ==== LLM mapping for full rows + deterministic fallback ======================
_LLM_SYSTEM = """You are a finance extraction engine that maps raw bank statement transaction objects
(from OCR/Document-Intelligence) into a strict ERP row schema. Return STRICT JSON only. Never hallucinate; use null when a field is not present.

Rules:
- Dates: 'YYYY-MM-DD' or null.
- Amount sign: credits (paid in / deposit / credit / salary) are POSITIVE; debits are NEGATIVE.
- 'description': transaction kind (e.g., 'Debit Card Transaction', 'ATM Cash Withdrawal', 'Salary Transfer'), concise.
- 'transaction_text': merchant/counterparty text (e.g., 'CARREFOUR RAK', 'FAB', 'KEYSTONE GROUP LLC FZ').
- 'check_no': if cheque is mentioned, extract cheque number (4+ alphanum) else null.
- 'related_party_name': counterparty or merchant when clearly present.
- 'additional_transaction_info' and 'reference': approval/ref/rrn/auth-like tokens when present.
- 'document_no': the single best reference/transaction/document number from the raw description.
  STRICTLY DO NOT return dates/times or tokens with month names (JAN..DEC) or formatted dates (DD-MMM-YYYY etc.).
  Prefer 10–15 digit numerics or uppercase alphanumerics near 'Ref/Reference/RRN/Auth/Approval/Txn'.
- Preserve input order; one output row per input row. Do not omit rows.
"""

_LLM_USER_TEMPLATE = """Context:
- Statement End Date (if known): {statement_end_date}
- Bank Name (if known): {bank_name}
- Default Currency: {default_currency}

Map the following transaction objects to ERP rows. Input array length = {n}.
Return EXACT JSON with top-level key "rows" (array length {n}). Each element MUST have exactly these keys:

[
  "transaction_date",
  "value_date",
  "description",
  "transaction_text",
  "statement_amount",
  "check_no",
  "related_party_name",
  "additional_transaction_info",
  "reference",
  "document_no"
]

Input transactions JSON:
{tx_json}
"""

def _deterministic_row_from_tx(
    t: Dict[str, Any],
    stmt_date: Optional[date],
    bank_name: Optional[str]
) -> Dict[str, Any]:
    desc = _getv(t, "Description")
    txn_date = _as_date(_getv(t, "TransactionDate", "PostingDate", "PostDate", "Date", "BookingDate", "ValueDate"))
    val_date = _as_date(_getv(t, "ValueDate", "ValDate", "Value Date"))
    amt = _signed_amount_from_tx(t)
    check_no = extract_check_no_from_text(desc or "")
    # use local deterministic extractor first; if none, try tiny LLM helper
    local_docno = extract_document_no_from_text(desc or "")
    if not local_docno:
        local_docno = _ai_extract_document_no(
            desc or "", bank_name, txn_date, amt, val_date, stmt_date
        )
    kind = _getv(t, "Type", "TransactionType")
    tx_text = desc
    return {
        "transaction_date": txn_date.isoformat() if txn_date else None,
        "value_date": val_date.isoformat() if val_date else None,
        "description": (kind or None),
        "transaction_text": tx_text,
        "statement_amount": amt,
        "check_no": check_no,
        "related_party_name": None,
        "additional_transaction_info": None,
        "reference": None,
        "document_no": local_docno,
    }

def _ai_map_rows(
    tx_items: List[Dict[str, Any]],
    statement_end_date: Optional[date],
    bank_name: Optional[str],
    default_currency: str,
) -> List[Dict[str, Any]]:
    # If no client, deterministic path
    if _openai_client is None or not tx_items:
        return [
            _deterministic_row_from_tx(t, statement_end_date, bank_name)
            for t in tx_items
        ]

    user = _LLM_USER_TEMPLATE.format(
        statement_end_date=statement_end_date.isoformat() if statement_end_date else None,
        bank_name=bank_name,
        default_currency=default_currency,
        n=len(tx_items),
        tx_json=json.dumps(tx_items, ensure_ascii=False, default=str),
    )

    def _coerce_rows(obj: Any, n: int) -> Optional[List[Dict[str, Any]]]:
        # Accept only list with required keys; coerce missing keys to None; truncate/pad to n
        if not isinstance(obj, dict):
            return None
        rows = obj.get("rows")
        if not isinstance(rows, list):
            return None
        # Coerce length
        if len(rows) != n:
            return None
        required_keys = {
            "transaction_date", "value_date", "description", "transaction_text",
            "statement_amount", "check_no", "related_party_name",
            "additional_transaction_info", "reference", "document_no"
        }
        out: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                return None
            coerced = {k: r.get(k) if k in r else None for k in required_keys}
            out.append(coerced)
        return out

    # First attempt
    try:
        raw = _openai_json([
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user},
        ])
        rows = _coerce_rows(raw, len(tx_items))
        if rows is None:
            raise ValueError("bad_rows_shape")
    except Exception as e1:
        log.warning("ai_map_rows.first_attempt_failed", extra={"extra": {"err": str(e1)}})
        # Retry once with an explicit fixer instruction
        try:
            fixer_user = user + "\n\nIMPORTANT: Return EXACTLY {n} rows. DO NOT drop or add rows. Keep input order.".format(n=len(tx_items))
            raw2 = _openai_json([
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": fixer_user},
            ])
            rows = _coerce_rows(raw2, len(tx_items))
            if rows is None:
                raise ValueError("bad_rows_shape_retry")
        except Exception as e2:
            log.error("ai_map_rows.retry_failed_fallback_deterministic", extra={"extra": {"err": str(e2)}})
            return [
                _deterministic_row_from_tx(t, statement_end_date, bank_name)
                for t in tx_items
            ]

    # Post-fix conversions and sanity
    fixed: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        # Dates -> date strings or None
        td = _as_date(r.get("transaction_date"))
        vd = _as_date(r.get("value_date"))
        # Amount -> float or None
        amt = _num_soft(r.get("statement_amount"))
        if amt is None:
            amt = _signed_amount_from_tx(tx_items[i])
        # Ensure doc no is valid; if not, use deterministic
        raw_desc = _getv(tx_items[i], "Description") or ""
        docno = r.get("document_no")
        if docno and _bad_token(str(docno)):
            docno = None
        if not docno:
            docno = extract_document_no_from_text(raw_desc) or _ai_extract_document_no(
                raw_desc, bank_name, _as_date(r.get("transaction_date")),
                amt, _as_date(r.get("value_date")), statement_end_date
            )

        fixed.append({
            "transaction_date": td.isoformat() if td else None,
            "value_date": vd.isoformat() if vd else None,
            "description": r.get("description"),
            "transaction_text": r.get("transaction_text"),
            "statement_amount": amt,
            "check_no": r.get("check_no") or extract_check_no_from_text(raw_desc),
            "related_party_name": r.get("related_party_name"),
            "additional_transaction_info": r.get("additional_transaction_info"),
            "reference": r.get("reference"),
            "document_no": docno,
        })
    return fixed

# ==== BANK STATEMENT INGEST ===================================================
@router.post("/ingest/bank-statement")
async def ingest_bank_statement(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    currency: Optional[str] = Form(default=None),
):
    log.info("ingest.bank_statement.start", extra={"extra": {
        "has_file": bool(file), "has_url": bool(url),
        "filename": getattr(file, "filename", None), "url": url
    }})

    if not file and not url:
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                url = body.get("url")
            except Exception:
                pass
    if file and url:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'url'.")

    source = None
    with log_duration("ingest.bank_statement.analyze"):
        try:
            if file:
                source = file.filename
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                result = di_bank.analyze_bank_stmt_bytes(
                    data, getattr(file, "content_type", None)
                )
            elif url:
                source = url
                result = di_bank.analyze_bank_stmt_url(url)
            else:
                raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")
        except Exception as e:
            log.exception("ingest.bank_statement.di_error")
            raise HTTPException(status_code=502, detail=f"Document Intelligence error: {e}")

    shaped = di_bank.bank_stmt_to_dict(result)
    fields = shaped.get("fields") or {}
    accounts = shaped.get("accounts") or []

    bank_name_raw = _infer_bank_name(fields, result, source)
    bank_name = _normalize_bank_name(bank_name_raw or "Unknown")
    statement_date = _infer_statement_date(fields, result, accounts)
    ccy = (currency or _infer_currency_from_result(result) or DEFAULT_CCY).upper()

    total_input_rows = sum(len(a.get("Transactions") or []) for a in accounts)
    log.info("ingest.bank_statement.shaped", extra={"extra": {
        "accounts": len(accounts), "model_tx_count": total_input_rows,
        "bank_name": bank_name, "statement_date": str(statement_date) if statement_date else None
    }})

    # ---- Stitch wrapped lines PER ACCOUNT & assign stable anchor line numbers
    stitched_accounts: List[Tuple[Optional[str], List[Dict[str, Any]], int, Dict[int, int]]] = []
    for acc in accounts:
        acct_no = _pick_account_no(acc, result)  # robust digits-only
        txs = acc.get("Transactions") or []
        stitched, appended = _stitch_wrapped_lines(txs)

        anchor_line_map: Dict[int, int] = {}
        anchor_no = 0
        for idx, _t in enumerate(stitched):
            anchor_no += 1
            anchor_line_map[idx] = anchor_no

        stitched_accounts.append((acct_no, stitched, appended, anchor_line_map))
        if appended or (acct_no is None):
            log.info("ingest.bank_statement.stitch", extra={"extra": {
                "account": acct_no, "continuations_merged": appended,
                "original_rows": len(txs), "stitched_rows": len(stitched)
            }})

    # --- Build rows using LLM + DI fallbacks ----------------------------------
    stage_rows: List[Dict[str, Any]] = []
    reco_rows: List[Dict[str, Any]] = []

    last_tx_date: Dict[Optional[str], Optional[date]] = {}
    last_val_date: Dict[Optional[str], Optional[date]] = {}

    batch: List[Dict[str, Any]] = []
    batch_meta: List[Dict[str, Any]] = []

    def _flush_batch():
        nonlocal batch, batch_meta, stage_rows, reco_rows, last_tx_date, last_val_date
        if not batch: return
        ai_rows = _ai_map_rows(batch, statement_date, bank_name, ccy)

        for i, ai in enumerate(ai_rows):
            acct_no = batch_meta[i]["acct_no"]
            raw_desc = (batch_meta[i]["desc"] or "").strip()
            line_no = batch_meta[i]["anchor_no"]

            # keep only clean digit account numbers 8–20 chars
            acct_sanitized = None
            if acct_no:
                _acct_digits = _digits_only(str(acct_no))
                if 8 <= len(_acct_digits) <= 20:
                    acct_sanitized = _acct_digits

            # dates: LLM -> DI -> rolling backfill
            txn_date = _as_date(ai.get("transaction_date"))
            if txn_date is None:
                t = batch_meta[i]["tx_obj"]
                txn_date = _as_date(_getv(t, "TransactionDate", "PostingDate", "PostDate", "Date", "BookingDate", "ValueDate"))
            if txn_date is None:
                txn_date = last_tx_date.get(acct_no)

            val_date = _as_date(ai.get("value_date"))
            if val_date is None:
                t = batch_meta[i]["tx_obj"]
                val_date = _as_date(_getv(t, "ValueDate", "ValDate", "Value Date"))
            if val_date is None:
                val_date = txn_date or last_val_date.get(acct_no)

            # amount
            amt = _num_soft(ai.get("statement_amount"))
            if amt is None:
                amt = _signed_amount_from_tx(batch_meta[i]["tx_obj"])
            if amt is None and (txn_date is None and not raw_desc):
                # hopelessly empty row; skip
                continue

            desc_composed = _compose_description(ai.get("description"), ai.get("transaction_text"))
            desc_final = (desc_composed or raw_desc or "").strip()
            merch_norm = _norm_merchant(desc_final)

            ref = ai.get("reference") or ai.get("additional_transaction_info")

            local_docno = extract_document_no_from_text(raw_desc) or extract_check_no_from_text(raw_desc or "")
            llm_docno = (ai.get("document_no") or None)
            if llm_docno and _bad_token(str(llm_docno)): llm_docno = None
            document_no = local_docno or llm_docno

            if raw_desc and "INTEREST EARNED" in raw_desc.upper():
                document_no = None

            check_no = ai.get("check_no") or extract_check_no_from_text(raw_desc or "")

            fp = _fp([
                acct_sanitized or "",
                statement_date.isoformat() if statement_date else "",
                line_no,
                f"{(amt or 0):.2f}",
                ccy,
                merch_norm or "",
                ref or "",
            ])

            stage_rows.append({
                "txn_date": txn_date,
                "amount": amt,
                "currency": ccy,
                "merchant_raw": raw_desc,
                "merchant_norm": merch_norm,
                "reference": ref,
                "source_file": batch_meta[i]["source"],
                "fingerprint": fp,
            })

            reco_rows.append({
                "Bank Account No.": acct_sanitized,
                "Bank Name": bank_name,
                "Statement Date": statement_date,
                "Reconciliation date": None,
                "Statement Line No.": line_no,
                "Document No.": document_no,
                "Transaction Date": txn_date,
                "Description": desc_final,
                "Statement Amount": amt,
                "Difference": None,
                "Applied Amount": None,
                "Applied Entries": 0,
                "Value Date": val_date or txn_date,
                "Check No.": check_no,
                "Related-Party Name": ai.get("related_party_name"),
                "Additional Transaction Info": ai.get("additional_transaction_info"),
                "Statement Type": RECO_STATEMENT_TYPE,
                "Transaction Text": raw_desc,
                "Related-Party Bank Acc. No.": None,
                "Related-Party Address": None,
                "Related-Party City": None,
            })

            last_tx_date[acct_no] = txn_date or last_tx_date.get(acct_no)
            last_val_date[acct_no] = (val_date or txn_date) or last_val_date.get(acct_no)

        batch, batch_meta = [], []

    # accumulate batches (preserving order) from stitched txs
    for acct_no, stitched_txs, _appended, anchor_map in stitched_accounts:
        for idx, t in enumerate(stitched_txs or []):
            tx_desc = _getv(t, "Description")
            batch.append(t)
            batch_meta.append({
                "acct_no": acct_no,
                "desc": tx_desc,
                "source": (file.filename if file else url),
                "tx_obj": t,
                "anchor_no": anchor_map[idx],
            })
            if len(batch) >= AI_TX_BATCH_SIZE:
                _flush_batch()
    _flush_batch()

    log.info("ingest.bank_statement.rows_built", extra={"extra": {
        "rows": len(stage_rows),
        "reco_rows": len(reco_rows),
        "bank_name": bank_name,
        "statement_date": str(statement_date) if statement_date else None
    }})

    if not stage_rows:
        log.warning("ingest.bank_statement.no_rows_detected")
        return {"status": "no_rows_detected", "inserted": 0, "reco_inserted": 0, "total": 0}

    # --- Insert both sets -----------------------------------------------------
    inserted = 0
    reco_inserted = 0
    with log_duration("ingest.bank_statement.insert", rows=len(stage_rows)):
        async with db_conn() as conn:
            for r in stage_rows:
                ok = await conn.fetchval(
                    """
                    insert into erp_ai.transactions_staging
                    (txn_date, amount, currency, merchant_raw, merchant_norm, reference, source_file, fingerprint)
                    values ($1,$2,$3,$4,$5,$6,$7,$8)
                    on conflict (fingerprint) do nothing
                    returning 1;
                    """,
                    r["txn_date"], r["amount"], r["currency"],
                    r["merchant_raw"], r["merchant_norm"], r["reference"],
                    r["source_file"], r["fingerprint"]
                )
                if ok == 1:
                    inserted += 1

            for rr in reco_rows:
                ok2 = await conn.fetchval(
                    """
                    insert into erp_ai.bank_reconciliation_staging
                    ("Bank Account No.", "Bank Name", "Statement Date", "Reconciliation date",
                     "Statement Line No.", "Document No.", "Transaction Date", "Description",
                     "Statement Amount", "Difference", "Applied Amount", "Applied Entries",
                     "Value Date", "Check No.", "Related-Party Name", "Additional Transaction Info",
                     "Statement Type", "Transaction Text", "Related-Party Bank Acc. No.",
                     "Related-Party Address", "Related-Party City")
                    values
                    ($1,$2,$3,$4,
                     $5,$6,$7,$8,
                     $9,$10,$11,$12,
                     $13,$14,$15,$16,
                     $17,$18,$19,$20,$21)
                    returning 1;
                    """,
                    rr["Bank Account No."], rr["Bank Name"], rr["Statement Date"], rr["Reconciliation date"],
                    rr["Statement Line No."], rr["Document No."], rr["Transaction Date"], rr["Description"],
                    rr["Statement Amount"], rr["Difference"], rr["Applied Amount"], rr["Applied Entries"],
                    rr["Value Date"], rr["Check No."], rr["Related-Party Name"], rr["Additional Transaction Info"],
                    rr["Statement Type"], rr["Transaction Text"], rr["Related-Party Bank Acc. No."],
                    rr["Related-Party Address"], rr["Related-Party City"]
                )
                if ok2 == 1:
                    reco_inserted += 1

    log.info("ingest.bank_statement.done", extra={"extra": {
        "inserted": inserted, "reco_inserted": reco_inserted, "total": len(stage_rows)
    }})
    return {"status": "ok", "inserted": inserted, "reco_inserted": reco_inserted, "total": len(stage_rows)}

# ==== RECEIPT INGEST (LLM-normalized, minimal) ===============================
_LLM_RECEIPT_SYSTEM = """You normalize receipt fields into strict JSON.
- date -> 'YYYY-MM-DD' or null
- total -> number or null
- currency -> 3-letter code if present else null
- merchant -> cleaned merchant name (short) or null
Return: {"date": "...", "total": 0.0, "currency": "...", "merchant": "..."}"""

def _ai_norm_receipt(obj: Dict[str, Any], default_currency: str) -> Dict[str, Any]:
    if _openai_client is None:
        # deterministic fallback
        date_raw = obj.get("TransactionDate") or obj.get("Date") or obj.get("InvoiceDate") or obj.get("ReceiptDate")
        total_raw = obj.get("Total") or obj.get("TotalAmount") or obj.get("AmountDue") or obj.get("GrandTotal")
        merchant_raw = obj.get("MerchantName") or obj.get("VendorName") or obj.get("Supplier") or obj.get("Merchant")
        currency_raw = obj.get("Currency") or obj.get("CurrencyCode")
        return {
            "date": (_as_date(date_raw).isoformat() if _as_date(date_raw) else None),
            "total": _num_soft(total_raw),
            "currency": (str(currency_raw).upper() if currency_raw else default_currency),
            "merchant": merchant_raw,
        }

    user = f"Default Currency: {default_currency}\nRaw fields JSON:\n{json.dumps(obj, ensure_ascii=False, default=str)}"
    out = _openai_json([
        {"role": "system", "content": _LLM_RECEIPT_SYSTEM},
        {"role": "user", "content": user},
    ])
    return {
        "date": out.get("date"),
        "total": out.get("total"),
        "currency": out.get("currency") or default_currency,
        "merchant": out.get("merchant"),
    }

@router.post("/ingest/receipt")
async def ingest_receipt(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    currency: Optional[str] = Form(default=None),
):
    log.info("ingest.receipt.start", extra={"extra": {
        "has_file": bool(file), "has_url": bool(url),
        "filename": getattr(file, "filename", None), "url": url
    }})

    if not file and not url:
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                url = body.get("url")
            except Exception:
                pass
    if file and url:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'url'.")

    source = None
    with log_duration("ingest.receipt.analyze"):
        try:
            if file:
                source = file.filename
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                result = di_receipt.analyze_receipt_bytes(
                    data, getattr(file, "content_type", None)
                )
            elif url:
                source = url
                result = di_receipt.analyze_receipt_url(url)
            else:
                raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")
        except Exception as e:
            log.exception("ingest.receipt.di_error")
            raise HTTPException(status_code=502, detail=f"Document Intelligence error: {e}")

    shaped = di_receipt.receipt_to_dict(result)
    ccy = (currency or _infer_currency_from_result(result) or DEFAULT_CCY).upper()

    norm = _ai_norm_receipt(shaped.get("fields", {}), ccy)
    tx_date = _as_date(norm.get("date"))
    total = _num_soft(norm.get("total"))
    merch = norm.get("merchant")
    ccy_final = (norm.get("currency") or ccy).upper()

    fp = _fp([
        tx_date.isoformat() if tx_date else "",
        f"{(total or 0):.2f}",
        ccy_final,
        (merch or "").upper().strip(),
        source or "",
    ])

    inserted = 0
    with log_duration("ingest.receipt.insert"):
        async with db_conn() as conn:
            ok = await conn.fetchval(
                """
                insert into erp_ai.receipts_staging
                (receipt_date, total_amount, currency, merchant, source_file, fingerprint)
                values ($1,$2,$3,$4,$5,$6)
                on conflict (fingerprint) do nothing
                returning 1;
                """,
                tx_date, total, ccy_final, merch, source, fp
            )
            if ok == 1:
                inserted = 1

    log.info("ingest.receipt.done", extra={"extra": {"inserted": inserted}})
    return {"status": "ok", "inserted": inserted, "fingerprint": fp}


# ==== RECEIPT INGEST (LLM-normalized, minimal) ===============================
_LLM_RECEIPT_SYSTEM = """You normalize receipt fields into strict JSON.
- date -> 'YYYY-MM-DD' or null
- total -> number or null
- currency -> 3-letter code if present else null
- merchant -> cleaned merchant name (short) or null
Return: {"date": "...", "total": 0.0, "currency": "...", "merchant": "..."}"""

def _ai_norm_receipt(obj: Dict[str, Any], default_currency: str) -> Dict[str, Any]:
    if _openai_client is None:
        # deterministic fallback
        date_raw = obj.get("TransactionDate") or obj.get("Date") or obj.get("InvoiceDate") or obj.get("ReceiptDate")
        total_raw = obj.get("Total") or obj.get("TotalAmount") or obj.get("AmountDue") or obj.get("GrandTotal")
        merchant_raw = obj.get("MerchantName") or obj.get("VendorName") or obj.get("Supplier") or obj.get("Merchant")
        currency_raw = obj.get("Currency") or obj.get("CurrencyCode")
        return {
            "date": (_as_date(date_raw).isoformat() if _as_date(date_raw) else None),
            "total": _num_soft(total_raw),
            "currency": (str(currency_raw).upper() if currency_raw else default_currency),
            "merchant": merchant_raw,
        }

    user = f"Default Currency: {default_currency}\nRaw fields JSON:\n{json.dumps(obj, ensure_ascii=False, default=str)}"
    out = _openai_json([
        {"role": "system", "content": _LLM_RECEIPT_SYSTEM},
        {"role": "user", "content": user},
    ])
    return {
        "date": out.get("date"),
        "total": out.get("total"),
        "currency": out.get("currency") or default_currency,
        "merchant": out.get("merchant"),
    }


@router.post("/ingest/receipt")
async def ingest_receipt(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    url:  Optional[str]       = Form(default=None),
):
    log.info("ingest.receipt.start", extra={"extra": {
        "has_file": bool(file), "has_url": bool(url),
        "filename": getattr(file, "filename", None), "url": url
    }})

    if not file and not url:
        if request.headers.get("content-type","").startswith("application/json"):
            try:
                body = await request.json()
                url = body.get("url")
            except Exception:
                pass
    if file and url:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'url'.")

    source = None
    with log_duration("ingest.receipt.analyze"):
        try:
            if file:
                source = file.filename
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                result = di_receipt.analyze_receipt_bytes(data, getattr(file, "content_type", None))
            elif url:
                source = url
                result = di_receipt.analyze_receipt_url(url)
            else:
                raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")
        except Exception as e:
            log.exception("ingest.receipt.di_error")
            raise HTTPException(status_code=502, detail=f"Document Intelligence error: {e}")

    shaped = di_receipt.receipt_to_dict(result)
    f = shaped.get("fields", {}) or {}

    ai = _ai_norm_receipt(f, DEFAULT_CCY)
    receipt_date = _as_date(ai.get("date"))
    total = _num_soft(ai.get("total"))
    currency = (ai.get("currency") or DEFAULT_CCY).upper()
    merchant = ai.get("merchant")
    merch_norm = _norm_merchant(merchant)

    if receipt_date is None or total is None:
        log.warning("ingest.receipt.missing_core_fields", extra={"extra": {
            "date_raw": ai.get("date"), "total_raw": ai.get("total")
        }})
        raise HTTPException(status_code=422, detail="Could not parse receipt date/total.")

    fp = _fp([receipt_date, f"{total:.2f}", merch_norm])

    inserted = 0
    with log_duration("ingest.receipt.insert"):
        async with db_conn() as conn:
            ok = await conn.fetchval(
                """
                insert into erp_ai.receipts_staging
                (receipt_date, total_amount, currency, merchant_name, merchant_norm, source_file, fingerprint)
                values ($1,$2,$3,$4,$5,$6,$7)
                on conflict (fingerprint) do nothing
                returning 1;
                """,
                receipt_date, total, currency, merchant, merch_norm, source, fp
            )
            if ok == 1:
                inserted = 1

    log.info("ingest.receipt.done", extra={"extra": {
        "inserted": inserted, "date": str(receipt_date), "total": total,
        "currency": currency, "merchant": merchant, "fingerprint": fp
    }})

    return {
        "status": "ok",
        "receipt": {
            "date": str(receipt_date), "total": total, "currency": currency,
            "merchant": merchant, "fingerprint": fp
        },
        "inserted": inserted
    }
