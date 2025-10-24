# app/routers/tx_ingest.py
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from datetime import datetime, date
import hashlib, re, os

from app.db import db_conn
from app.di_extractors import bank_statement as di_bank
from app.di_extractors import receipt as di_receipt
from app.utils.logger import get_logger, log_duration

router = APIRouter(prefix="/tx", tags=["transactions-ingest"])
log = get_logger(__name__)

DEFAULT_CCY = os.getenv("DEFAULT_CURRENCY", "AED")
RECO_STATEMENT_TYPE = os.getenv("RECO_STATEMENT_TYPE", "Bank Reconciliation")

# ------------------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------------------
_DATE_PATTERNS: List[Tuple[str, str]] = [
    # ISO & Y/M/D
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),

    # D/M/Y numeric
    (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
    (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y"),
    (r"^\d{2}/\d{2}/\d{2}$",  "%d/%m/%y"),
    (r"^\d{2}-\d{2}-\d{2}$",  "%d-%m-%y"),

    # M/D/Y numeric (US-style)
    (r"^\d{1,2}/\d{1,2}/\d{4}$", "%m/%d/%Y"),
    (r"^\d{1,2}-\d{1,2}-\d{4}$", "%m-%d-%Y"),
    (r"^\d{1,2}/\d{1,2}/\d{2}$",  "%m/%d/%y"),
    (r"^\d{1,2}-\d{1,2}-\d{2}$",  "%m-%d-%y"),

    # Day-first with month names
    (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$",  "%d %b %Y"),
    (r"^\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}$", "%d %B %Y"),
    (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{2}$",  "%d %b %y"),
    (r"^\d{1,2}\s+[A-Za-z]{3,}\s+\d{2}$", "%d %B %y"),
    (r"^\d{1,2}/[A-Za-z]{3}/\d{4}$",      "%d/%b/%Y"),
    (r"^\d{1,2}-[A-Za-z]{3}-\d{4}$",      "%d-%b-%Y"),
    (r"^\d{1,2}/[A-Za-z]{3}/\d{2}$",      "%d/%b/%y"),
    (r"^\d{1,2}-[A-Za-z]{3}-\d{2}$",      "%d-%b-%y"),

    # Month-first with names (with/without comma)
    (r"^[A-Za-z]{3,}\s+\d{1,2},\s*\d{4}$", "%B %d, %Y"),
    (r"^[A-Za-z]{3}\s+\d{1,2},\s*\d{4}$",  "%b %d, %Y"),
    (r"^[A-Za-z]{3,}\s+\d{1,2}\s+\d{4}$",  "%B %d %Y"),
    (r"^[A-Za-z]{3}\s+\d{1,2}\s+\d{4}$",   "%b %d %Y"),
]

# date tokens we’ll also try to discover inside descriptions
_DESC_DATE_TOKENS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
    r"\b\d{2}-\d{2}-\d{4}\b",
    r"\b\d{4}/\d{2}/\d{2}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,}\s+\d{2}\b",
    r"\b\d{1,2}/[A-Za-z]{3}/\d{4}\b",
    r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",
    r"\b\d{1,2}/[A-Za-z]{3}/\d{2}\b",
    r"\b\d{1,2}-[A-Za-z]{3}-\d{2}\b",
    # numeric day/month without year like 31/07 or 07-31 or 31.7
    r"\b\d{1,2}[./-]\d{1,2}\b",
]

_RX_DAY_MON_NOYEAR = re.compile(r"^(\d{1,2})\s+([A-Za-z]{3,})$")
_RX_NUM_NOYEAR = re.compile(r"^(\d{1,2})[./-](\d{1,2})$")

def _try_parse_with_month_case(s: str, fmt: str) -> Optional[date]:
    if "%b" in fmt or "%B" in fmt:
        parts = s.split()
        s = " ".join(p.capitalize() if p.isalpha() else p for p in parts)
    try:
        return datetime.strptime(s, fmt).date()
    except Exception:
        return None

def _try_parse_numeric_no_year(s: str, default_year: Optional[int]) -> Optional[date]:
    """
    Accept "31/07", "07-31", "31.7" etc and inject default_year.
    Prefer **day-first** (DD/MM) for GCC banks when ambiguous.
    """
    if not default_year:
        return None
    m = _RX_NUM_NOYEAR.match(s.strip())
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    # decide ordering
    if a > 12 and 1 <= b <= 12:
        day, mon = a, b
    elif b > 12 and 1 <= a <= 12:
        day, mon = b, a
    else:
        day, mon = a, b  # ambiguous -> DD/MM
    # build and test
    for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(f"{day:02d}/{mon:02d}/{default_year}", fmt).date()
        except Exception:
            continue
    return None

def _to_date_strict(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v).strip()
    for rx, fmt in _DATE_PATTERNS:
        if re.match(rx, s):
            d = _try_parse_with_month_case(s, fmt)
            if d:
                return d
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        return None

def _to_date_flex(v: Any, default_year: Optional[int] = None) -> Optional[date]:
    # Try strict formats first
    d = _to_date_strict(v)
    if d is not None:
        return d
    s = (str(v) if v is not None else "").strip()
    if not s:
        return None
    # numeric day/month without year
    d2 = _try_parse_numeric_no_year(s, default_year)
    if d2:
        return d2
    # "DD Mon" with no year → inject year
    m = _RX_DAY_MON_NOYEAR.match(s)
    if m and default_year:
        day, mon = m.group(1), m.group(2)
        for fmt in ("%d %b %Y", "%d %B %Y"):
            d3 = _try_parse_with_month_case(f"{day} {mon} {default_year}", fmt)
            if d3:
                return d3
    return None

def _to_num(v: Any):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    neg = "(" in s and ")" in s
    s = s.replace(",", "")
    s2 = re.sub(r"[^\d.\-]", "", s)
    if s2 in ("", ".", "-", "-."):
        return None
    try:
        n = float(s2)
        return -abs(n) if neg and n > 0 else n
    except Exception:
        return None

def _norm_merchant(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    x = s.upper().strip()
    x = re.sub(r"[^A-Z0-9 &\.\-]", " ", x)
    x = re.sub(r"\b(LLC|LTD|CO|INC|PJSC|PVT)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x or None

def _fp(parts: List[str]) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _getv(d: Dict[str, Any], *keys: str) -> Any:
    """Return d[k]['value'] if present, else d[k] if primitive/non-null."""
    for k in keys:
        if k in d:
            obj = d[k]
            if isinstance(obj, dict) and "value" in obj:
                return obj["value"]
            if obj is not None:
                return obj
    return None

def _date_from_tx(tx: Dict[str, Any], desc: Optional[str], stmt_year: Optional[int]) -> Optional[date]:
    # Include common date fields; fall back to description scanning
    raw = _getv(tx, "Date", "TransactionDate", "PostingDate", "PostDate", "ValueDate", "BookingDate")
    dt = _to_date_flex(raw, default_year=stmt_year)
    if dt is not None:
        return dt
    if desc:
        for rx in _DESC_DATE_TOKENS:
            m = re.search(rx, desc)
            if m:
                dt2 = _to_date_flex(m.group(0), default_year=stmt_year)
                if dt2:
                    return dt2
    return None

def _amount_from_tx_raw_sign(num: Optional[float], raw: Any) -> Optional[float]:
    if num is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw or "")
    if s.strip().startswith("-") or ("(" in s and ")" in s):
        return -abs(num)
    return num

def _signed_amount_from_tx(tx: Dict[str, Any]) -> Optional[float]:
    raw_amount = _getv(tx, "Amount", "TransactionAmount")
    amt = _to_num(raw_amount)
    if amt is not None:
        tval = str(_getv(tx, "Type", "CreditDebit", "TransactionType") or "").strip().lower()
        if tval in ("debit", "dr", "withdrawal", "payment", "out"):
            return -abs(amt)
        if tval in ("credit", "cr", "deposit", "in"):
            return abs(amt)
        return _amount_from_tx_raw_sign(abs(amt), raw_amount)

    dep = _to_num(_getv(tx, "DepositAmount", "Credit", "PaidIn", "In"))
    wdr = _to_num(_getv(tx, "WithdrawalAmount", "Debit", "PaidOut", "Out"))
    if dep is not None or wdr is not None:
        dep = dep or 0.0
        wdr = wdr or 0.0
        return dep - abs(wdr)

    credit = _to_num(_getv(tx, "Credit"))
    debit  = _to_num(_getv(tx, "Debit"))
    if credit is not None:
        return abs(credit)
    if debit is not None:
        return -abs(debit)
    return None

# ------------------------------------------------------------------------------
# Currency helpers
# ------------------------------------------------------------------------------
_CURRENCY_HINTS = {
    "£": "GBP", "GBP": "GBP", "€": "EUR", "EUR": "EUR", "$": "USD", "USD": "USD",
    "AED": "AED", "SAR": "SAR", "QAR": "QAR", "OMR": "OMR", "BHD": "BHD"
}

def _infer_currency_from_result(di_result: Any) -> Optional[str]:
    """Scan DI result lines/tables for currency symbols/codes."""
    try:
        for p in getattr(di_result, "pages", []) or []:
            for ln in getattr(p, "lines", []) or []:
                txt = (getattr(ln, "content", "") or "").upper()
                for token, code in _CURRENCY_HINTS.items():
                    if token in txt:
                        return code
        for tbl in getattr(di_result, "tables", []) or []:
            for c in getattr(tbl, "cells", []) or []:
                txt = (getattr(c, "content", "") or "").upper()
                for token, code in _CURRENCY_HINTS.items():
                    if token in txt:
                        return code
    except Exception:
        pass
    return None

_CHANNEL_PREFIX = re.compile(r"^(BACS|FASTER PAYMENT|CHQ|CHEQUE|CARD|POS|ATM|BANK)\b[\s:-]*", re.I)

def _clean_channel_prefix(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = _CHANNEL_PREFIX.sub("", s or "").strip()
    return t or s

_REF_RX = re.compile(r"(AUTH|APPROVAL|RRN|REF|CHQ|CHEQUE)[\s:#-]*([A-Z0-9]{4,})", re.I)

# ------------------------------------------------------------------------------
# Year inference helpers
# ------------------------------------------------------------------------------
def _collect_years_from_any(obj: Any) -> List[int]:
    years: List[int] = []
    def _walk(x: Any):
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        elif isinstance(x, str):
            for m in re.finditer(r"(19|20)\d{2}", x):
                y = int(m.group(0))
                if 1900 <= y <= 2100:
                    years.append(y)
    _walk(obj)
    return years

def _collect_months_from_accounts(accounts: List[Dict[str, Any]]) -> List[int]:
    months: List[int] = []
    mon_map = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
    }
    for acc in accounts or []:
        for t in (acc.get("Transactions") or []):
            v = _getv(t, "Date", "TransactionDate", "PostingDate", "PostDate", "ValueDate", "BookingDate")
            s = (str(v) if v is not None else "").strip().lower()
            m1 = re.match(r"^\d{1,2}\s+([a-z]{3,})$", s)
            m2 = re.match(r"^([a-z]{3,})\s+\d{1,2}$", s)
            token = (m1.group(1) if m1 else (m2.group(1) if m2 else None))
            if token:
                key = token.replace("sept","sep")[:3]
                mm = mon_map.get(key, 0)
                if 1 <= mm <= 12:
                    months.append(mm)
    return months

def _infer_year_from_months(tx_months: List[int], now: Optional[date] = None) -> Optional[int]:
    if not tx_months:
        return None
    now = now or date.today()
    return now.year

def _resolve_stmt_year(shaped: Dict[str, Any], accounts: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    """
    Try multiple strategies to get a year:
    1) Concrete year from shaped->fields StatementEndDate/StartDate
    2) Any 4-digit year anywhere in the shaped dict
    3) Infer from months present in the transaction rows
    """
    fields = shaped.get("fields") or {}
    stmt_end = (fields.get("StatementEndDate") or {}).get("value")
    stmt_start = (fields.get("StatementStartDate") or {}).get("value")

    for label, candidate in (("end_date", stmt_end), ("start_date", stmt_start)):
        d = _to_date_strict(candidate)
        if d:
            return d.year, f"from_{label}"

    years = _collect_years_from_any(shaped)
    if years:
        return max(years), "from_text_scan"

    months = _collect_months_from_accounts(accounts)
    y = _infer_year_from_months(months, date.today())
    if y:
        return y, "from_months_inference"

    return None, "unknown"

# ------------------------------------------------------------------------------
# Deep scan + canonical date extraction
# ------------------------------------------------------------------------------
def _deep_scan_key_like(obj: Any, include_tokens: Tuple[str, ...], exclude_tokens: Tuple[str, ...] = ()) -> Optional[Any]:
    """
    Recursively search dict/list for a key whose normalized name contains all include_tokens
    and none of exclude_tokens; return its primitive value (prefer .get('value') or .get('content')).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = k.replace(" ", "").replace("_", "").lower()
            if all(t in kk for t in include_tokens) and not any(t in kk for t in exclude_tokens):
                if isinstance(v, dict):
                    if "value" in v: return v.get("value")
                    if "content" in v: return v.get("content")
                return v
            r = _deep_scan_key_like(v, include_tokens, exclude_tokens)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for it in obj:
            r = _deep_scan_key_like(it, include_tokens, exclude_tokens)
            if r is not None:
                return r
    return None

def _parse_any_date(v: Any, year_hint: Optional[int]) -> Optional[date]:
    d = _to_date_strict(v)
    if d is None and isinstance(v, str):
        d = _to_date_flex(v, year_hint)
    return d

def _extract_tx_and_value_dates(t: Dict[str, Any], year_hint: Optional[int]) -> Tuple[Optional[date], Optional[date]]:
    """
    FAB (and most GCC bank statements) print two separate date columns:
      - Date (posting/booking date)  -> should be our Transaction Date
      - Value Date                   -> should be our Value Date

    DI sometimes mislabels/extracts these. We must:
      1) Treat Posting/Booking/Date as *transaction* date candidates.
      2) Treat only ValueDate/ValDate as *value* date candidates.
      3) Never copy value->transaction. (Only fill value from transaction if missing.)
    """

    # 1) Prefer posting/booking/date for the transaction date (printed "Date" column)
    txn_raw = _getv(
        t,
        "TransactionDate", "TxnDate", "TransDate", "TranDate",
        "PostingDate", "PostDate", "BookingDate", "BookDate",
        "Date",
    )

    # 2) Restrict value date to explicit value-date fields
    val_raw = _getv(t, "ValueDate", "ValDate", "Value Date")

    # Deep scans: steer posting/booking/date to txn; value date to val
    if txn_raw is None:
        txn_raw = (
            _deep_scan_key_like(t, ("posting", "date"))
            or _deep_scan_key_like(t, ("book", "date"))
            or _deep_scan_key_like(t, ("date",), ("value",))  # bare "date" but NOT "value date"
        )
    if val_raw is None:
        val_raw = _deep_scan_key_like(t, ("value", "date"))

    txn_date = _parse_any_date(txn_raw, year_hint)
    val_date = _parse_any_date(val_raw, year_hint)

    # Last resort for transaction date: parse from description tokens using year hint
    if txn_date is None:
        desc = _getv(t, "Description")
        txn_date = _date_from_tx(t, desc, year_hint)

    # IMPORTANT: Only fill Value Date from Transaction Date when missing (never the other way)
    if val_date is None and txn_date is not None:
        val_date = txn_date

    return txn_date, val_date


# ------------------------------------------------------------------------------
# Bank name inference (robust, UAE-focused; prioritizes specificity)
# ------------------------------------------------------------------------------
def _infer_bank_name_from_text(di_result: Any, filename_hint: Optional[str]) -> Tuple[Optional[str], str]:
    # Gather text
    first_page_text = ""
    all_text = ""
    try:
        pages = getattr(di_result, "pages", []) or []
        if pages:
            first_page_text = "\n".join((ln.content or "") for ln in (pages[0].lines or []))
        all_text = "\n".join((ln.content or "") for p in pages for ln in (p.lines or []))
    except Exception:
        pass

    # Strong identifiers (BIC/SWIFT → official)
    bic_map = {
        "CBDUAEAD": "Commercial Bank of Dubai",
        "ADCBAEAA": "Abu Dhabi Commercial Bank",
        "FABUAEAD": "First Abu Dhabi Bank",
        "NBADAEAA": "First Abu Dhabi Bank",
        "EBILAEAD": "Emirates NBD",
        "MSHQAEAD": "Mashreq Bank",
        "DUIBAEAD": "Dubai Islamic Bank",
        "RAKBAEAA": "RAKBANK",
        "AEDIAEAD": "Emirates Islamic",
    }
    upper_all = all_text.upper()
    for bic, name in bic_map.items():
        if bic in upper_all:
            return name, "from_bic_code"

    # Specific phrase matches (most specific first)
    phrase_order: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\bCOMMERCIAL\s+BANK\s+OF\s+DUBAI\b", re.I), "Commercial Bank of Dubai"),
        (re.compile(r"\bCBD\b(?![A-Z])", re.I), "Commercial Bank of Dubai"),
        (re.compile(r"\bABU\s+DHABI\s+COMMERCIAL\s+BANK\b", re.I), "Abu Dhabi Commercial Bank"),
        (re.compile(r"\bADCB\b", re.I), "Abu Dhabi Commercial Bank"),
        (re.compile(r"\bFIRST\s+ABU\s+DHABI\b", re.I), "First Abu Dhabi Bank"),
        (re.compile(r"\bFAB\b(?!RIC)", re.I), "First Abu Dhabi Bank"),
        (re.compile(r"\bEMIRATES\s+NBD\b", re.I), "Emirates NBD"),
        (re.compile(r"\bMASHREQ\b", re.I), "Mashreq Bank"),
        (re.compile(r"\bDUBAI\s+ISLAMIC\s+BANK\b", re.I), "Dubai Islamic Bank"),
        (re.compile(r"\bDIB\b", re.I), "Dubai Islamic Bank"),
        (re.compile(r"\bRAK\s*BANK\b|\bRAKBANK\b", re.I), "RAKBANK"),
        (re.compile(r"\bEMIRATES\s+ISLAMIC\b", re.I), "Emirates Islamic"),
        (re.compile(r"\bSTANDARD\s+CHARTERED\b", re.I), "Standard Chartered"),
        (re.compile(r"\bHSBC\b", re.I), "HSBC"),
    ]

    # Prefer first page (letterhead) first
    for rx, name in phrase_order:
        if rx.search(first_page_text):
            return name, "from_first_page_text"

    # Then full text
    for rx, name in phrase_order:
        if rx.search(all_text):
            return name, "from_full_text"

    # Finally look at filename hints
    if filename_hint:
        s = str(filename_hint).upper()
        if "COMMERCIAL BANK OF DUBAI" in s or " CBD " in f" {s} " or s.startswith("CBD"):
            return "Commercial Bank of Dubai", "from_filename"
        if "ABU DHABI COMMERCIAL" in s or "ADCB" in s:
            return "Abu Dhabi Commercial Bank", "from_filename"
        if "FIRST ABU DHABI" in s or " FAB " in f" {s} " or s.startswith("FAB"):
            return "First Abu Dhabi Bank", "from_filename"
        if "EMIRATES NBD" in s or "ENBD" in s:
            return "Emirates NBD", "from_filename"
        if "MASHREQ" in s:
            return "Mashreq Bank", "from_filename"
        if "DUBAI ISLAMIC" in s or "DIB" in s:
            return "Dubai Islamic Bank", "from_filename"
        if "RAKBANK" in s or "RAK BANK" in s:
            return "RAKBANK", "from_filename"
        if "EMIRATES ISLAMIC" in s:
            return "Emirates Islamic", "from_filename"
        if "STANDARD CHARTERED" in s:
            return "Standard Chartered", "from_filename"
        if "HSBC" in s:
            return "HSBC", "from_filename"

    return None, "unknown"

# ------------------------------------------------------------------------------
# ROUTE: POST /tx/ingest/bank-statement
# ------------------------------------------------------------------------------
@router.post("/ingest/bank-statement")
async def ingest_bank_statement(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    url:  Optional[str]       = Form(default=None),
    currency: Optional[str]   = Form(default=None),
):
    log.info("ingest.bank_statement.start", extra={"extra": {
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
    with log_duration("ingest.bank_statement.analyze"):
        try:
            if file:
                source = file.filename
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                result = di_bank.analyze_bank_stmt_bytes(data, getattr(file, "content_type", None))
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

    # Resolve statement year once
    stmt_year, stmt_year_strategy = _resolve_stmt_year(shaped, accounts)
    log.info("ingest.bank_statement.year_resolve", extra={"extra": {
        "stmt_year": stmt_year, "strategy": stmt_year_strategy
    }})

    # Bank name with robust fallback (CBD prioritized properly)
    bank_name = (fields.get("BankName") or {}).get("value")
    bank_strategy = "from_model_field" if bank_name else None
    if not bank_name:
        bank_name, bank_strategy = _infer_bank_name_from_text(result, source)
    log.info("ingest.bank_statement.bank_name_resolve", extra={"extra": {
        "bank_name": bank_name, "strategy": bank_strategy
    }})

    statement_date = _to_date_strict((fields.get("StatementEndDate") or {}).get("value"))

    model_tx_count = sum(len(a.get("Transactions") or []) for a in accounts)
    log.info("ingest.bank_statement.shaped", extra={"extra": {
        "accounts": len(accounts), "model_tx_count": model_tx_count, "stmt_year": stmt_year
    }})

    # Preview (just for logging)
    preview: List[Dict[str, Any]] = []
    for acc in accounts[:1]:
        for t in (acc.get("Transactions") or [])[:5]:
            td, vd = _extract_tx_and_value_dates(t, stmt_year)
            preview.append({
                "TransactionDate": td.isoformat() if td else None,
                "ValueDate": vd.isoformat() if vd else None,
                "Desc": _getv(t, "Description"),
                "Deposit": _getv(t, "DepositAmount", "Credit", "PaidIn"),
                "Withdraw": _getv(t, "WithdrawalAmount", "Debit", "PaidOut"),
                "Amount": _getv(t, "Amount", "TransactionAmount"),
                "Type": _getv(t, "Type", "CreditDebit", "TransactionType"),
            })
    if preview:
        log.info("ingest.bank_statement.preview", extra={"extra": {"sample": preview}})

    # currency (form -> inferred -> default)
    ccy = (currency or _infer_currency_from_result(result) or DEFAULT_CCY).upper()

    # Build rows for both tables
    stage_rows: List[Dict[str, Any]] = []
    reco_rows:  List[Dict[str, Any]] = []

    c_missing_amt = c_missing_date = c_skipped_blank = 0
    missing_date_samples: List[Dict[str, Any]] = []
    backfilled_tx_total = 0
    backfilled_val_total = 0

    for acc in accounts:
        acct_no = (acc.get("AccountNumber") or {}).get("value")
        line_no = 0  # per-account line numbering

        # rolling dates (fix continuation lines)
        last_tx_date: Optional[date] = None
        last_val_date: Optional[date] = None

        for t in (acc.get("Transactions") or []):
            desc = _getv(t, "Description")

            # skip pure blanks
            if not any([
                _getv(t, "Date", "TransactionDate", "PostingDate", "PostDate", "ValueDate", "BookingDate"),
                desc,
                _getv(t, "Amount", "TransactionAmount", "DepositAmount", "WithdrawalAmount", "Credit", "Debit", "PaidIn", "PaidOut")
            ]):
                c_skipped_blank += 1
                continue

            amt = _signed_amount_from_tx(t)
            if amt is None:
                c_missing_amt += 1
                continue

            tx_date, val_date = _extract_tx_and_value_dates(t, stmt_year)

            # --- backfill dates when DI split the row and dropped the dates ---
            if tx_date is None and last_tx_date is not None:
                tx_date = last_tx_date
                backfilled_tx_total += 1
            if val_date is None and last_val_date is not None:
                val_date = last_val_date
                backfilled_val_total += 1

            if tx_date is None:
                c_missing_date += 1
                if len(missing_date_samples) < 6:
                    missing_date_samples.append({
                        "desc": desc,
                        "raw_date_candidates": {
                            "Date": _getv(t, "Date"),
                            "TransactionDate": _getv(t, "TransactionDate", "TxnDate", "TransDate", "TranDate"),
                            "PostingDate": _getv(t, "PostingDate", "PostDate"),
                            "ValueDate": _getv(t, "ValueDate", "ValDate", "Value Date"),
                            "BookingDate": _getv(t, "BookingDate", "BookDate"),
                        }
                    })
                continue

            # stage (legacy): keep merchant_* behavior as-is
            desc_clean = _clean_channel_prefix(desc)
            merch_norm = _norm_merchant(desc_clean)

            ref = None
            m = _REF_RX.search(desc or "")
            if m:
                ref = m.group(2)

            next_line_no = line_no + 1

            fp = _fp([
                acct_no,                 # account number in the statement
                statement_date,          # statement end date (acts as the statement identity)
                next_line_no,            # per-statement line number (stable order in PDF)
                f"{amt:.2f}",            # keep old parts too (harmless, extra stability)
                ccy,
                merch_norm,
                ref,
            ])

            stage_rows.append({
                "txn_date": tx_date, "amount": amt, "currency": ccy,
                "merchant_raw": desc, "merchant_norm": merch_norm,
                "reference": ref, "source_file": source, "fingerprint": fp
            })

            # reconciliation row
            line_no += 1
            check_no = None
            if m and m.group(1).upper() in ("CHQ", "CHEQUE"):
                check_no = m.group(2)

            reco_rows.append({
                "Bank Account No.": acct_no,
                "Bank Name": bank_name,
                "Statement Date": statement_date,          # statement-end as statement date
                "Reconciliation date": statement_date,     # same default
                "Statement Line No.": line_no,
                "Document No.": None,                      # unknown at parse time
                "Transaction Date": tx_date,
                "Description": (desc_clean or desc),
                "Statement Amount": amt,                   # signed amount
                "Difference": None,                        # ERP can compute later
                "Applied Amount": None,
                "Applied Entries": 0,
                "Value Date": val_date or tx_date,         # always filled if tx_date exists
                "Check No.": check_no,
                "Related-Party Name": None,
                "Additional Transaction Info": ref,        # store reference if any
                "Statement Type": RECO_STATEMENT_TYPE,     # per requirement
                "Transaction Text": (desc_clean or desc),
                "Related-Party Bank Acc. No.": None,
                "Related-Party Address": None,
                "Related-Party City": None,
            })

            # update rolling dates now that this row is accepted
            if tx_date is not None:
                last_tx_date = tx_date
            if val_date is not None:
                last_val_date = val_date

    log.info("ingest.bank_statement.rows_built", extra={"extra": {
        "rows": len(stage_rows),
        "reco_rows": len(reco_rows),
        "skipped_missing_amount": c_missing_amt,
        "skipped_missing_date": c_missing_date,
        "skipped_pure_blank": c_skipped_blank,
        "missing_date_samples": missing_date_samples,
        "backfilled_tx_date": backfilled_tx_total,
        "backfilled_val_date": backfilled_val_total,
    }})

    if not stage_rows:
        log.warning("ingest.bank_statement.no_rows_detected")
        return {"status": "no_rows_detected", "inserted": 0, "reco_inserted": 0, "total": 0}

    # Insert both sets
    inserted = 0
    reco_inserted = 0
    with log_duration("ingest.bank_statement.insert", rows=len(stage_rows)):
        async with db_conn() as conn:
            # transactions_staging
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

            # bank_reconciliation_staging
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

# ------------------------------------------------------------------------------
# ROUTE: POST /tx/ingest/receipt
# ------------------------------------------------------------------------------
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

    raw_date  = (f.get("TransactionDate") or {}).get("value")
    raw_total = (f.get("Total")            or {}).get("value")
    currency  = (f.get("Currency")         or {}).get("value") or DEFAULT_CCY
    merchant  = (f.get("MerchantName")     or {}).get("value")
    merch_norm = _norm_merchant(merchant)

    log.info("ingest.receipt.shaped", extra={"extra": {
        "has_date": raw_date is not None, "has_total": raw_total is not None,
        "has_currency": currency is not None, "has_merchant": merchant is not None
    }})

    def _to_date_any(v: Any):
        d = _to_date_strict(v)
        if d is None and isinstance(v, str):
            d = _to_date_flex(v, None)
        return d

    receipt_date = _to_date_any(raw_date)
    total        = _to_num(raw_total)

    if receipt_date is None or total is None:
        log.warning("ingest.receipt.missing_core_fields", extra={"extra": {
            "date_raw": raw_date, "total_raw": raw_total
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
