# app/routers/tx_ingest.py
from __future__ import annotations

import os, re, json, hashlib
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
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""))
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
        for fmt in ("%Y-%m-%d","%d-%b-%Y","%d-%B-%Y","%d/%m/%Y","%Y/%m/%d","%m/%d/%Y"):
            try: return datetime.strptime(s, fmt).date()
            except Exception: pass
        try: return datetime.fromisoformat(s.replace("Z","+00:00")).date()
        except Exception:
            try: return date.fromisoformat(s[:10])
            except Exception: return None
    return None

def _num_soft(v: Any) -> Optional[float]:
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v)
    neg = "(" in s and ")" in s
    s2 = re.sub(r"[^0-9.\-]", "", s)
    if s2 in ("",".","-","-."): return None
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

def _fp(parts: List[str]) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _infer_currency_from_result(di_result: Any) -> Optional[str]:
    hints = {"£":"GBP","GBP":"GBP","€":"EUR","EUR":"EUR","$":"USD","USD":"USD",
             "AED":"AED","SAR":"SAR","QAR":"QAR","OMR":"OMR","BHD":"BHD"}
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
    m = re.search(r"Statement\s+generated\s+on:\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", text, re.I)
    if m:
        d = _as_date(m.group(1))
        if d: return d
    m2 = re.search(r"Account\s+statement:\s*.*?\bto\b\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", text, re.I | re.S)
    if m2:
        d = _as_date(m2.group(1))
        if d: return d
    max_tx: Optional[date] = None
    for acc in accounts or []:
        for t in (acc.get("Transactions") or []):
            cand = _as_date(_getv(t, "TransactionDate","PostingDate","PostDate","Date","BookingDate","ValueDate"))
            if cand and (max_tx is None or cand > max_tx):
                max_tx = cand
    return max_tx

# ==== Amount + continuation utilities ========================================
def _signed_amount_from_tx(tx: Dict[str, Any]) -> Optional[float]:
    raw_amount = _getv(tx, "Amount", "TransactionAmount")
    amt = _num_soft(raw_amount)
    if amt is not None:
        tval = str(_getv(tx, "Type","CreditDebit","TransactionType") or "").strip().lower()
        if tval in ("debit","dr","withdrawal","payment","out","card debit","debit card transaction"):
            return -abs(amt)
        if tval in ("credit","cr","deposit","in","salary transfer","salary","paid in"):
            return abs(amt)
        s = str(raw_amount or "")
        if s.strip().startswith("-") or ("(" in s and ")" in s):
            return -abs(amt)
        return amt

    dep = _num_soft(_getv(tx, "DepositAmount","Credit","PaidIn","In"))
    wdr = _num_soft(_getv(tx, "WithdrawalAmount","Debit","PaidOut","Out"))
    if dep is not None or wdr is not None:
        dep = dep or 0.0
        wdr = wdr or 0.0
        return dep - abs(wdr)

    credit = _num_soft(_getv(tx, "Credit"))
    debit  = _num_soft(_getv(tx, "Debit"))
    if credit is not None: return abs(credit)
    if debit  is not None: return -abs(debit)
    return None

def _desc_of(tx: Dict[str, Any]) -> Optional[str]:
    return _getv(tx, "Description")

def _candidate_dates(tx: Dict[str, Any]) -> Set[date]:
    ks = ("TransactionDate","PostingDate","PostDate","Date","BookingDate","ValueDate","ValDate","Value Date")
    out: Set[date] = set()
    for k in ks:
        d = _as_date(_getv(tx, k))
        if d: out.add(d)
    return out

def _is_continuation(prev_anchor: Dict[str, Any], curr: Dict[str, Any]) -> bool:
    """
    Treat `curr` as a continuation of `prev_anchor` when:
      - curr has NO amount, AND
      - (curr has NO date) OR (curr's date(s) equal prev's date(s))
    This captures visual wrap rows where DI repeats dates but omits amount.
    """
    if not prev_anchor or not curr: return False
    if not ((_desc_of(curr) or "").strip()): return False
    if _signed_amount_from_tx(curr) is not None:  # continuation rows never carry amount
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

# ==== LLM mapping =============================================================
_LLM_SYSTEM = """You are a finance extraction engine that maps raw bank statement transaction objects
(from OCR/Document-Intelligence) into a strict ERP row schema. Return STRICT JSON only.
Never hallucinate; use null when a field is not present. Rules:

- Dates: 'YYYY-MM-DD' or null.
- Amount sign: credits (paid in / deposit / credit / salary) are POSITIVE; debits are NEGATIVE.
- 'description': transaction kind (e.g., 'Debit Card Transaction', 'ATM Cash Withdrawal', 'Salary Transfer'), concise.
- 'transaction_text': merchant/counterparty text (e.g., 'CARREFOUR RAK', 'FAB', 'KEYSTONE GROUP LLC FZ').
- 'check_no': if cheque is mentioned, extract cheque number (4+ alphanum) else null.
- 'related_party_name': counterparty or merchant when clearly present.
- 'additional_transaction_info' and 'reference': approval/ref/rrn/auth-like tokens when present.
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
  "reference"
]

Input transactions JSON:
{tx_json}
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

def _ai_map_rows(
    tx_items: List[Dict[str, Any]],
    statement_end_date: Optional[date],
    bank_name: Optional[str],
    default_currency: str,
) -> List[Dict[str, Any]]:
    user = _LLM_USER_TEMPLATE.format(
        statement_end_date=statement_end_date.isoformat() if statement_end_date else None,
        bank_name=bank_name,
        default_currency=default_currency,
        n=len(tx_items),
        tx_json=json.dumps(tx_items, ensure_ascii=False, default=str),
    )
    out = _openai_json([
        {"role": "system", "content": _LLM_SYSTEM},
        {"role": "user", "content": user},
    ])
    rows = out.get("rows") or []
    if not isinstance(rows, list) or len(rows) != len(tx_items):
        raise RuntimeError("LLM output rows length mismatch")
    return rows

# ==== BANK STATEMENT INGEST ===================================================
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

    bank_name = _infer_bank_name(fields, result, source)
    statement_date = _infer_statement_date(fields, result, accounts)
    ccy = (currency or _infer_currency_from_result(result) or DEFAULT_CCY).upper()

    total_input_rows = sum(len(a.get("Transactions") or []) for a in accounts)
    log.info("ingest.bank_statement.shaped", extra={"extra": {
        "accounts": len(accounts), "model_tx_count": total_input_rows,
        "bank_name": bank_name, "statement_date": str(statement_date) if statement_date else None
    }})

    # ---- Stitch wrapped lines PER ACCOUNT & assign stable anchor line numbers
    stitched_accounts: List[Tuple[str, List[Dict[str, Any]], int, Dict[int,int]]] = []
    for acc in accounts:
        acct_no = (acc.get("AccountNumber") or {}).get("value")
        txs = acc.get("Transactions") or []
        stitched, appended = _stitch_wrapped_lines(txs)

        # assign stable anchor numbers now (1..N) after stitching
        anchor_line_map: Dict[int,int] = {}
        anchor_no = 0
        for idx, _t in enumerate(stitched):
            anchor_no += 1
            anchor_line_map[idx] = anchor_no

        stitched_accounts.append((acct_no, stitched, appended, anchor_line_map))
        if appended:
            log.info("ingest.bank_statement.stitch", extra={"extra": {
                "account": acct_no, "continuations_merged": appended,
                "original_rows": len(txs), "stitched_rows": len(stitched)
            }})

    # --- Build rows using LLM + DI fallbacks ----------------------------------
    stage_rows: List[Dict[str, Any]] = []
    reco_rows:  List[Dict[str, Any]] = []

    batch, batch_meta = [], []
    def _flush_batch():
        nonlocal batch, batch_meta, stage_rows, reco_rows
        if not batch: return
        ai_rows = _ai_map_rows(batch, statement_date, bank_name, ccy)

        # rolling dates per account
        last_tx_date: Dict[str, Optional[date]] = {}
        last_val_date: Dict[str, Optional[date]] = {}

        for i, ai in enumerate(ai_rows):
            acct_no = batch_meta[i]["acct_no"]
            raw_desc = batch_meta[i]["desc"]
            line_no = batch_meta[i]["anchor_no"]  # stable across batches

            # ---- dates: LLM -> DI -> rolling backfill
            txn_date = _as_date(ai.get("transaction_date"))
            if txn_date is None:
                t = batch_meta[i]["tx_obj"]
                txn_date = _as_date(_getv(t, "TransactionDate","PostingDate","PostDate","Date","BookingDate","ValueDate"))
            if txn_date is None:
                txn_date = last_tx_date.get(acct_no)

            val_date = _as_date(ai.get("value_date"))
            if val_date is None:
                t = batch_meta[i]["tx_obj"]
                val_date = _as_date(_getv(t, "ValueDate","ValDate","Value Date"))
            if val_date is None:
                val_date = txn_date or last_val_date.get(acct_no)

            # ---- amount: LLM -> DI fallback
            amt = _num_soft(ai.get("statement_amount"))
            if amt is None:
                amt = _signed_amount_from_tx(batch_meta[i]["tx_obj"])

            # if truly empty, skip (rare after stitch)
            if amt is None and (txn_date is None and not raw_desc):
                continue

            desc_composed = _compose_description(ai.get("description"), ai.get("transaction_text"))
            desc_final = desc_composed or raw_desc
            merch_norm = _norm_merchant(desc_final)
            ref = ai.get("reference") or ai.get("additional_transaction_info")

            fp = _fp([
                acct_no,
                statement_date.isoformat() if statement_date else "",
                line_no,
                f"{(amt or 0):.2f}",
                ccy,
                merch_norm or "",
                ref or "",
            ])

            # stage row
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

            # reco row
            reco_rows.append({
                "Bank Account No.": acct_no,
                "Bank Name": bank_name,
                "Statement Date": statement_date,
                "Reconciliation date": statement_date,
                "Statement Line No.": line_no,
                "Document No.": None,
                "Transaction Date": txn_date,
                "Description": desc_final,
                "Statement Amount": amt,
                "Difference": None,
                "Applied Amount": None,
                "Applied Entries": 0,
                "Value Date": val_date or txn_date,
                "Check No.": ai.get("check_no"),
                "Related-Party Name": ai.get("related_party_name"),
                "Additional Transaction Info": ai.get("additional_transaction_info"),
                "Statement Type": RECO_STATEMENT_TYPE,
                "Transaction Text": ai.get("transaction_text") or desc_final,
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
            raise HTTPException(status_code=502, detail="Document Intelligence error: {e}")

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
