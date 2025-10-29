# app/routers/tx_ingest.py
from __future__ import annotations

import os, re, json, hashlib, unicodedata
from typing import Optional, Dict, Any, List
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

# ==== OpenAI client (official SDK v1) for receipt normalization ===============
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    _openai_client_error = None
except Exception as e:
    _openai_client = None
    _openai_client_error = e

# ==== Minimal helpers (used by receipt path) ==================================
def _as_date(v: Any) -> Optional[date]:
    if v is None: return None
    if isinstance(v, date) and not isinstance(v, datetime): return v
    if isinstance(v, datetime): return v.date()
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
            try: return datetime.strptime(s, fmt).date()
            except Exception: pass
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
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

def _fp(parts: List[str]) -> str:
    base = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

# =============================================================================
# BANK STATEMENT INGEST (delegates shaping + AI to di_bank.process_*)
# =============================================================================
@router.post("/ingest/bank-statement")
async def ingest_bank_statement(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    currency: Optional[str] = Form(default=None),
):
    """
    Ingest a bank statement (file or URL), parse with Azure DI via di_bank,
    and insert rows into:
      - erp_ai.transactions_staging
      - erp_ai.bank_reconciliation_staging
    """
    log.info("ingest.bank_statement.start", extra={"extra": {
        "has_file": bool(file), "has_url": bool(url),
        "filename": getattr(file, "filename", None), "url": url
    }})

    # Allow JSON body {"url": "..."} as well.
    if not file and not url and request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            pass

    if file and url:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'url'.")
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")

    # Analyze + process (single call into di_bank)
    with log_duration("ingest.bank_statement.analyze_and_process"):
        try:
            if file:
                source = file.filename
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                proc = di_bank.analyze_and_process_bytes(
                    data,
                    getattr(file, "content_type", None),
                    source=source,
                    currency_override=currency,
                )
            else:
                proc = di_bank.analyze_and_process_url(
                    url,
                    currency_override=currency,
                )
        except HTTPException:
            raise
        except Exception as e:
            log.exception("ingest.bank_statement.di_error")
            raise HTTPException(status_code=502, detail=f"Document Intelligence error: {e}")

    stage_rows: List[Dict[str, Any]] = proc.get("stage_rows", [])
    reco_rows:  List[Dict[str, Any]] = proc.get("reco_rows", [])
    bank_name: Optional[str]          = proc.get("bank_name")
    statement_date: Optional[date]    = proc.get("statement_date")
    ccy: str                          = (proc.get("currency") or DEFAULT_CCY).upper()

    log.info("ingest.bank_statement.rows_ready", extra={"extra": {
        "stage_rows": len(stage_rows),
        "reco_rows": len(reco_rows),
        "bank_name": bank_name,
        "statement_date": str(statement_date) if statement_date else None,
        "currency": ccy
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
                    r.get("txn_date"),
                    r.get("amount"),
                    r.get("currency") or ccy,
                    r.get("merchant_raw"),
                    r.get("merchant_norm"),
                    r.get("reference"),
                    r.get("source_file"),
                    r.get("fingerprint"),
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
                    rr.get("Bank Account No."),
                    rr.get("Bank Name") or bank_name,
                    rr.get("Statement Date") or statement_date,
                    rr.get("Reconciliation date"),
                    rr.get("Statement Line No."),
                    rr.get("Document No."),
                    rr.get("Transaction Date"),
                    rr.get("Description"),
                    rr.get("Statement Amount"),
                    rr.get("Difference"),
                    rr.get("Applied Amount"),
                    rr.get("Applied Entries"),
                    rr.get("Value Date"),
                    rr.get("Check No."),
                    rr.get("Related-Party Name"),
                    rr.get("Additional Transaction Info"),
                    rr.get("Statement Type") or RECO_STATEMENT_TYPE,
                    rr.get("Transaction Text"),
                    rr.get("Related-Party Bank Acc. No."),
                    rr.get("Related-Party Address"),
                    rr.get("Related-Party City"),
                )
                if ok2 == 1:
                    reco_inserted += 1

    log.info("ingest.bank_statement.done", extra={"extra": {
        "inserted": inserted, "reco_inserted": reco_inserted, "total": len(stage_rows)
    }})
    return {"status": "ok", "inserted": inserted, "reco_inserted": reco_inserted, "total": len(stage_rows)}

# =============================================================================
# RECEIPT INGEST (LLM-normalized; items extraction handled inside di_receipt)
# =============================================================================
_LLM_RECEIPT_SYSTEM = """You normalize receipt fields into strict JSON.
- date -> 'YYYY-MM-DD' or null
- total -> number or null
- currency -> 3-letter code if present else null
- merchant -> cleaned merchant name (short) or null
Return: {"date": "...", "total": 0.0, "currency": "...", "merchant": "..."}"""

def _openai_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if _openai_client is None:
        raise RuntimeError(f"OpenAI client not available: {_openai_client_error}")
    resp = _openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt or "{}")
    except Exception as e:
        raise RuntimeError(f"LLM returned non-JSON: {e}: {txt[:400]}")

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

def _item_get_value(it: Dict[str, Any], key: str) -> Any:
    obj = it.get(key) or {}
    return obj.get("value")

@router.post("/ingest/receipt")
async def ingest_receipt(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    url:  Optional[str]       = Form(default=None),
    currency: Optional[str]   = Form(default=None),
):
    """
    Ingest a receipt (file or URL). Uses di_receipt for Azure DI parsing.
    Normalizes a few header fields with a tiny LLM (or deterministic fallback),
    and inserts into:
      - erp_ai.receipts_staging (header)
      - erp_ai.receipt_items_staging (per-line items, if present)

    Also logs each item line and returns a cleaned 'items' preview in the response.
    """
    log.info("ingest.receipt.start", extra={"extra": {
        "has_file": bool(file), "has_url": bool(url),
        "filename": getattr(file, "filename", None), "url": url
    }})

    # accept {"url": "..."} JSON too
    if not file and not url and request.headers.get("content-type","").startswith("application/json"):
        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            pass
    if file and url:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'url'.")
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")

    # analyze via DI
    source = None
    with log_duration("ingest.receipt.analyze"):
        try:
            if file:
                source = file.filename
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                result = di_receipt.analyze_receipt_bytes(data, getattr(file, "content_type", None))
            else:
                source = url
                result = di_receipt.analyze_receipt_url(url)
        except Exception as e:
            log.exception("ingest.receipt.di_error")
            raise HTTPException(status_code=502, detail=f"Document Intelligence error: {e}")

    shaped = di_receipt.receipt_to_dict(result)
    fields = shaped.get("fields", {}) or {}
    items  = shaped.get("items", []) or []
    ccy_default = (currency or DEFAULT_CCY).upper()

    ai = _ai_norm_receipt(fields, ccy_default)
    receipt_date = _as_date(ai.get("date"))
    total = _num_soft(ai.get("total"))
    ccy_final = (ai.get("currency") or ccy_default).upper()
    merchant = ai.get("merchant")
    merch_norm = _norm_merchant(merchant)

    if receipt_date is None or total is None:
        log.warning("ingest.receipt.missing_core_fields", extra={"extra": {
            "date_raw": ai.get("date"), "total_raw": ai.get("total")
        }})
        raise HTTPException(status_code=422, detail="Could not parse receipt date/total.")

    # header fingerprint (link key for items)
    fp = _fp([
        receipt_date.isoformat() if receipt_date else "",
        f"{(total or 0):.2f}",
        ccy_final,
        merch_norm or "",
        source or "",
    ])

    def _item_get_value(it: Dict[str, Any], key: str) -> Optional[Any]:
        v = it.get(key)
        if isinstance(v, dict):
            return v.get("value")
        return v

    header_inserted = 0
    items_inserted = 0
    items_view: List[Dict[str, Any]] = []

    with log_duration("ingest.receipt.insert_all"):
        async with db_conn() as conn:
            # receipts_staging insert (note the column names!)
            ok = await conn.fetchval(
                """
                insert into erp_ai.receipts_staging
                  (receipt_date, total_amount, currency, merchant_name, merchant_norm, source_file, fingerprint)
                values ($1,$2,$3,$4,$5,$6,$7)
                on conflict (fingerprint) do nothing
                returning 1;
                """,
                receipt_date, total, ccy_final, merchant, merch_norm, source, fp
            )
            if ok == 1:
                header_inserted = 1

            # items
            if items:
                line_no = 0
                for it in items:
                    desc        = _item_get_value(it, "Description")
                    qty         = _num_soft(_item_get_value(it, "Quantity"))
                    price       = _num_soft(_item_get_value(it, "Price"))
                    line_total  = _num_soft(_item_get_value(it, "TotalPrice"))
                    product_code= _item_get_value(it, "ProductCode")
                    line_date   = _as_date(_item_get_value(it, "Date"))
                    category    = _item_get_value(it, "Category")

                    # Skip an all-empty row
                    if not (desc or qty is not None or price is not None or line_total is not None
                            or product_code or line_date or category):
                        continue

                    line_no += 1

                    # Log each item (so it "prints" in your logs)
                    log.info("ingest.receipt.item", extra={"extra": {
                        "line_no": line_no,
                        "description": desc,
                        "quantity": qty,
                        "unit_price": price,
                        "line_total": line_total,
                        "product_code": product_code,
                        "date": str(line_date) if line_date else None,
                        "category": category
                    }})

                    items_view.append({
                        "line_no": line_no,
                        "description": desc,
                        "quantity": qty,
                        "unit_price": price,
                        "line_total": line_total,
                        "product_code": product_code,
                        "date": line_date.isoformat() if line_date else None,
                        "category": category,
                    })

                    item_fp = _fp([
                        fp,
                        str(line_no),
                        (desc or "").strip().upper(),
                        "" if qty is None else f"{qty:.4f}",
                        "" if price is None else f"{price:.4f}",
                        "" if line_total is None else f"{line_total:.4f}",
                        (product_code or "").strip().upper(),
                    ])

                    ok_item = await conn.fetchval(
                        """
                        insert into erp_ai.receipt_items_staging
                          (receipt_fingerprint, line_no, description, quantity, unit_price, line_total,
                           product_code, line_date, category, currency, source_file, item_fingerprint)
                        values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                        on conflict (item_fingerprint) do nothing
                        returning 1;
                        """,
                        fp, line_no, desc, qty, price, line_total,
                        product_code, line_date, category, ccy_final, source, item_fp
                    )
                    if ok_item == 1:
                        items_inserted += 1

    log.info("ingest.receipt.done", extra={"extra": {
        "header_inserted": header_inserted,
        "items_inserted": items_inserted,
        "date": str(receipt_date),
        "total": total,
        "currency": ccy_final,
        "merchant": merchant,
        "fingerprint": fp
    }})

    return {
        "status": "ok",
        "inserted_header": header_inserted,
        "inserted_items": items_inserted,
        "fingerprint": fp,
        "receipt": {
            "date": receipt_date.isoformat() if receipt_date else None,
            "total": total,
            "currency": ccy_final,
            "merchant": merchant,
        },
        "items": items_view,
    }
