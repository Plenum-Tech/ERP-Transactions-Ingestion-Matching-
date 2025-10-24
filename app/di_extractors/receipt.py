# app/di_extractors/receipt.py
"""
Receipt extractor using Azure AI Document Intelligence (v4) with table/line fallbacks.

- analyze_receipt_url() / analyze_receipt_bytes() -> AnalyzeResult
- receipt_to_dict() -> dict with fields/items (value + confidence)

Env:
  AZ_DI_ENDPOINT_PRIMARY (or AZ_DI_ENDPOINT)
  AZ_DI_KEY
  ALLOW_DEGRADED_FALLBACK=true|false   (optional; default true)
  MODEL_RECEIPT_FALLBACK=prebuilt-document
"""

import os, re
from typing import Any, Dict, List, Optional, Iterable, Tuple
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from app.utils.logger import get_logger

load_dotenv()
_log = get_logger(__name__)

ENDPOINT = os.getenv("AZ_DI_ENDPOINT_PRIMARY", "https://erpdocintelligence.cognitiveservices.azure.com/") or os.getenv("AZ_DI_ENDPOINT")
KEY      = os.getenv("AZ_DI_KEY", "6J510LpGX7uJgC22e6nzazFrwN1uf59dBT9q4XwqZzAeHBIfm5OpJQQJ99BIACYeBjFXJ3w3AAALACOGOE5h")

MODEL_ID = "prebuilt-receipt"
ALLOW_DEGRADED_FALLBACK = os.getenv("ALLOW_DEGRADED_FALLBACK", "true").lower() == "true"
MODEL_ID_FALLBACK       = os.getenv("MODEL_RECEIPT_FALLBACK", "prebuilt-document")

# ----------------------------- client/analyze -----------------------------
def _client() -> DocumentIntelligenceClient:
    if not ENDPOINT or not KEY:
        raise RuntimeError("Set AZ_DI_ENDPOINT and AZ_DI_KEY in your environment/.env")
    _log.debug("di.receipt.client.init", extra={"extra": {"endpoint": ENDPOINT, "model": MODEL_ID}})
    return DocumentIntelligenceClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

def analyze_receipt_url(doc_url: str):
    c = _client()
    try:
        _log.info("di.receipt.analyze_url", extra={"extra": {"model": MODEL_ID, "url": doc_url}})
        return c.begin_analyze_document(MODEL_ID, AnalyzeDocumentRequest(url_source=doc_url)).result()
    except (ResourceNotFoundError, HttpResponseError) as e:
        _log.warning("di.receipt.model_error_url", extra={"extra": {"model": MODEL_ID, "status": getattr(e, "status_code", None), "message": getattr(e, "message", str(e))}})
        if ALLOW_DEGRADED_FALLBACK:
            _log.info("di.receipt.fallback_model_url", extra={"extra": {"from": MODEL_ID, "to": MODEL_ID_FALLBACK, "url": doc_url}})
            return c.begin_analyze_document(MODEL_ID_FALLBACK, AnalyzeDocumentRequest(url_source=doc_url)).result()
        raise

def analyze_receipt_bytes(content: bytes, content_type: Optional[str] = None):
    if not content:
        raise ValueError("Uploaded file is empty.")
    c = _client()
    try:
        _log.info("di.receipt.analyze_bytes", extra={"extra": {"model": MODEL_ID, "content_type": content_type or "application/octet-stream", "size": len(content or b'')}})
        return c.begin_analyze_document(MODEL_ID, content, content_type=content_type or "application/octet-stream").result()
    except (ResourceNotFoundError, HttpResponseError) as e:
        _log.warning("di.receipt.model_error_bytes", extra={"extra": {"model": MODEL_ID, "status": getattr(e, "status_code", None), "message": getattr(e, "message", str(e))}})
        if ALLOW_DEGRADED_FALLBACK:
            _log.info("di.receipt.fallback_model_bytes", extra={"extra": {"from": MODEL_ID, "to": MODEL_ID_FALLBACK}})
            return c.begin_analyze_document(MODEL_ID_FALLBACK, content, content_type=content_type or "application/octet-stream").result()
        raise

# ----------------------------- v4-safe value helpers -----------------------------
def _kv(value_obj: Any) -> Dict[str, Any]:
    """Return {'value': <typed>, 'confidence': <float>} handling v4 and legacy accessors."""
    if value_obj is None:
        return {"value": None, "confidence": None}
    conf = getattr(value_obj, "confidence", None)
    val = getattr(value_obj, "value", None)
    if val is None:
        for attr in ("value_string","value_number","value_date","value_time","value_address","value_currency"):
            if hasattr(value_obj, attr):
                v = getattr(value_obj, attr)
                if hasattr(v, "amount"):
                    v = v.amount
                val = v
                break
    if val is None and hasattr(value_obj, "content"):
        val = getattr(value_obj, "content")
    if hasattr(val, "amount"):
        val = val.amount
    return {"value": val, "confidence": conf}

def _get_array(field: Any) -> Iterable[Any]:
    if field is None:
        return []
    arr = getattr(field, "value", None)
    if isinstance(arr, list):
        return arr
    arr = getattr(field, "value_array", None)
    if isinstance(arr, list):
        return arr
    return []

def _get_props(field: Any) -> Dict[str, Any]:
    if field is None:
        return {}
    props = getattr(field, "properties", None)
    if isinstance(props, dict):
        return props
    props = getattr(field, "value_object", None)
    if isinstance(props, dict):
        return props
    return {}

# ----------------------------- fallbacks: tables + lines -----------------------------
_CCY_SYMBOLS = {
    "£": "GBP", "€": "EUR", "$": "USD", "¥": "JPY", "₹": "INR", "₽": "RUB", "₩": "KRW",
    "د.إ": "AED", "﷼": "SAR", "₺": "TRY", "₫": "VND", "฿": "THB", "₴": "UAH"
}
_CCY_CODES = {"AED","USD","GBP","EUR","INR","SAR","QAR","KWD","OMR","BHD","CAD","AUD","NZD","JPY","CNY","SGD","HKD"}

_TOTAL_HINTS = ("total", "amount due", "grand total", "balance due", "total due", "total amount")
_DATE_RX = [
    r"\b\d{4}-\d{2}-\d{2}\b",           # 2024-11-06
    r"\b\d{2}/\d{2}/\d{4}\b",           # 06/11/2024
    r"\b\d{2}-\d{2}-\d{4}\b",           # 06-11-2024
    r"\b\d{4}/\d{2}/\d{2}\b",           # 2024/11/06
    r"\b\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\b",  # 6 Nov 2024
]

def _parse_amount_with_currency(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    # currency code in text?
    for code in _CCY_CODES:
        if re.search(rf"\b{code}\b", text, re.I):
            amt = _extract_number(text)
            return (amt, code if amt is not None else None)
    # symbol?
    for sym, code in _CCY_SYMBOLS.items():
        if sym in text:
            amt = _extract_number(text)
            return (amt, code if amt is not None else None)
    # just a number (no currency)
    return (_extract_number(text), None)

def _extract_number(text: str) -> Optional[float]:
    m = re.search(r"([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def _lines(result):
    pages = getattr(result, "pages", []) or []
    for p in pages:
        for ln in getattr(p, "lines", []) or []:
            txt = getattr(ln, "content", "") or ""
            if txt.strip():
                yield txt.strip()

def _tables(result):
    return getattr(result, "tables", []) or []

def _grid_from_table(tbl) -> List[List[str]]:
    cells = getattr(tbl, "cells", []) or []
    max_r = max((c.row_index + getattr(c, "row_span", 1) - 1 for c in cells), default=-1)
    max_c = max((c.column_index + getattr(c, "column_span", 1) - 1 for c in cells), default=-1)
    grid = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for c in cells:
        r, cidx = c.row_index, c.column_index
        txt = (getattr(c, "content", "") or "").strip()
        grid[r][cidx] = txt
    return grid

def _fallback_totals_from_tables(result) -> Tuple[Optional[float], Optional[str]]:
    parsed = 0
    for tbl in _tables(result):
        grid = _grid_from_table(tbl)
        for row in grid[-8:]:  # last few rows often contain totals
            joined = " ".join(row).strip()
            if not joined:
                continue
            if any(h in joined.lower() for h in _TOTAL_HINTS):
                amt, ccy = _parse_amount_with_currency(joined)
                if amt is not None:
                    parsed += 1
                    _log.info("di.receipt.table.total_found", extra={"extra": {"row": joined, "amount": amt, "currency": ccy}})
                    return amt, ccy
    if parsed == 0:
        _log.debug("di.receipt.table.no_total")
    return None, None

def _fallback_date_from_lines(result) -> Optional[str]:
    # prefer lines with explicit 'date' keyword
    for ln in _lines(result):
        if "date" in ln.lower():
            for rx in _DATE_RX:
                m = re.search(rx, ln)
                if m:
                    _log.info("di.receipt.lines.date_found", extra={"extra": {"line": ln, "date": m.group(0)}})
                    return m.group(0)
    # else: first date-looking token anywhere
    for ln in _lines(result):
        for rx in _DATE_RX:
            m = re.search(rx, ln)
            if m:
                _log.info("di.receipt.lines.date_found_loose", extra={"extra": {"line": ln, "date": m.group(0)}})
                return m.group(0)
    _log.debug("di.receipt.lines.no_date")
    return None

def _fallback_currency_from_lines(result) -> Optional[str]:
    for ln in _lines(result):
        # code
        for code in _CCY_CODES:
            if re.search(rf"\b{code}\b", ln, re.I):
                _log.info("di.receipt.lines.currency_code_found", extra={"extra": {"line": ln, "currency": code}})
                return code
        # symbol
        for sym, code in _CCY_SYMBOLS.items():
            if sym in ln:
                _log.info("di.receipt.lines.currency_symbol_found", extra={"extra": {"line": ln, "currency": code}})
                return code
    _log.debug("di.receipt.lines.no_currency")
    return None

def _fallback_merchant_from_lines(result) -> Optional[str]:
    stop = {"receipt", "invoice", "tax", "total", "amount", "date", "order", "no.", "number"}
    for ln in _lines(result):
        s = re.sub(r"[^A-Za-z0-9 .,&@'-]", " ", ln).strip()
        if not s or len(s) < 3:
            continue
        low = s.lower()
        if any(w in low for w in stop):
            continue
        # heuristics: words + not just numbers/symbols
        if re.search(r"[A-Za-z]{2,}", s):
            _log.info("di.receipt.lines.merchant_found", extra={"extra": {"line": ln, "merchant": s[:120]}})
            return s[:120]
    _log.debug("di.receipt.lines.no_merchant")
    return None

def _fallback_total_from_lines(result) -> Tuple[Optional[float], Optional[str]]:
    for ln in _lines(result):
        if any(h in ln.lower() for h in _TOTAL_HINTS):
            amt, ccy = _parse_amount_with_currency(ln)
            if amt is not None:
                _log.info("di.receipt.lines.total_found", extra={"extra": {"line": ln, "amount": amt, "currency": ccy}})
                return amt, ccy
    _log.debug("di.receipt.lines.no_total")
    return None, None

# ----------------------------- shaping -----------------------------
_HEADER_FIELDS: List[str] = [
    "MerchantName",
    "MerchantPhoneNumber",
    "MerchantAddress",
    "MerchantAddressRecipient",
    "TransactionDate",
    "TransactionTime",
    "Total",
    "Subtotal",
    "TotalTax",
    "Tip",
    "Currency",
    "PaymentMethod",
    "CardNumber",
    "ReceiptType",
    "CustomerName",
    "OrderNumber",
    "InvoiceId",
]
_ITEM_FIELDS: List[str] = ["Description", "Quantity", "Price", "TotalPrice", "ProductCode", "Date", "Category"]

def receipt_to_dict(result) -> Dict[str, Any]:
    out: Dict[str, Any] = {"doc_type": None, "fields": {}, "items": []}
    if not result or not getattr(result, "documents", None):
        _log.warning("di.receipt.no_document")
        return out

    doc = result.documents[0]
    f   = getattr(doc, "fields", {}) or {}
    out["doc_type"] = getattr(doc, "doc_type", None)

    # 1) direct fields (from prebuilt-receipt)
    for name in _HEADER_FIELDS:
        if name in f:
            out["fields"][name] = _kv(f.get(name))

    # 2) items (if model provided)
    items = f.get("Items")
    for it in _get_array(items):
        props = _get_props(it)
        out["items"].append({k: _kv(props.get(k)) for k in _ITEM_FIELDS})

    # quick presence log
    def _getv(key: str):
        obj = out["fields"].get(key)
        return None if not obj else obj.get("value")

    _log.info("di.receipt.model_output", extra={"extra": {
        "has_total": _getv("Total") is not None,
        "has_date": _getv("TransactionDate") is not None,
        "has_currency": _getv("Currency") is not None,
        "has_merchant": _getv("MerchantName") is not None,
        "items": len(out["items"])
    }})

    # 3) fallbacks for key headers if missing or empty
    # total
    if _getv("Total") is None:
        t_amt, t_ccy = _fallback_totals_from_tables(result)
        if t_amt is None:
            t_amt, t_ccy = _fallback_total_from_lines(result)
        if t_amt is not None:
            out["fields"]["Total"] = {"value": t_amt, "confidence": None}
            _log.info("di.receipt.fallback.total_applied", extra={"extra": {"amount": t_amt}})
        if t_ccy and _getv("Currency") is None:
            out["fields"]["Currency"] = {"value": t_ccy, "confidence": None}
            _log.info("di.receipt.fallback.currency_applied", extra={"extra": {"currency": t_ccy}})

    # date
    if _getv("TransactionDate") is None:
        d = _fallback_date_from_lines(result)
        if d:
            out["fields"]["TransactionDate"] = {"value": d, "confidence": None}
            _log.info("di.receipt.fallback.date_applied", extra={"extra": {"date": d}})

    # merchant
    if _getv("MerchantName") is None:
        m = _fallback_merchant_from_lines(result)
        if m:
            out["fields"]["MerchantName"] = {"value": m, "confidence": None}
            _log.info("di.receipt.fallback.merchant_applied", extra={"extra": {"merchant": m}})

    # currency (if still missing)
    if _getv("Currency") is None:
        ccy = _fallback_currency_from_lines(result)
        if ccy:
            out["fields"]["Currency"] = {"value": ccy, "confidence": None}
            _log.info("di.receipt.fallback.currency_applied", extra={"extra": {"currency": ccy}})

    # final summary
    _log.info("di.receipt.shaped", extra={"extra": {
        "has_total": out["fields"].get("Total", {}).get("value") is not None,
        "has_date": out["fields"].get("TransactionDate", {}).get("value") is not None,
        "has_currency": out["fields"].get("Currency", {}).get("value") is not None,
        "has_merchant": out["fields"].get("MerchantName", {}).get("value") is not None,
        "items": len(out["items"])
    }})
    return out

