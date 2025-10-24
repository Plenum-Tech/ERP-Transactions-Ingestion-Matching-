# app/di_extractors/bank_statement.py
"""
Bank Statement extractor using Azure AI Document Intelligence (v4) with table fallback.

- analyze_bank_stmt_url() / analyze_bank_stmt_bytes() -> AnalyzeResult
- bank_stmt_to_dict() -> dict with fields/accounts/transactions (value + confidence)
- print_bank_stmt() -> console output (optional)

Env:
  AZ_DI_ENDPOINT_PRIMARY (or AZ_DI_ENDPOINT)
  AZ_DI_KEY
  MODEL_BANKSTMT                (default: prebuilt-bankStatement.us)
  ALLOW_DEGRADED_FALLBACK=true  (optional)
  MODEL_BANKSTMT_FALLBACK       (default: prebuilt-document)
"""

import os, re
from typing import Any, Dict, List, Optional, Iterable, Set
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from app.utils.logger import get_logger

load_dotenv()
_log = get_logger(__name__)

ENDPOINT = os.getenv("AZ_DI_ENDPOINT_PRIMARY", "https://erpdocintelligence.cognitiveservices.azure.com/") or os.getenv("AZ_DI_ENDPOINT")
KEY      = os.getenv("AZ_DI_KEY", "6J510LpGX7uJgC22e6nzazFrwN1uf59dBT9q4XwqZzAeHBIfm5OpJQQJ99BIACYeBjFXJ3w3AAALACOGOE5h")

MODEL_ID                = os.getenv("MODEL_BANKSTMT", "prebuilt-bankStatement.us")
ALLOW_DEGRADED_FALLBACK = os.getenv("ALLOW_DEGRADED_FALLBACK", "true").lower() == "true"
MODEL_ID_FALLBACK       = os.getenv("MODEL_BANKSTMT_FALLBACK", "prebuilt-document")

# -----------------------------
# Client helpers
# -----------------------------
def _client() -> DocumentIntelligenceClient:
    if not ENDPOINT or not KEY:
        raise RuntimeError("Set AZ_DI_ENDPOINT_PRIMARY (or AZ_DI_ENDPOINT) and AZ_DI_KEY in your environment/.env")
    _log.debug("di.client.init", extra={"extra": {"endpoint": ENDPOINT, "model": MODEL_ID}})
    return DocumentIntelligenceClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# -----------------------------
# Analyze helpers (URL / bytes with optional degraded fallback)
# -----------------------------
def _analyze_url(model_id: str, url: str):
    c = _client()
    _log.info("di.bank_stmt.analyze_url", extra={"extra": {"model": model_id, "url": url}})
    return c.begin_analyze_document(model_id, AnalyzeDocumentRequest(url_source=url)).result()

def _analyze_bytes(model_id: str, content: bytes, content_type: Optional[str]):
    c = _client()
    _log.info("di.bank_stmt.analyze_bytes", extra={"extra": {"model": model_id, "content_type": content_type or "application/octet-stream", "size": len(content or b'')}})
    return c.begin_analyze_document(model_id, content, content_type=content_type or "application/octet-stream").result()

def analyze_bank_stmt_url(doc_url: str):
    try:
        return _analyze_url(MODEL_ID, doc_url)
    except (ResourceNotFoundError, HttpResponseError) as e:
        _log.warning("di.bank_stmt.model_error_url", extra={"extra": {"model": MODEL_ID, "status": getattr(e, "status_code", None), "message": getattr(e, "message", str(e))}})
        if ALLOW_DEGRADED_FALLBACK and (getattr(e, "status_code", None) == 404 or "model" in (getattr(e, "message", "") or "").lower()):
            _log.info("di.bank_stmt.fallback_model_url", extra={"extra": {"from": MODEL_ID, "to": MODEL_ID_FALLBACK}})
            return _analyze_url(MODEL_ID_FALLBACK, doc_url)
        raise

def analyze_bank_stmt_bytes(content: bytes, content_type: Optional[str] = None):
    if not content:
        raise ValueError("Uploaded file is empty.")
    try:
        return _analyze_bytes(MODEL_ID, content, content_type)
    except (ResourceNotFoundError, HttpResponseError) as e:
        _log.warning("di.bank_stmt.model_error_bytes", extra={"extra": {"model": MODEL_ID, "status": getattr(e, "status_code", None), "message": getattr(e, "message", str(e))}})
        if ALLOW_DEGRADED_FALLBACK and (getattr(e, "status_code", None) == 404 or "model" in (getattr(e, "message", "") or "").lower()):
            _log.info("di.bank_stmt.fallback_model_bytes", extra={"extra": {"from": MODEL_ID, "to": MODEL_ID_FALLBACK}})
            return _analyze_bytes(MODEL_ID_FALLBACK, content, content_type)
        raise

# -----------------------------
# Safe value helpers (v4 + legacy)
# -----------------------------
def _kv(value_obj: Any) -> Dict[str, Any]:
    """Return {'value': <typed>, 'confidence': <float>} handling v4 and legacy accessors."""
    if value_obj is None:
        return {"value": None, "confidence": None}

    conf = getattr(value_obj, "confidence", None)

    # v4 preferred
    val = getattr(value_obj, "value", None)

    # legacy convenience props
    if val is None:
        for attr in ("value_number", "value_string", "value_date", "value_time", "value_address", "value_currency"):
            if hasattr(value_obj, attr):
                v = getattr(value_obj, attr)
                if hasattr(v, "amount"):
                    v = v.amount
                val = v
                break

    # final fallback to content
    if val is None and hasattr(value_obj, "content"):
        val = getattr(value_obj, "content")

    # unwrap currency-like objects that still have .amount
    if hasattr(val, "amount"):
        val = getattr(val, "amount", val)

    return {"value": val, "confidence": conf}

def _get_array(field: Any) -> Iterable[Any]:
    """Return iterable from .value (v4) or .value_array (legacy)."""
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
    """Return dict-like props from .properties (v4) or .value_object (legacy)."""
    if field is None:
        return {}
    props = getattr(field, "properties", None)
    if isinstance(props, dict):
        return props
    props = getattr(field, "value_object", None)
    if isinstance(props, dict):
        return props
    return {}

def _print_kv(label: str, obj: Any):
    k = _kv(obj)
    if k["value"] is not None:
        print(f"{label}: {k['value']} has confidence: {k['confidence']}")

# -----------------------------
# Table fallback helpers
# -----------------------------
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

def _find_header_indices(row: List[str]) -> Dict[str, int]:
    """
    Map header names we care about to column indices, tolerating variants:
    Date / Description / Paid In / Paid Out / Amount / Type(Credit/Debit)
    """
    idx = {}
    for i, cell in enumerate(row):
        s = cell.strip().lower()
        if not s:
            continue
        if "date" in s and "update" not in s:
            idx.setdefault("date", i)
        if "description" in s or "details" in s or "narration" in s:
            idx.setdefault("desc", i)
        if "paid in" in s or "credit" in s or "deposit" in s or "amount in" in s:
            idx.setdefault("in", i)
        if "paid out" in s or "debit" in s or "withdraw" in s or "amount out" in s or s == "payment":
            idx.setdefault("out", i)
        if s == "amount" or "amount" in s:
            idx.setdefault("amount", i)
        if "type" in s or "credit/debit" in s or "cr/dr" in s:
            idx.setdefault("type", i)
        if "balance" in s:
            idx.setdefault("balance", i)
        if "reference" in s or "ref" in s or "rrn" in s:
            idx.setdefault("ref", i)
    return idx

def _parse_money(s: str) -> Optional[float]:
    if not s:
        return None
    neg = "(" in s and ")" in s
    s2 = re.sub(r"[^\d.\-]", "", s)
    if s2 in ("", ".", "-", "-."):
        return None
    try:
        v = float(s2)
        return -abs(v) if neg and v > 0 else v
    except Exception:
        return None

def _transactions_from_tables(result) -> List[Dict[str, Dict[str, Any]]]:
    """
    Build transactions from detected tables, returning the same shape as model Transactions[]:
    [{ "Date": {...}, "Description": {...}, "DepositAmount": {...}, "WithdrawalAmount": {...},
       "Amount": {...}, "Type": {...}, "Balance": {...}, "Reference": {...} }, ...]
    """
    out: List[Dict[str, Dict[str, Any]]] = []
    table_idx = -1
    for tbl in _tables(result):
        table_idx += 1
        grid = _grid_from_table(tbl)
        if not grid:
            _log.debug("di.bank_stmt.table.empty", extra={"extra": {"table_index": table_idx}})
            continue

        header_row_idx = None
        header_map: Dict[str, int] = {}
        for r, row in enumerate(grid[:6]):
            header_map = _find_header_indices(row)
            if "date" in header_map and "desc" in header_map and any(k in header_map for k in ("in","out","amount")):
                header_row_idx = r
                break
        if header_row_idx is None:
            _log.debug("di.bank_stmt.table.no_headers", extra={"extra": {"table_index": table_idx}})
            continue

        _log.debug("di.bank_stmt.table.headers", extra={"extra": {"table_index": table_idx, "headers": header_map}})
        dcol  = header_map.get("date")
        xcol  = header_map.get("desc")
        incol = header_map.get("in")
        ocol  = header_map.get("out")
        amcol = header_map.get("amount")
        tpcol = header_map.get("type")
        bcol  = header_map.get("balance")
        rcol  = header_map.get("ref")

        before = len(out)
        for row in grid[header_row_idx + 1:]:
            date_raw = row[dcol] if dcol is not None and dcol < len(row) else ""
            desc     = row[xcol] if xcol is not None and xcol < len(row) else ""

            paid_in  = row[incol] if incol is not None and incol < len(row) else ""
            paid_out = row[ocol] if ocol is not None and ocol < len(row) else ""
            amount   = row[amcol] if amcol is not None and amcol < len(row) else ""
            type_cd  = row[tpcol] if tpcol is not None and tpcol < len(row) else ""
            balance  = row[bcol] if bcol is not None and bcol < len(row) else ""
            ref      = row[rcol] if rcol is not None and rcol < len(row) else ""

            if not (date_raw or desc or paid_in or paid_out or amount):
                continue
            if desc.strip().lower().startswith(("balance", "subtotal", "opening", "closing")):
                continue

            rec: Dict[str, Dict[str, Any]] = {
                "Date":             {"value": date_raw, "confidence": None},
                "Description":      {"value": desc,     "confidence": None},
                "DepositAmount":    {"value": _parse_money(paid_in),  "confidence": None},
                "WithdrawalAmount": {"value": _parse_money(paid_out), "confidence": None},
            }

            # For single Amount column, keep raw signed value in Amount field
            if amount:
                rec["Amount"] = {"value": _parse_money(amount), "confidence": None}
            if type_cd:
                rec["Type"] = {"value": type_cd, "confidence": None}
            if balance:
                rec["Balance"] = {"value": _parse_money(balance), "confidence": None}
            if ref:
                rec["Reference"] = {"value": ref, "confidence": None}

            out.append(rec)
        _log.info("di.bank_stmt.table.rows_parsed", extra={"extra": {"table_index": table_idx, "parsed_rows": len(out) - before}})
    return out

# -----------------------------
# Shaping (with table fallback)
# -----------------------------
# keep a whitelist of extra tx fields we want to preserve from the model
_EXTRA_TX_FIELDS: Set[str] = {
    "Amount", "TransactionAmount", "TransactionType", "Type", "CreditDebit",
    "Balance", "AvailableBalance", "RunningBalance",
    "Reference", "Ref", "RRN", "CheckNumber", "ChequeNumber",
    "Merchant", "MerchantName", "CardNumber", "Currency", "ExchangeRate"
}

def bank_stmt_to_dict(result) -> Dict[str, Any]:
    """
    Geometry-free JSON:
    {
      "fields": { <name>: {"value": ..., "confidence": ...}, ... },
      "accounts": [
        {
          "AccountNumber": {...},
          "AccountType": {...},
          "BeginningBalance": {...},
          "EndingBalance": {...},
          "TotalServiceFees": {...},
          "Transactions": [ {...}, ... ]
        }
      ]
    }
    """
    out: Dict[str, Any] = {"fields": {}, "accounts": []}
    if not result or not getattr(result, "documents", None):
        _log.warning("di.bank_stmt.no_document")
        return out

    doc = result.documents[0]
    f   = getattr(doc, "fields", {}) or {}

    # Header-level fields
    top_names = [
        "AccountHolderName", "AccountHolderAddress",
        "BankName", "BankAddress",
        "StatementStartDate", "StatementEndDate",
    ]
    out["fields"] = {name: _kv(f.get(name)) for name in top_names}

    # Accounts (preferred: model output)
    total_model_txs = 0
    accounts = f.get("Accounts")
    first_tx_keys: Set[str] = set()
    if accounts:
        for acc in _get_array(accounts):
            props = _get_props(acc)
            txs: List[Dict[str, Any]] = []
            for t in _get_array(props.get("Transactions")):
                tprops = _get_props(t)
                # always keep these
                rec = {
                    "Date":              _kv(tprops.get("Date")),
                    "Description":       _kv(tprops.get("Description")),
                    "DepositAmount":     _kv(tprops.get("DepositAmount")),
                    "WithdrawalAmount":  _kv(tprops.get("WithdrawalAmount")),
                }
                # also copy any extra known fields if present
                present_keys = []
                for k in list(_EXTRA_TX_FIELDS):
                    if k in tprops:
                        rec[k] = _kv(tprops.get(k))
                        present_keys.append(k)
                if not first_tx_keys and present_keys:
                    first_tx_keys = set(present_keys)
                txs.append(rec)
            total_model_txs += len(txs)
            out["accounts"].append({
                "AccountNumber":     _kv(props.get("AccountNumber")),
                "AccountType":       _kv(props.get("AccountType")),
                "BeginningBalance":  _kv(props.get("BeginningBalance")),
                "EndingBalance":     _kv(props.get("EndingBalance")),
                "TotalServiceFees":  _kv(props.get("TotalServiceFees")),
                "Transactions":      txs,
            })

    _log.info("di.bank_stmt.model_output", extra={"extra": {
        "accounts": len(out["accounts"]),
        "txs_model": total_model_txs,
        "sample_extra_tx_fields": sorted(list(first_tx_keys))[:8] if first_tx_keys else []
    }})

    # Fallback: if model didn't produce any tx rows, parse tables
    if total_model_txs == 0:
        txs = _transactions_from_tables(result)
        _log.info("di.bank_stmt.table_fallback", extra={"extra": {"txs_from_tables": len(txs)}})
        if txs:
            out["accounts"].append({
                "AccountNumber":     {"value": None, "confidence": None},
                "AccountType":       {"value": None, "confidence": None},
                "BeginningBalance":  {"value": None, "confidence": None},
                "EndingBalance":     {"value": None, "confidence": None},
                "TotalServiceFees":  {"value": None, "confidence": None},
                "Transactions":      txs,
            })

    # final summary
    total_all_txs = sum(len(a.get("Transactions") or []) for a in out.get("accounts", []))
    _log.info("di.bank_stmt.shaped", extra={"extra": {"accounts": len(out.get("accounts", [])), "txs_total": total_all_txs}})
    return out

# -----------------------------
# Printing (sample-like)
# -----------------------------
def print_bank_stmt(result) -> None:
    if not result or not getattr(result, "documents", None):
        print("No bank statement detected.")
        return

    d = bank_stmt_to_dict(result)
    print("--------Recognizing statement--------")
    # header fields
    for k, v in d["fields"].items():
        if v["value"] is not None:
            print(f"{k}: {v['value']} has confidence: {v['confidence']}")

    # accounts + transactions
    for i, acc in enumerate(d["accounts"]):
        print(f"...Account #{i + 1}")
        for name in ["AccountNumber","AccountType","BeginningBalance","EndingBalance","TotalServiceFees"]:
            v = acc.get(name, {"value": None})
            if v and v["value"] is not None:
                print(f"......{name}: {v['value']} has confidence: {v.get('confidence')}")
        if acc.get("Transactions"):
            print("......Transactions:")
            for j, t in enumerate(acc["Transactions"]):
                print(f".........Transaction #{j + 1}")
                for tn in ["Date","Description","DepositAmount","WithdrawalAmount","Amount","Type","Balance","Reference"]:
                    tv = t.get(tn, {"value": None})
                    if tv and tv["value"] is not None:
                        print(f"............{tn}: {tv['value']} has confidence: {tv.get('confidence')}")
    print("--------------------------------------")
