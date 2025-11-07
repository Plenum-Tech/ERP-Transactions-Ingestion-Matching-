# app/routers/tx_bank_reco.py
from __future__ import annotations

import io, base64, traceback
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder

from app.services.bank_reco_evaluator import run_matcher, to_excel_bytes  # ensure both are exported

# Optional logger
try:
    from app.utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

router = APIRouter(prefix="/tx/bank-reco", tags=["bank-reconciliation"])

# ---- JSON sanitization helper ------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """
    Make result JSON-serializable: handles pandas/NumPy/decimal/datetime types.
    """
    return jsonable_encoder(
        obj,
        custom_encoder={
            pd.Timestamp: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            Decimal: float,
            # Pandas NaT -> None
            type(pd.NaT): lambda v: None,
        },
    )

@router.post("/match")
async def match_all_in_one(
    statement_file: UploadFile = File(..., description="Bank statement (CSV/XLS/XLSX)"),
    ledger_file: UploadFile = File(..., description="Bank ledger (CSV/XLS/XLSX)"),
    tol_abs: float = Query(2.0, description="Absolute amount tolerance (± units)"),
    tol_pct: float = Query(0.005, description="Relative tolerance (e.g., 0.005 = 0.5%)"),
    llm_summary: bool = Query(False, description="Attach GPT-written summary if API key configured"),
    export: str = Query(
        "json",
        description="Return format: 'json', 'xlsx', or 'both' (JSON + base64 Excel)",
        regex="^(json|xlsx|both)$",
    ),
    filename: str | None = Query(None, description="Optional override for XLSX download filename"),
):
    """
    One-stop reconciliation: parse → normalize → 1:1 + composites → recommendations → (optional) LLM summary.

    - ?export=json  -> returns JSON (default)
    - ?export=xlsx  -> returns an Excel workbook as a file download
    - ?export=both  -> returns JSON and includes the Excel workbook as base64 in the payload
    """
    try:
        # Validate file types
        for f, kind in [(statement_file, "statement"), (ledger_file, "ledger")]:
            if not f or not f.filename:
                raise HTTPException(status_code=400, detail=f"Missing {kind} file")
            name = (f.filename or "").lower()
            if not (name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls")):
                raise HTTPException(
                    status_code=400,
                    detail=f"{kind} must be CSV or Excel (.csv/.xlsx/.xls). Got: {f.filename}",
                )

        st_bytes = await statement_file.read()
        lg_bytes = await ledger_file.read()

        log.info(
            "bank-reco.match | statement=%s(%s bytes) ledger=%s(%s bytes) tol_abs=%.4f tol_pct=%.6f llm=%s export=%s",
            statement_file.filename, len(st_bytes),
            ledger_file.filename, len(lg_bytes),
            tol_abs, tol_pct, llm_summary, export
        )

        # Run reconciliation
        result: Dict[str, Any] = run_matcher(
            statement_bytes=st_bytes,
            statement_filename=statement_file.filename or "statement.csv",
            ledger_bytes=lg_bytes,
            ledger_filename=ledger_file.filename or "ledger.csv",
            tol_abs=tol_abs,
            tol_pct=tol_pct,
            enable_llm_summary=llm_summary,
        )

        # === export=xlsx -> return file download ===
        if export == "xlsx":
            try:
                xlsx_bytes = to_excel_bytes(result)
            except Exception as ex:
                log.error("Excel build failed: %s", ex)
                log.debug(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to build Excel: {ex}")

            safe_name = filename or f"bank_reco_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return StreamingResponse(
                io.BytesIO(xlsx_bytes),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{safe_name}"'}
            )

        # === export=both -> return JSON + base64 Excel ===
        if export == "both":
            payload = {"ok": True, **result}
            try:
                xlsx_bytes = to_excel_bytes(result)
                safe_name = filename or f"bank_reco_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
                payload["_excel"] = {
                    "filename": safe_name,
                    "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "bytes_b64": base64.b64encode(xlsx_bytes).decode("ascii"),
                }
            except Exception as ex:
                log.error("Excel build failed for export=both: %s", ex)
                log.debug(traceback.format_exc())
                payload["_excel_error"] = f"Failed to build Excel: {ex}"

            return JSONResponse(content=_json_safe(payload))

        # === default -> JSON only ===
        return JSONResponse(content=_json_safe({"ok": True, **result}))

    except HTTPException:
        raise
    except Exception as e:
        log.error("bank-reco.match failed: %s", e)
        log.debug(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Match failed: {e}")
