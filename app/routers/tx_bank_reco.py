# app/routers/tx_bank_reco.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import BytesIO

from app.services.bank_reco_evaluator import (
    build_inputs_from_statement_and_ledger,
    evaluate_llm,
)

router = APIRouter(prefix="/tx/bank-reco", tags=["bank-reconciliation"])

def _read_any(f: UploadFile) -> pd.DataFrame:
    try:
        name = (f.filename or "").lower()
        data = f.file.read()
        if name.endswith(".csv"):
            return pd.read_csv(BytesIO(data))
        if name.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(BytesIO(data))
            # pick first non-empty sheet
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    return df
            return xl.parse(xl.sheet_names[0])
        raise HTTPException(400, "Only .csv/.xlsx/.xls supported.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

@router.post("/match/files")
async def match_bank_statement_files(
    statement_file: UploadFile = File(..., description="Bank statement lines (.csv/.xlsx)"),
    ledger_file:    UploadFile = File(..., description="General Ledger / BALE (.csv/.xlsx)"),
    filter_zero_diff: bool = True,
):
    """
    Upload two spreadsheets (statement + ledger). Returns auto/review/no_match decisions
    with selected_ledger_id (if AUTO) and reasons.
    """
    try:
        statement_df = _read_any(statement_file)
        ledger_df    = _read_any(ledger_file)

        st_norm, lg_norm = build_inputs_from_statement_and_ledger(
            statement_df, ledger_df, filter_zero_diff=filter_zero_diff
        )
        result = evaluate_llm(st_norm, lg_norm)

        return {
            "file_mapping": {
                "statement": statement_file.filename,
                "ledger": ledger_file.filename,
            },
            "summary": result["summary"],
            "results": result["results"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Bank reconciliation failed: {e}")
