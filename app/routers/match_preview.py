# app/routers/match_preview.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import pandas as pd
from io import BytesIO

from app.utils.logger import get_logger, log_duration
from app.services.llm_evaluator import evaluate_llm

router = APIRouter(prefix="/poc", tags=["poc-matching"])
log = get_logger(__name__)

def _read_csv(file: UploadFile) -> pd.DataFrame:
    if not file:
        raise HTTPException(status_code=400, detail="File missing")
    data = file.file.read()
    try:
        return pd.read_csv(BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

@router.post("/match/llm-eval")
async def llm_eval_endpoint(
    mismatches: UploadFile = File(..., description="CSV: tx_id,post_date,amount,currency,merchant_desc"),
    receipts:   UploadFile = File(..., description="CSV: receipt_id,tx_date,total_amount,currency,merchant_canonical")
) -> Dict[str, Any]:
    mm_df = _read_csv(mismatches)
    rc_df = _read_csv(receipts)

    # time the evaluation using your context manager
    with log_duration("poc.match.llm_eval", mismatches_rows=len(mm_df), receipts_rows=len(rc_df)):
        try:
            result = evaluate_llm(mm_df, rc_df)
            return result
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception:
            log.exception("LLM evaluation failed")
            raise HTTPException(status_code=500, detail="LLM evaluation failed")
