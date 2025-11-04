# app/routers/lines_explain.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
import pandas as pd
from io import BytesIO
from app.services.lines_explainer import explain_lines

router = APIRouter(prefix="/poc", tags=["poc-lines"])

@router.post("/lines/explain", response_description="Returns an Excel file with ReasonCodes and Suggestion columns")
async def explain_lines_endpoint(lines: UploadFile = File(..., description="Lines.xlsx from BC reconciliation")):
    if not lines.filename.lower().endswith((".xlsx",".xls")):
        raise HTTPException(400, "Please upload an Excel file (.xlsx/.xls)")
    try:
        content = await lines.read()
        df = pd.read_excel(BytesIO(content))
        out_df = explain_lines(df)
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            out_df.to_excel(xw, index=False, sheet_name="LinesExplained")
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="Lines_explained.xlsx"'}
        )
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, f"Failed to process: {e}")
