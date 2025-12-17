from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from .inference import (
    CsvDecodingError,
    MissingColumnsError,
    predict_from_bytes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HacoPita box id predictor")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result_df = predict_from_bytes(file_bytes)
    except MissingColumnsError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {exc}",
        ) from exc
    except CsvDecodingError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV must be encoded in utf-8 or cp932.",
        ) from exc
    except Exception as exc:  # pragma: no cover - fallback
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    original_name = Path(file.filename or "predictions").stem or "predictions"
    download_name = f"{original_name}_with_predictions.csv"
    headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers=headers,
    )
