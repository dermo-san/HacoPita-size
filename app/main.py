from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from urllib.parse import quote

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


def _ascii_safe_filename(name: str, default: str = "predictions.csv") -> str:
    base = os.path.basename(name or "")
    if not base:
        return default

    stem, ext = os.path.splitext(base)
    ext = ext if ext else ".csv"

    stem_ascii = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not stem_ascii:
        stem_ascii = "predictions"

    return f"{stem_ascii}{ext}"


def _content_disposition(original_filename: str, fallback_ascii: str) -> str:
    quoted = quote(original_filename)
    return f'attachment; filename="{fallback_ascii}"; filename*=UTF-8\'\'{quoted}'


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
    orig = file.filename or "input.csv"
    download_orig = orig.rsplit(".", 1)[0] + "_with_predictions.csv"

    fallback = _ascii_safe_filename(download_orig, default="predictions_with_predictions.csv")
    cd = _content_disposition(download_orig, fallback)
    headers = {"Content-Disposition": cd}

    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers=headers,
    )
