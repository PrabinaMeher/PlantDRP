"""
plantdrp.server
===============
FastAPI backend that connects the HTML frontend
to the PlantDRP predictor.

Route registration order matters in FastAPI:
  1. API routes  (@app.get / @app.post) — registered first
  2. StaticFiles mounts                — registered last
Any mount registered before an API route can shadow it.
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="PlantDRP API",
    description="Plant Disease Resistance Protein Predictor",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "web" / "index.html"


#  1. API routes (must come before any app.mount()) ─

@app.get("/")
def home():
    return FileResponse(str(HTML_FILE))


@app.get("/health")
def health():
    """Quick liveness check — useful for the browser auto-open poller."""
    return {"status": "ok", "tool": "PlantDRP"}


@app.post("/predict")
async def predict(
    file:      UploadFile = File(...),
    model:     str        = Form("ds2"),
    threshold: float      = Form(0.5),
    device:    str        = Form("auto"),
):
    """
    Receive a FASTA file, run prediction, return JSON results.

    Form fields
    -----------
    file      : protein FASTA file (multipart upload)
    model     : ds1 | ds2  (default: ds2)
    threshold : float 0–1  (default: 0.5)
    device    : auto | cpu | cuda | mps  (default: auto)
    """
    #  validate model 
    model = model.lower().strip()
    if model not in ("ds1", "ds2"):
        return JSONResponse(
            status_code=422,
            content={"error": f"Invalid model '{model}'. Choose ds1 or ds2."},
        )

    #  validate threshold 
    if not 0.0 <= threshold <= 1.0:
        return JSONResponse(
            status_code=422,
            content={"error": f"threshold must be between 0 and 1, got {threshold}."},
        )

    #  save upload to a temp file 
    suffix = Path(file.filename or "upload").suffix or ".fasta"
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, delete=False
        ) as tmp:
            content = await file.read()
            if not content:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Uploaded file is empty."},
                )
            tmp.write(content)
            tmp_path = tmp.name

        #  run prediction 
        from plantdrp import Predictor  # lazy import keeps startup fast

        pred = Predictor(model=model, device=device, threshold=threshold)
        df   = pred.predict(tmp_path, verbose=False)

        results = df.to_dict(orient="records")

        return {
            "status":    "success",
            "model":     model.upper(),
            "total":     len(df),
            "dr_count":  int((df["prediction"] == "DR").sum()),
            "ndr_count": int((df["prediction"] == "Non-DR").sum()),
            "results":   results,
        }

    except FileNotFoundError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    except Exception as exc:
        # Return the real exception message so the frontend can display it
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(exc).__name__}: {exc}"},
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


#  2. Static file mounts (always LAST — mounts shadow routes below them) 

_dataset_dir = BASE_DIR / "Dataset"
if _dataset_dir.exists():
    app.mount(
        "/Dataset",
        StaticFiles(directory=str(_dataset_dir)),
        name="dataset",
    )
