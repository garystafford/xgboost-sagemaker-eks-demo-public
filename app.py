from typing import List

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request


app = FastAPI()
booster = xgb.Booster()
booster.load_model("xgboost-model")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = (await request.body()).decode("utf-8").strip()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    rows: List[List[float]] = []
    for line in body.splitlines():
        try:
            rows.append([float(value) for value in line.split(",")])
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid CSV row: {line}"
            ) from exc

    data = np.array(rows, dtype=float)
    predictions = booster.predict(xgb.DMatrix(data))
    return {"predictions": predictions.tolist()}
