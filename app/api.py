from fastapi import FastAPI

from app.ml import ML
from app.schemas import *

app = FastAPI()

ml = ML()


@app.post("/match_products", response_model=list[Product])
async def predict(data: list[ProductPredict]) -> list[Product]:
    return ml.predict(data)
