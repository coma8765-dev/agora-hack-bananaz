import datetime
import logging
import time

from fastapi import FastAPI

from app.ml import ML
from app.schemas import *

app = FastAPI()

ml = ML()

logger = logging.getLogger(__name__)


@app.post("/match_products", response_model=list[Product])
async def predict(data: list[ProductPredict]) -> list[Product]:
    d = datetime.datetime.now()
    r = ml.predict(data)
    print("Global", datetime.datetime.now() - d)
    return r
