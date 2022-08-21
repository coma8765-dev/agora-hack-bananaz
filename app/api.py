import threading

import aiohttp
from fastapi import FastAPI

from app.ml import ML
from app.schemas import *

app = FastAPI()

ml = ML()


@app.post("/match_products", response_model=list[Product])
async def predict(data: list[ProductPredict]) -> list[Product]:
    return ml.predict(data)


# @app.post("/core/retrain")
# async def retrain(data_url: str):
#     with aiohttp.ClientSession() as session:
#         async with session.get(data_url) as resp:
#             data = await resp.json()
#             card = data["card_image"]
#             async with session.get(card) as resp2:
#                 test = await resp2.read()
#                 with open("cardtest2.png", "wb") as f:
#                     f.write(test)
#
#     threading.Thread(target=ml.retrain, args=(data_url,))


# @app.get("/core/version", response_model=list[int])
# async def change_version():
#     ...


# @app.patch(
#     "/core/version",
#     description="Change the current version of the model",
# )
# async def change_version(version: int):
#     ml.change_version(version)
