from pydantic.main import BaseModel


class ProductPredict(BaseModel):
    id: str
    name: str
    props: list[str]


class ProductScore(BaseModel):
    reference_id: str
    score: float


class Product(BaseModel):
    id: str
    reference_id: str | None
    scores: list[ProductScore]
