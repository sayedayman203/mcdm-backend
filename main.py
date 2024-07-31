from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from MCDM.mcdm import MCDMProblem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Criteria(BaseModel):
    beneficial: bool
    compared2best: int
    compared2worst: int


class CriteriasConfig(BaseModel):
    values: dict[str, Criteria]
    best_criteria: str
    worst_criteria: str


class Decision(BaseModel):
    low: float
    mid: float
    high: float
    T: float
    I: float
    F: float


@app.post("/calculate")
async def calculate_mcdm(
    criterias_config: CriteriasConfig,
    alternates: list[str],
    decisions: list[list[list[Decision]]] = Body(
        description="decision_makers -> alternates -> criterias -> decision",
    ),
):
    problem = MCDMProblem(criterias_config, alternates, decisions)
    return problem.solve()
