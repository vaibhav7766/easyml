from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
import pandas as pd
import secrets

from visualization import Visualization
from crud import HistoryCRUD
from models import History
from db import SessionDep
from enums import Plots


# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
DATASETS_PATH = "./static/datasets/"
BASE_URL = "http://localhost:8000"


# Create a new history record
@app.post("/history", response_model=History)
async def create_history(project_name: str, session: SessionDep, dataset: UploadFile):
    name = dataset.filename.split(".")[0]
    ext = dataset.filename.split(".")[-1]
    filename = f"{name}-{secrets.token_hex(10)}.{ext}"
    dataset_url = f"{BASE_URL}/static/datasets/{filename}"
    contents = await dataset.read()
    with open(f"{DATASETS_PATH}{filename}", "wb") as f:
        f.write(contents)

    history = History(project_name=project_name, dataset_url=dataset_url)
    history = HistoryCRUD.create_history(session, history)
    return history


# Get all history records
@app.get("/history", response_model=list[History])
async def get_histories(session: SessionDep):
    histories = HistoryCRUD.get_histories(session)
    return histories


# Get a single history record
@app.get("/history/{id}", response_model=History)
async def get_history(session: SessionDep, id: int):
    history = HistoryCRUD.get_history(session=session, id=id)
    return history


# Visualization
@app.get("/visualize/{plot_name}")
async def visualize(
    plot_name: Plots,
    session: SessionDep,
    history_id: int,
    x: str | None = None,
    y: str | None = None,
):
    history = HistoryCRUD.get_history(session=session, id=history_id)
    csv_path = history.dataset_url.replace(BASE_URL, ".")
    df = pd.read_csv(csv_path)
    v = Visualization(df)
    plot_function = getattr(v, plot_name.value)
    plot_html = plot_function(x=x, y=y)
    return {"plot_name": plot_name, "plot_html": plot_html}
