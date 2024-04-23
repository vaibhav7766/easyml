from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
import pandas as pd
import secrets

from visualization import Visualization
from crud import ProjectCRUD
from models import Project
from db import SessionDep
from enums import Plots


# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
DATASETS_PATH = "./static/datasets/"
BASE_URL = "http://localhost:8000"


# Create a new project record
@app.post("/project", response_model=Project)
async def create_project(
    user_id: int, name: str, session: SessionDep, dataset: UploadFile
):
    fname = dataset.filename.split(".")[0]
    ext = dataset.filename.split(".")[-1]
    filename = f"{fname}-{secrets.token_hex(10)}.{ext}"
    dataset_url = f"{BASE_URL}/static/datasets/{filename}"
    contents = await dataset.read()
    with open(f"{DATASETS_PATH}{filename}", "wb") as f:
        f.write(contents)

    project = Project(user_id=user_id, name=name, dataset_url=dataset_url)
    project = ProjectCRUD.create_project(session, project)
    return project


# Visualization
@app.get("/visualize/{plot_name}")
async def visualize(
    plot_name: Plots,
    session: SessionDep,
    project_id: int,
    x: str | None = None,
    y: str | None = None,
):
    project = ProjectCRUD.get_project(session=session, id=project_id)
    csv_path = project.dataset_url.replace(BASE_URL, ".")
    df = pd.read_csv(csv_path)
    v = Visualization(df)
    plot_function = getattr(v, plot_name.value)
    plot_html = plot_function(x=x, y=y)
    return {"plot_name": plot_name, "plot_html": plot_html}
