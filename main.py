from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
import pandas as pd
import secrets

from visualization import Visualization
from preprocessing import Preprocessing
from metrics import Metrics
from crud import ProjectCRUD
from models import Project
from db import SessionDep
from enums import Plots ,Options,Modes,Modals

# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
DATASETS_PATH = "./static/datasets/"
BASE_URL = "http://localhost:8000"
FINAL_DATASET_PATH = "./static/final/"
EDIT_DATASET_PATH = "./static/edit/"
MODAL_PATH = "./static/modals/"

# Create a new project record
@app.post("/project", response_model=Project)
async def create_project(
    user_id: int, name: str, session: SessionDep, dataset: UploadFile
):
    fname = dataset.filename.split(".")[0].replace(' ', '_')
    ext = dataset.filename.split(".")[-1]
    filename = f"{fname}-{secrets.token_hex(10)}.{ext}"
    dataset_url = f"{BASE_URL}/static/datasets/{filename}"
    contents = await dataset.read()
    with open(f"{DATASETS_PATH}{filename}", "wb") as f:
        f.write(contents)

    project = Project(user_id=user_id, name=name, dataset_url=dataset_url
                      ,edit_dataset_url=dataset_url,final_dataset_url=dataset_url)
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

@app.get("/preprocess/{option}")
async def preprocess(
    option: Options,
    mode: Modes,
    session: SessionDep,
    project_id: int, ):
    project = ProjectCRUD.get_project(session=session, id=project_id)
    csv_name = project.dataset_url.replace(BASE_URL+DATASETS_PATH[1:], "")
    csv_path = project.edit_dataset_url.replace(BASE_URL, ".")    
    df = pd.read_csv(csv_path)
    v = Preprocessing(df, csv_name, FINAL_DATASET_PATH, EDIT_DATASET_PATH)
    preprocess_function = getattr(v, option.value)
    preprocessed_df = preprocess_function(mode)
    edit_dataset_url = f"{BASE_URL}/static/edit/edit_{option}_{csv_name}"
    final_dataset_url = f"{BASE_URL}/static/final/final_{csv_name}"
    project = Project(user_id=project_id, name=project.name, dataset_url=project.dataset_url
                      ,edit_dataset_url=edit_dataset_url,final_dataset_url=final_dataset_url)
    ProjectCRUD.update_project(session, project)
    return {"df": preprocessed_df,"option":option,"mode":mode}

@app.get("/metrics/{Models}")
async def metrics(
    Models: Modals,
    session: SessionDep,
    x: str | None ,
    y: str | None ,
    project_id: int ):
    project = ProjectCRUD.get_project(session=session, id=project_id)
    csv_path = project.final_dataset_url.replace(BASE_URL, ".")
    df = pd.read_csv(csv_path)
    m = Metrics(df,x=x, y=y, modal_path=MODAL_PATH)
    evaluation_metrics = getattr(m, Models.value)
    return {"Models":Models,"evaluation metrics":evaluation_metrics}