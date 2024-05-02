from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
import pandas as pd
import secrets

from enums import Plots, Options, Modes, Models
from visualization import Visualization
from preprocessing import Preprocessing
from metrics import Metrics
from crud import ProjectCRUD
from models import Project
from db import SessionDep
import os

# Create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_URL = "http://localhost:8000"
DATASETS_PATH = "./static/datasets/"
FINAL_DATASET_PATH = "./static/final/"
EDIT_DATASET_PATH = "./static/edit/"
MODEL_PATH = "./static/models/"

if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)

if not os.path.exists(FINAL_DATASET_PATH):
    os.makedirs(FINAL_DATASET_PATH)

if not os.path.exists(EDIT_DATASET_PATH):
    os.makedirs(EDIT_DATASET_PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


# Create a new project record
@app.post("/project", response_model=Project)
async def create_project(
    user_id: int, name: str, description: str, session: SessionDep, dataset: UploadFile
):
    fname = dataset.filename.split(".")[0].replace(" ", "_")
    ext = dataset.filename.split(".")[-1]
    filename = f"{fname}-{secrets.token_hex(10)}.{ext}"
    dataset_url = f"{BASE_URL}/static/datasets/{filename}"
    contents = await dataset.read()
    with open(f"{DATASETS_PATH}{filename}", "wb") as f:
        f.write(contents)

    project = Project(
        user_id=user_id,
        name=name,
        description=description,
        dataset_url=dataset_url,
        edit_dataset_url=dataset_url,
        final_dataset_url=dataset_url,
    )
    project = ProjectCRUD.create_project(session, project)
    return project


def description(df: pd.DataFrame) -> dict:
    df = df
    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum(),
            "Data Type": df.dtypes,
        }
    ).to_html(index=False)
    desc = df.describe()
    desc.reset_index(inplace=True)
    desc.rename(columns={"index": "metric"}, inplace=True)
    isna = pd.DataFrame(df.isna().sum())
    isna.reset_index(inplace=True)
    isna.rename(columns={"index": "Column", 0: "Null Count"}, inplace=True)
    return {
        "dims": df.shape,
        "df": df.to_html(index=False),
        "info": info,
        "desc": desc.to_html(index=False),
        "isna": isna.to_html(index=False),
    }


@app.get("/describe")
async def describe(session: SessionDep, project_id: int):
    project = ProjectCRUD.get_project(session=session, id=project_id)
    csv_path = project.dataset_url.replace(BASE_URL, ".")
    df = pd.read_csv(csv_path)
    return description(df)


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
    project_id: int,
):
    project = ProjectCRUD.get_project(session=session, id=project_id)
    csv_name = project.dataset_url.replace(BASE_URL + DATASETS_PATH[1:], "")
    csv_path = project.edit_dataset_url.replace(BASE_URL, ".")
    df = pd.read_csv(csv_path)
    v = Preprocessing(df, csv_name, FINAL_DATASET_PATH, EDIT_DATASET_PATH)
    preprocess_function = getattr(v, option.value)
    preprocess_function(mode)
    edit_dataset_url = f"{BASE_URL}/static/edit/edit_{option}_{csv_name}"
    final_dataset_url = f"{BASE_URL}/static/final/final_{csv_name}"
    project = Project(
        user_id=project_id,
        name=project.name,
        dataset_url=project.dataset_url,
        edit_dataset_url=edit_dataset_url,
        final_dataset_url=final_dataset_url,
    )
    return description(v.df_final)


@app.get("/metrics/{model}")
async def metrics(
    model: Models, session: SessionDep, x: str | None, y: str | None, project_id: int
):
    project = ProjectCRUD.get_project(session=session, id=project_id)
    csv_path = project.final_dataset_url.replace(BASE_URL, ".")
    df = pd.read_csv(csv_path)
    m = Metrics(df, x=x, y=y, model_path=MODEL_PATH, project_id=project_id)
    evaluation_metrics = getattr(m, model.value)
    return {"model": model, "evaluation metrics": evaluation_metrics}
