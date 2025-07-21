
# Standard library imports
import os

# Third-party imports
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, Query, Depends, Body
from typing import Annotated
import pandas as pd

# Local application imports
# from enums import Plots, Options, Modes, Models
# from visualization import Visualization
from preprocessing import Preprocessing
# from metrics import Metrics
# from crud import ProjectCRUD
from models import create_project_dict, project_from_mongo
from db import SessionDep

# Create FastAPI app

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "Welcome to the EasyML Backend!"}


# 1) Dashboard endpoint
@app.get("/dashboard")
async def dashboard(user_id: Annotated[str, Query(description="User ID")], session: SessionDep = Depends()):
    """
    Main dashboard view that displays an overview of the ML project status and key metrics for the user.
    """
    # TODO: Fetch projects and metrics for the user
    return {"dashboard": f"Overview for user {user_id}"}


# 2) Upload endpoint (updated)
@app.post("/upload")
async def upload_file(
    file: UploadFile,
    project_name: Annotated[str, Query(description="Name of the project")],
    user_id: Annotated[str, Query(description="User ID")],
    session: SessionDep = Depends(),
):
    """
    Upload a file to the server and create a new project for the user.
    """
    if not file.filename.endswith('.csv'):
        return {"error": "Only CSV files are allowed."}
    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{user_id}_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    # Create a new project document in MongoDB
    project_doc = create_project_dict(name=project_name, file_path=file_location, user_id=user_id)
    result = session["projects"].insert_one(project_doc)
    return {"message": "File uploaded successfully", "project_id": str(result.inserted_id)}


# 3) Preprocessing endpoint
@app.post("/preprocessing")
async def preprocessing(
    project_id: Annotated[int, Query(description="Project ID")],
    user_id: Annotated[str, Query(description="User ID")],
    choices: Annotated[dict, Body(description="Preprocessing choices")],
    session: SessionDep = Depends(),
):
    """
    Handles data preprocessing tasks for the user's project.
    """
    data = pd.read_csv(f"uploads/{user_id}_{project_id}.csv")
    preprocessing = Preprocessing(data)
    for choice, column in choices.items():
        if choice == Options.ENCODE:
            preprocessing.encode(column=column, choice=choices[choice])
        elif choice == Options.IMPUTATION:
            preprocessing.imputation(column=column, choice=choices[choice])
        elif choice == Options.DELETE_COLUMNS:
            preprocessing.delete_columns(selected_columns=choices[choice])
        elif choice == Options.NORMALIZE:
            preprocessing.normalize(column=column, choice=choices[choice])
    # Save the processed data back to CSV
    processed_file_path = f"uploads/processed_{user_id}_{project_id}.csv"
    preprocessing.data.to_csv(processed_file_path, index=False)
    # Update the project with the new file path in MongoDB
    from bson import ObjectId
    result = session["projects"].update_one(
        {"_id": ObjectId(project_id), "user_id": user_id},
        {"$set": {"file_path": processed_file_path}}
    )
    if result.matched_count == 0:
        return {"error": "Project not found"}
    return {"preprocessing": f"Preprocessing for project {project_id} (user {user_id})"}


# 4) Visualizations endpoint
@app.get("/visualizations")
async def visualizations(
    project_id: Annotated[str, Query()],
    user_id: Annotated[str, Query()],
    plot_type: Annotated[str, Query()],
    x: Annotated[str, Query()] = None,
    y: Annotated[str, Query()] = None,
    session: SessionDep = Depends(),
):
    path = f"uploads/processed_{user_id}_{project_id}.csv"
    if not os.path.exists(path):
        path = f"uploads/{user_id}_{project_id}.csv"
    df = pd.read_csv(path)
    vis = Visualization(df)
    image_base64 = vis.plot(plot_type=plot_type, x=x, y=y)
    return {"image": image_base64}


# 5) Feature selection endpoint
@app.post("/feature_selection")
async def feature_selection(
    project_id: Annotated[int, Query(description="Project ID")],
    user_id: Annotated[str, Query(description="User ID")],
    method: Annotated[str, Query(description="Feature selection method")],
    session: SessionDep = Depends(),
):
    """
    Provides methods for feature importance ranking and selection for the user's project.
    """
    # TODO: Implement feature selection logic
    return {"feature_selection": f"{method} for project {project_id} (user {user_id})"}


# 6) Choosing models endpoint
@app.post("/choosing_models")
async def choosing_models(
    project_id: Annotated[int, Query(description="Project ID")],
    user_id: Annotated[str, Query(description="User ID")],
    model_name: Annotated[str, Query(description="Model name")],
    hyperparameters: Annotated[dict, Query(description="Model hyperparameters")],
    session: SessionDep = Depends(),
):
    """
    Allows users to select and configure ML algorithms and hyperparameters for the user's project.
    """
    # TODO: Implement model selection/configuration logic
    return {"choosing_models": f"{model_name} for project {project_id} (user {user_id})"}


# 7) Results endpoint
@app.get("/results")
async def results(
    project_id: Annotated[int, Query(description="Project ID")],
    user_id: Annotated[str, Query(description="User ID")],
    session: SessionDep = Depends(),
):
    """
    Displays model evaluation metrics, performance comparisons, and prediction results for the user's project.
    """
    # TODO: Implement results logic
    return {"results": f"Results for project {project_id} (user {user_id})"}