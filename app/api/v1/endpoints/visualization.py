"""
Visualization endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any

from app.services.visualization import VisualizationService
from app.services.file_service import FileService
from app.schemas.schemas import VisualizationRequest, VisualizationResponse, ErrorResponse
from app.core.enums import PlotType

router = APIRouter()
file_service = FileService()


async def get_data_dependency(file_id: str):
    """Dependency to load data from file"""
    result = await file_service.load_data(file_id)
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
    return result["data"]


@router.post("/generate", response_model=VisualizationResponse)
async def generate_visualization(request: VisualizationRequest):
    """
    Generate a visualization from uploaded data
    
    - **file_id**: ID of the uploaded data file
    - **plot_type**: Type of plot to generate
    - **x_column**: Column for X-axis (optional for some plots)
    - **y_column**: Column for Y-axis (optional for some plots)
    - **color_column**: Column for color grouping (optional)
    - **plot_params**: Additional plot parameters (optional)
    """
    # Load data
    data = await get_data_dependency(request.file_id)
    
    # Create visualization service
    viz_service = VisualizationService(data)
    
    # Generate plot
    result = viz_service.generate_plot(
        plot_type=request.plot_type,
        x_column=request.x_column,
        y_column=request.y_column,
        color_column=request.color_column,
        **request.plot_params
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return VisualizationResponse(
        success=True,
        plotly_json=result["plotly_json"],
        plot_type=request.plot_type.value if hasattr(request.plot_type, 'value') else str(request.plot_type),
        columns_used={
            "x_column": request.x_column,
            "y_column": request.y_column,
            "color_column": request.color_column
        },
        plot_info=result.get("plot_info", {})
    )


@router.get("/plot-types")
async def get_available_plot_types():
    """
    Get list of available plot types and their requirements
    """
    plot_info = {
        PlotType.HISTOGRAM: {
            "description": "Distribution of a single variable",
            "required_columns": ["x_column"],
            "optional_columns": ["color_column"],
            "supports_categorical": True
        },
        PlotType.SCATTER: {
            "description": "Relationship between two continuous variables",
            "required_columns": ["x_column", "y_column"],
            "optional_columns": ["color_column"],
            "supports_categorical": False
        },
        PlotType.BOXPLOT: {
            "description": "Distribution and outliers of data",
            "required_columns": ["y_column"],
            "optional_columns": ["x_column", "color_column"],
            "supports_categorical": True
        },
        PlotType.VIOLIN: {
            "description": "Distribution shape and density",
            "required_columns": ["y_column"],
            "optional_columns": ["x_column", "color_column"],
            "supports_categorical": True
        },
        PlotType.CORRELATION_MATRIX: {
            "description": "Correlation matrix or 2D data",
            "required_columns": [],
            "optional_columns": ["x_column", "y_column"],
            "supports_categorical": False
        },
        PlotType.PAIRPLOT: {
            "description": "Pairwise relationships between variables",
            "required_columns": [],
            "optional_columns": ["color_column"],
            "supports_categorical": False
        },
        PlotType.PCA_SCATTER: {
            "description": "Principal Component Analysis visualization",
            "required_columns": [],
            "optional_columns": ["color_column"],
            "supports_categorical": False
        },
        PlotType.COUNTPLOT: {
            "description": "Count of categorical values",
            "required_columns": ["x_column"],
            "optional_columns": ["color_column"],
            "supports_categorical": True
        },
        PlotType.KDE: {
            "description": "Probability density estimation",
            "required_columns": ["x_column"],
            "optional_columns": ["color_column"],
            "supports_categorical": False
        },
        PlotType.PIE: {
            "description": "Pie chart for categorical distribution",
            "required_columns": ["x_column"],
            "optional_columns": [],
            "supports_categorical": True
        },
        PlotType.TARGET_MEAN: {
            "description": "Mean of target by category",
            "required_columns": ["x_column", "y_column"],
            "optional_columns": [],
            "supports_categorical": True
        },
        PlotType.STACKED_BAR: {
            "description": "Stacked bar chart",
            "required_columns": ["x_column", "y_column"],
            "optional_columns": ["color_column"],
            "supports_categorical": True
        },
        PlotType.CHI_SQUARED_HEATMAP: {
            "description": "Chi-squared test heatmap",
            "required_columns": ["x_column", "y_column"],
            "optional_columns": [],
            "supports_categorical": True
        },
        PlotType.CLASS_IMBALANCE: {
            "description": "Class distribution analysis",
            "required_columns": ["x_column"],
            "optional_columns": [],
            "supports_categorical": True
        },
        PlotType.LEARNING_CURVE: {
            "description": "Model learning curve analysis",
            "required_columns": ["x_column", "y_column"],
            "optional_columns": [],
            "supports_categorical": False
        }
    }
    
    return {
        "success": True,
        "plot_types": [plot_type.value for plot_type in PlotType],
        "plot_info": plot_info
    }


@router.get("/data-info/{file_id}")
async def get_data_info_for_plotting(file_id: str):
    """
    Get data information useful for plotting
    
    - **file_id**: ID of the uploaded data file
    """
    data = await get_data_dependency(file_id)
    
    viz_service = VisualizationService(data)
    data_info = viz_service.get_data_info()
    
    return {
        "success": True,
        "data_info": data_info
    }


@router.post("/batch")
async def generate_batch_visualizations(
    file_id: str,
    plot_types: List[PlotType],
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None
):
    """
    Generate multiple visualizations at once
    
    - **file_id**: ID of the uploaded data file
    - **plot_types**: List of plot types to generate
    - **x_column**: Default X-axis column for all plots
    - **y_column**: Default Y-axis column for all plots
    - **color_column**: Default color grouping column for all plots
    """
    data = await get_data_dependency(file_id)
    viz_service = VisualizationService(data)
    
    results = []
    errors = []
    
    for plot_type in plot_types:
        try:
            result = viz_service.generate_plot(
                plot_type=plot_type,
                x_column=x_column,
                y_column=y_column,
                color_column=color_column
            )
            
            if "error" in result:
                errors.append({"plot_type": plot_type.value, "error": result["error"]})
            else:
                results.append({
                    "plot_type": plot_type.value,
                    "plot_base64": result["plot_base64"],
                    "plot_info": result.get("plot_info", {})
                })
        except Exception as e:
            errors.append({"plot_type": plot_type.value, "error": str(e)})
    
    return {
        "success": True,
        "results": results,
        "errors": errors,
        "total_requested": len(plot_types),
        "successful": len(results),
        "failed": len(errors)
    }
