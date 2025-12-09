"""
FastAPI Application for Otto Product Classification
Supports both single and batch predictions with confidence scores
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import pickle
import numpy as np
import pandas as pd
import uvicorn
from pathlib import Path
import logging
from datetime import datetime
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Otto Product Classification API",
    description="XGBoost-based product classification with confidence scores",
    version="1.0.0"
)

# Global variables for model and encoder
model = None
label_encoder = None
feature_names = None
MODEL_PATH = Path(__file__).parent.parent / "model" / "xgboost_optimized_model.pkl"
ENCODER_PATH = Path(__file__).parent.parent / "model" / "label_encoder.pkl"


# ============================================================================
# Data Models (Pydantic schemas)
# ============================================================================

class PredictionInput(BaseModel):
    """Schema for single prediction input"""
    features: List[float] = Field(..., min_items=93, max_items=93)
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 93:
            raise ValueError('Expected exactly 93 features')
        if any(x < 0 for x in v):
            raise ValueError('All features must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 0.0, 0.0, 2.0] + [0.0] * 89  # 93 features total
            }
        }


class BatchPredictionInput(BaseModel):
    """Schema for batch prediction input"""
    instances: List[List[float]] = Field(..., min_items=1, max_items=10000)
    
    @validator('instances')
    def validate_instances(cls, v):
        if len(v) > 10000:
            raise ValueError('Maximum 10000 instances per batch')
        for i, instance in enumerate(v):
            if len(instance) != 93:
                raise ValueError(f'Instance {i}: Expected exactly 93 features, got {len(instance)}')
            if any(x < 0 for x in instance):
                raise ValueError(f'Instance {i}: All features must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    [1.0, 0.0, 0.0, 2.0] + [0.0] * 89,
                    [0.0, 1.0, 2.0, 0.0] + [0.0] * 89
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Schema for single prediction output"""
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_probabilities: Dict[str, float]
    prediction_time: str


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output"""
    predictions: List[PredictionOutput]
    total_instances: int
    prediction_time: str


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    encoder_loaded: bool
    timestamp: str


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_model():
    """Load the trained XGBoost model"""
    global model
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_label_encoder():
    """Load the label encoder"""
    global label_encoder
    try:
        if not ENCODER_PATH.exists():
            logger.error(f"Encoder file not found at {ENCODER_PATH}")
            raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")
        
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info(f"Label encoder loaded successfully from {ENCODER_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error loading label encoder: {str(e)}")
        raise


# ============================================================================
# Prediction Helper Functions
# ============================================================================

def make_single_prediction(features: List[float]) -> Dict:
    """
    Make a single prediction with confidence scores
    
    Args:
        features: List of 93 feature values
        
    Returns:
        Dictionary with prediction, confidence, and probabilities
    """
    try:
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Get prediction
        start_time = datetime.now()
        y_pred = model.predict(X)[0]
        y_pred_proba = model.predict_proba(X)[0]
        end_time = datetime.now()
        
        # Get predicted class name
        predicted_class = label_encoder.inverse_transform([y_pred])[0]
        
        # Get confidence (max probability)
        confidence = float(np.max(y_pred_proba))
        
        # Get all class probabilities
        all_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(y_pred_proba)
        }
        
        # Sort probabilities by value
        all_probabilities = dict(sorted(
            all_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        prediction_time = (end_time - start_time).total_seconds() * 1000  # ms
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "prediction_time": f"{prediction_time:.2f}ms"
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def make_batch_predictions(instances: List[List[float]]) -> List[Dict]:
    """
    Make batch predictions with confidence scores
    
    Args:
        instances: List of feature arrays
        
    Returns:
        List of prediction dictionaries
    """
    try:
        # Convert to numpy array
        X = np.array(instances)
        
        # Get predictions
        start_time = datetime.now()
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds() * 1000  # ms
        avg_time_per_instance = total_time / len(instances)
        
        # Process each prediction
        predictions = []
        for i in range(len(instances)):
            predicted_class = label_encoder.inverse_transform([y_pred[i]])[0]
            confidence = float(np.max(y_pred_proba[i]))
            
            all_probabilities = {
                label_encoder.inverse_transform([j])[0]: float(prob)
                for j, prob in enumerate(y_pred_proba[i])
            }
            
            all_probabilities = dict(sorted(
                all_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            predictions.append({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "prediction_time": f"{avg_time_per_instance:.2f}ms"
            })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and encoder on startup"""
    logger.info("Starting Otto Classification API...")
    try:
        load_model()
        load_label_encoder()
        logger.info("API ready to serve predictions!")
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Otto Product Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "predict_csv": "/predict_csv (POST)",
            "model_info": "/model_info"
        },
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if (model is not None and label_encoder is not None) else "unhealthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model_info", tags=["General"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features": 93,
        "n_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "model_path": str(MODEL_PATH),
        "hyperparameters": {
            "n_estimators": 473,
            "max_depth": 15,
            "learning_rate": 0.048040601473231026,
            "subsample": 0.8305151450727134,
            "colsample_bytree": 0.5756137233125057,
            "gamma": 0.17199183360379053,
            "min_child_weight": 3
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_single(input_data: PredictionInput):
    """
    Make a single prediction
    
    - **features**: List of 93 feature values (all non-negative)
    
    Returns:
    - **predicted_class**: Predicted product class
    - **confidence**: Confidence score (0-1)
    - **all_probabilities**: Probability for each class
    - **prediction_time**: Time taken for prediction
    """
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = make_single_prediction(input_data.features)
        logger.info(f"Single prediction: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Make batch predictions
    
    - **instances**: List of feature arrays (max 10000 instances)
    - Each instance must have exactly 93 features
    
    Returns:
    - **predictions**: List of predictions with confidence scores
    - **total_instances**: Number of instances processed
    - **prediction_time**: Total time taken
    """
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        predictions = make_batch_predictions(input_data.instances)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction: {len(predictions)} instances in {total_time:.2f}ms")
        
        return {
            "predictions": predictions,
            "total_instances": len(predictions),
            "prediction_time": f"{total_time:.2f}ms"
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_csv", tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Make predictions from CSV file
    
    - **file**: CSV file with 93 feature columns
    - Returns: CSV file with predictions and confidence scores
    """
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        logger.info(f"Processing CSV with {len(df)} rows")
        
        # Check number of columns
        if len(df.columns) not in [93, 94]:  # 93 features or 93 + id column
            raise HTTPException(
                status_code=400,
                detail=f"Expected 93 or 94 columns, got {len(df.columns)}"
            )
        
        # Extract ID column if present
        if 'id' in df.columns:
            ids = df['id'].values
            X = df.drop('id', axis=1).values
        else:
            ids = np.arange(len(df))
            X = df.values
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'id': ids,
            'predicted_class': label_encoder.inverse_transform(y_pred),
            'confidence': np.max(y_pred_proba, axis=1)
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(label_encoder.classes_):
            results[f'prob_{class_name}'] = y_pred_proba[:, i]
        
        # Save to temporary file
        output_path = Path("temp_predictions.csv")
        results.to_csv(output_path, index=False)
        
        logger.info(f"CSV predictions completed: {len(results)} rows")
        
        return FileResponse(
            path=output_path,
            filename=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            media_type='text/csv'
        )
        
    except Exception as e:
        logger.error(f"CSV prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
