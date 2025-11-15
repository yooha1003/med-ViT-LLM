"""
FastAPI server for MRI diagnosis pipeline
Provides REST API endpoints for medical image analysis
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import nibabel as nib
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline.main_pipeline import MRIDiagnosisPipeline

# Initialize FastAPI app
app = FastAPI(
    title="MRI Diagnosis API",
    description="End-to-end MRI image analysis with segmentation, feature extraction, and clinical interpretation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None


# Pydantic models for request/response
class PatientInfo(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    chief_complaint: Optional[str] = None
    clinical_history: Optional[str] = None


class AnalysisRequest(BaseModel):
    structure_name: str = "brain region"
    patient_info: Optional[PatientInfo] = None
    clinical_context: Optional[str] = None
    use_llm: bool = False


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    analysis_id: str
    features: Optional[Dict] = None
    interpretation: Optional[str] = None
    report: Optional[Dict] = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    print("Initializing MRI Diagnosis Pipeline...")

    try:
        pipeline = MRIDiagnosisPipeline(
            segmentation_model_type="unet3d",
            llm_model_name="microsoft/phi-2",
            use_gpu=True,
            use_llm=False  # Can be configured via environment variable
        )
        print("✓ Pipeline initialized successfully!")

    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        pipeline = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="running",
        pipeline_loaded=pipeline is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        pipeline_loaded=pipeline is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_mri(
    file: UploadFile = File(...),
    structure_name: str = Form("brain region"),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    clinical_context: Optional[str] = Form(None),
    use_llm: bool = Form(False)
):
    """
    Analyze MRI image

    Upload a NIfTI file (.nii or .nii.gz) for analysis.
    Returns segmentation, features, and clinical interpretation.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    if not file.filename.endswith(('.nii', '.nii.gz')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload NIfTI file (.nii or .nii.gz)"
        )

    # Create temporary directory for this analysis
    temp_dir = Path(tempfile.mkdtemp())
    analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Save uploaded file
        input_path = temp_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Prepare patient info
        patient_info = {}
        if patient_age:
            patient_info['age'] = patient_age
        if patient_sex:
            patient_info['sex'] = patient_sex

        # Run pipeline
        results = pipeline.run(
            image_path=str(input_path),
            structure_name=structure_name,
            patient_info=patient_info if patient_info else None,
            clinical_context=clinical_context,
            output_dir=str(temp_dir / "output"),
            save_segmentation=True,
            save_features=True,
            save_report=True
        )

        # Prepare response
        response = AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            analysis_id=analysis_id,
            features=results.get('features'),
            interpretation=results.get('interpretation'),
            report=results.get('report'),
            timestamp=datetime.now().isoformat()
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Cleanup temporary files
        # Note: In production, you might want to keep these for a while
        # shutil.rmtree(temp_dir)
        pass


@app.post("/segment")
async def segment_only(
    file: UploadFile = File(...),
    structure_name: str = Form("brain region")
):
    """
    Perform segmentation only (no feature extraction or interpretation)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Save uploaded file
        input_path = temp_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run segmentation
        mask, probs = pipeline.segmentation_model.segment(image_path=str(input_path))

        # Save segmentation
        output_path = temp_dir / f"segmentation_{structure_name}.nii.gz"
        nifti_img = nib.load(str(input_path))
        mask_nifti = nib.Nifti1Image(mask, nifti_img.affine, nifti_img.header)
        nib.save(mask_nifti, str(output_path))

        # Return file
        return FileResponse(
            path=str(output_path),
            filename=f"segmentation_{structure_name}.nii.gz",
            media_type="application/gzip"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/features")
async def extract_features_only(
    image_file: UploadFile = File(...),
    mask_file: UploadFile = File(...)
):
    """
    Extract features from image and mask
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Save uploaded files
        image_path = temp_dir / image_file.filename
        mask_path = temp_dir / mask_file.filename

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)

        with open(mask_path, "wb") as buffer:
            shutil.copyfileobj(mask_file.file, buffer)

        # Extract features
        features = pipeline.feature_extractor.extract_all_features(
            image_path=str(image_path),
            mask_path=str(mask_path)
        )

        return JSONResponse(content={
            "success": True,
            "features": features,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    finally:
        shutil.rmtree(temp_dir)


@app.post("/interpret")
async def interpret_features(
    features: Dict,
    structure_name: str = "brain region",
    patient_info: Optional[PatientInfo] = None,
    clinical_context: Optional[str] = None
):
    """
    Generate clinical interpretation from features
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Convert patient info to dict
        patient_dict = patient_info.dict() if patient_info else None

        # Generate interpretation
        interpretation = pipeline.interpreter.generate_interpretation(
            features=features,
            structure_name=structure_name,
            patient_info=patient_dict,
            clinical_context=clinical_context
        )

        # Generate report
        report = pipeline.interpreter.generate_report(
            features=features,
            structure_name=structure_name,
            patient_info=patient_dict,
            clinical_context=clinical_context
        )

        return JSONResponse(content={
            "success": True,
            "interpretation": interpretation,
            "report": report,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {str(e)}")


@app.get("/info")
async def get_pipeline_info():
    """Get information about the pipeline configuration"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return JSONResponse(content={
        "segmentation_model": pipeline.segmentation_model.model_type,
        "device": str(pipeline.segmentation_model.device),
        "llm_available": hasattr(pipeline.interpreter, 'model_name'),
        "radiomics_available": pipeline.feature_extractor.use_radiomics
    })


def main():
    """Run the API server"""
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
