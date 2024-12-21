import os
import whisper
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import logging
from typing import Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcription API")

# Dictionary to store loaded models
models: Dict[str, Optional[any]] = {
    "base": None,
    "large": None
}

def load_model_with_retry(model_size: str, max_retries: int = 3) -> any:
    """Load model with retry mechanism"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading Whisper {model_size} model (attempt {attempt + 1}/{max_retries})...")
            model = whisper.load_model(model_size)
            logger.info(f"{model_size} model loaded successfully!")
            return model
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Failed to load model, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to load {model_size} model after {max_retries} attempts: {str(e)}")
                raise

async def get_model(model_size: str):
    """Get or load model if not already loaded"""
    if models[model_size] is None:
        models[model_size] = load_model_with_retry(model_size)
    return models[model_size]

async def process_audio(file: UploadFile, model_size: str):
    """Common function to process audio files with specified model"""
    if model_size not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model size. Supported sizes: {', '.join(models.keys())}"
        )
    
    # Check if file is provided
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {".mp3", ".wav", ".m4a"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        # Get the appropriate model
        model = await get_model(model_size)
        
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Transcribe the audio file
            result = model.transcribe(temp_file.name)
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            return JSONResponse(content={
                "text": result["text"],
                "segments": result["segments"]
            })
            
    except Exception as e:
        # Clean up temp file in case of error
        if 'temp_file' in locals():
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/fast")
async def transcribe_audio_fast(file: UploadFile):
    """Endpoint using the base model for faster processing"""
    return await process_audio(file, "base")

@app.post("/transcribe/accurate")
async def transcribe_audio_accurate(file: UploadFile):
    """Endpoint using the large model for higher accuracy"""
    return await process_audio(file, "large")

@app.get("/")
async def root():
    model_status = {
        name: "loaded" if model is not None else "not loaded"
        for name, model in models.items()
    }
    
    return {
        "message": "Audio Transcription API is running",
        "endpoints": {
            "/transcribe/fast": "Quick transcription using base model (~142MB)",
            "/transcribe/accurate": "High-accuracy transcription using large model (~2.9GB)"
        },
        "model_status": model_status
    }
