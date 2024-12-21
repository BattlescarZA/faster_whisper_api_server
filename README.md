# Whisper Audio Transcription API

A FastAPI-based service that transcribes audio files using OpenAI's Whisper models. The API provides two endpoints with different models optimized for either speed or accuracy.

## Features

- Two transcription endpoints:
  - `/transcribe/fast`: Quick transcription using base model
  - `/transcribe/accurate`: High-accuracy transcription using large model
- Support for multiple audio formats (.mp3, .wav, .m4a)
- Returns both complete transcription and timestamped segments
- Lazy loading of models (only loads when first requested)
- Automatic retry mechanism for model downloads
- FastAPI-powered REST API with automatic OpenAPI documentation

## Setup with Conda

1. Create a new conda environment:
```bash
conda create -n whisper_api python=3.10
conda activate whisper_api
```

2. Install PyTorch (required for Whisper):
```bash
# For CUDA (GPU) support:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### POST /transcribe/fast
Quick transcription using the base model (~142MB).

- **Request**: Multipart form data with an audio file
- **Supported formats**: .mp3, .wav, .m4a
- **Returns**: JSON with transcribed text and segments
- **Model**: Whisper base model (loads on first request)

Example using curl:
```bash
curl -X POST -F "file=@your_audio.mp3" http://localhost:8000/transcribe/fast
```

### POST /transcribe/accurate
High-accuracy transcription using the large model (~2.9GB).

- **Request**: Multipart form data with an audio file
- **Supported formats**: .mp3, .wav, .m4a
- **Returns**: JSON with transcribed text and segments
- **Model**: Whisper large model (loads on first request)

Example using curl:
```bash
curl -X POST -F "file=@your_audio.mp3" http://localhost:8000/transcribe/accurate
```

Example response for both endpoints:
```json
{
    "text": "Transcribed text content...",
    "segments": [
        {
            "text": "Segment text...",
            "start": 0.0,
            "end": 2.5
        }
    ]
}
```

### GET /
Health check endpoint that shows:
- API status
- Available endpoints
- Current model loading status

## Error Handling

The API will return appropriate HTTP status codes and error messages:
- 400: Invalid request (no file, unsupported file type, or invalid model)
- 500: Server error during transcription or model loading

## System Requirements

- Python 3.10 or higher
- Sufficient RAM:
  - Minimum 4GB for base model
  - Minimum 8GB for large model
- GPU is recommended for faster transcription but not required
- CUDA support (optional, for GPU acceleration)

## Notes

- Models are loaded lazily (only when first requested)
- The base model provides a good balance of speed and accuracy
- The large model provides the highest accuracy but requires more resources
- First-time use of each endpoint will download the respective model:
  - Base model: ~142MB
  - Large model: ~2.9GB
- Model downloads include automatic retries with exponential backoff
- Processing time varies based on:
  - Length of the audio file
  - Selected model size
  - Available system resources (CPU/GPU)
