import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from .schemas import MatchResponse
from .adapters.recording_model_adapter import RecordingModelAdapter

app = FastAPI(title="AI Adapter Response API", version="0.1.0")
# configure adapter with a processing-time budget in seconds
adapter = RecordingModelAdapter(max_processing_time_seconds=8)


@app.post("/match", response_model=MatchResponse)
async def match_endpoint(recording: UploadFile = File(...), image: UploadFile = File(...), timestamp_ms: int = Form(...)):
    """
    Match a provided image against a frame extracted from the uploaded recording at timestamp_ms (milliseconds).
    - recording: video file (multipart/form-data)
    - image: image file (multipart/form-data)
    - timestamp_ms: integer milliseconds offset into the video to extract a frame
    Returns MatchResponse.
    """
    # read bytes into memory (careful with very large files in production)
    try:
        recording_bytes = await recording.read()
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded files: {e}")

    loop = asyncio.get_event_loop()
    try:
        # run the synchronous adapter in threadpool and enforce timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(None, adapter.match_image_at_time, recording_bytes, image_bytes, timestamp_ms),
            timeout=adapter.max_processing_time_seconds + 1,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return result