## Project Overview
This project implements the first working AI integration for the Aegis Kavach ecosystem. It establishes a pipeline where RTSP/Camera frames are processed via an ONNX AI Adapter to perform Person Detection on the Edge (CPU).
## Project Structure
```text
AI_ADAPTER/
├── adapter/
│   ├── main.py            # FastAPI Service (The AI Brain)
│   ├── yolov8n.onnx       # The AI Model (Must be exported/downloaded)
│   └── __pycache__/
├── frames/                # Storage for active frames
│   └── camera_1/
│       └── latest.jpg     # The current frame being analyzed
├── kavach/
│   └── runner.py          # The Camera Agent (Captures & Requests)
├── venv/                  # Python Virtual Environment
└── requirements.txt       # Dependencies
```

## How To Run
Step 1:Start the Ai Adapter
```
cd adapter
uvicorn main:app --port 9100 --reload
```

Step 2:Start the runner
```
cd kavach
python runner.py
```
## API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `health` | Service health check |
| `GET` | `capabilities` | Lists supported tasks (`["person_detection"]`) |
| `POST` | `infer` | Main inference endpoint. Accepts KAI-C formatted JSON. |

---

## Recording → Image Matching (Response API)

This module introduces a response-based AI workflow that enables matching a provided image against a frame extracted from a video recording at a specified timestamp.

### API Endpoint

| Method | Endpoint | Description |
|-------|----------|-------------|
| POST | `/match` | Match an image against a frame from a video recording |

### Request Format

The `/match` endpoint accepts `multipart/form-data` with the following fields:

| Field | Type | Description |
|------|------|-------------|
| `recording` | file | Video file (mp4, mov, etc.) |
| `image` | file | Image to match |
| `timestamp_ms` | integer | Timestamp inside the video (in milliseconds) |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/match" \
-F "recording=@VIDEO_TEST.mp4" \
-F "image=@IMAGE_FIND.jpeg" \
-F "timestamp_ms=1000"