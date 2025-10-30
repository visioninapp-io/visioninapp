YOLO Trainer (AI.model_trainer)
===============================

Minimal utilities to select a YOLO model (alias or .pt checkpoint), apply hyperparameters, and train via a simple payload API or inside FastAPI.

Quick start
-----------

```bash
pip install -r requirements.txt
python -m AI.examples.train_yolo
```

One-call payload training
-------------------------

```python
from AI.model_trainer import train_from_payload

# Detection (Ultralytics YOLO)
payload = {
    "model": "yolov8n",  # or "/abs/path/model.pt"
    "hyperparameters": {"epochs": 50, "imgsz": 640, "batch": 16},
    "fit_params": {"data": "/abs/path/data.yaml"}
}
trained, score = train_from_payload(payload)
```

Progress and metrics
--------------------

You can receive epoch updates and final metrics (e.g., precision/recall/mAP50-95) via a callback:

```python
from AI.model_trainer import build_model, train_model

events = []

def on_progress(ev):
    # ev = {"event": "epoch_end"|"train_end", "epoch": int, "total_epochs": int, "metrics": {...}, "save_dir": str}
    events.append(ev)
    print(ev)

yolo = build_model("yolov8n", {"epochs": 10, "imgsz": 640})
train_model(yolo, None, None, fit_params={
    "data": "/abs/path/data.yaml",
    "progress_callback": on_progress,
    "tick_interval": 1.0  # send a 'tick' event every second (default 1.0)
})

# After training
print("Final metrics:", getattr(yolo, "final_metrics", None))
print("Runs saved under:", getattr(yolo, "save_dir", None))
```

Retrieving progress (backend snapshots + polling)
------------------------------------------------

Recommended pattern:
- Use `progress_callback` to update a per-job snapshot in your backend (in-memory or Redis).
- Frontend polls `GET /jobs/{id}` every 2–5s to retrieve the latest snapshot.

Example (FastAPI skeleton):
```python
from typing import Any, Dict
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from AI.model_trainer import train_from_payload

app = FastAPI()
jobs: Dict[str, Dict[str, Any]] = {}

class TrainPayload(BaseModel):
    model: str
    hyperparameters: Dict[str, Any] | None = None
    fit_params: Dict[str, Any] | None = None

def on_progress_factory(job_id: str):
    def on_progress(ev: Dict[str, Any]):
        # Store only the latest snapshot
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "running", "progress": ev, "updated_at": __import__("time").time()}
    return on_progress

def _run(job_id: str, payload: Dict[str, Any]):
    jobs[job_id] = {"status": "running"}
    try:
        # inject server-side callback
        p = dict(payload)
        fp = dict(p.get("fit_params") or {})
        fp["progress_callback"] = on_progress_factory(job_id)
        p["fit_params"] = fp
        _, score = train_from_payload(p)
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "completed", "score": score}
    except Exception as e:
        jobs[job_id] = {**jobs.get(job_id, {}), "status": "failed", "error": str(e)}

@app.post("/train")
def train(payload: TrainPayload, bg: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued"}
    bg.add_task(_run, job_id, payload.dict(exclude_none=True))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    return jobs[job_id]
```

Optional push (SSE)
-------------------
If you prefer pushing updates, expose an SSE endpoint and `yield` each `on_progress` event to connected clients. Polling is simpler and adequate if you don’t need real-time streaming.

Model selection
---------------

- YOLO alias: "yolov8n", "yolov8s", etc. (auto-resolves to .pt)
- YOLO checkpoint: absolute path to `.pt`

Hyperparameters vs fit_params (YOLO)
------------------------------------

- hyperparameters: build-time settings (epochs, imgsz, batch, etc.).
- fit_params: training-time args passed to Ultralytics (REQUIRED: `data` YAML path).

```python
from AI.model_trainer import build_model, train_model

yolo = build_model("yolov8n", {"epochs": 50, "imgsz": 640, "batch": 16})
train_model(yolo, None, None, fit_params={"data": "/abs/path/data.yaml"})
```

FastAPI usage (payload-based)
-----------------------------

```python
from typing import Any, Dict
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from AI.model_trainer import train_from_payload

app = FastAPI()
jobs: Dict[str, Dict[str, Any]] = {}

class TrainPayload(BaseModel):
    model: str  # 'yolov8n' or '/path/model.pt'
    hyperparameters: Dict[str, Any] | None = None
    fit_params: Dict[str, Any] | None = None

def _run(job_id: str, payload: Dict[str, Any]):
    jobs[job_id] = {"status": "running"}
    try:
        _, score = train_from_payload(payload)
        jobs[job_id] = {"status": "completed", "score": score}
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}

@app.post("/train")
def train(payload: TrainPayload, bg: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued"}
    bg.add_task(_run, job_id, payload.dict(exclude_none=True))
    return {"job_id": job_id}
```

Notes
-----

- `X_train`/`y_train` are ignored for YOLO; pass dataset YAML via `fit_params.data`.
- To use a custom model (e.g., YOLOv8 variant or your `model.pt`), set `model` accordingly.
- Trained artifacts are managed by Ultralytics in `runs/` unless overridden.


