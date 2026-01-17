import os
from pathlib import Path

import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from webui.backend.cache import CachePolicy, CacheStore
from webui.backend.feedback import FeedbackExporter
from webui.backend.inference import run_inference
from webui.backend.models_store import ModelConfig, ModelStore
from webui.backend.schemas import FeedbackRequest, InferenceResponse

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("AICHECKER_CONFIG", "configs/default.yaml"))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


config = load_config()
webui_cfg = config.get("webui", {})
thresholds = config.get("thresholds", {})
model_cfg = ModelConfig(
    model_version="v0.1",
    tau_ai=thresholds.get("tau_ai", 0.9),
    tau_ood=thresholds.get("tau_ood", 0.6),
    tau_unknown=thresholds.get("tau_unknown", 0.55),
)
model_store = ModelStore(model_cfg)
cache_policy = CachePolicy(**webui_cfg.get("cache_policy", {}))
cache_store = CacheStore(Path("webui_cache.sqlite"), cache_policy)
feedback_exporter = FeedbackExporter(cache_store)

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")


@app.get("/")
def index() -> HTMLResponse:
    html_path = BASE_DIR / "frontend" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/infer", response_model=InferenceResponse)
async def infer(file: UploadFile = File(...)) -> InferenceResponse:
    img = Image.open(file.file).convert("RGB")
    config = model_store.info()
    result = run_inference(img, config.tau_ai, config.tau_ood, config.tau_unknown)
    cache_store.set(
        {
            "image_hash": result.image_hash,
            "image_bytes": result.image_bytes,
            "prob_ai": result.prob_ai,
            "decision": result.decision,
            "confidence": result.confidence,
            "ood_score": result.ood_score,
            "model_version": config.model_version,
        }
    )
    return InferenceResponse(
        prob_ai=result.prob_ai,
        decision=result.decision,
        confidence=result.confidence,
        ood_score=result.ood_score,
        tau_ai=config.tau_ai,
        model_version=config.model_version,
        tokens=result.tokens,
        image_hash=result.image_hash,
    )


@app.get("/history")
def history() -> list[dict]:
    return cache_store.list_history()


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    if request.label not in {"AI", "NOT AI"}:
        return {"ok": False, "error": "invalid label"}
    cache_store.set_feedback(request.image_hash, request.label)
    return {"ok": True}


@app.get("/export")
def export() -> dict:
    output_dir = Path("feedback_export")
    paths = feedback_exporter.export(output_dir)
    return {"ok": True, "count": len(paths), "dir": str(output_dir)}
