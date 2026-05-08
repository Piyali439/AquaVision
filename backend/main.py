import io, time, logging, os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from fastapi import Form

import numpy as np
import tensorflow as tf
from google import genai  # Use the new SDK
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./coral_data.db")
MODEL_PATH = os.getenv("MODEL_PATH", "coral_classification_final.keras")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0-MobileNetV2")

# ── Gemini Setup ─────────────────────────────────────────────────────────────
# Initialize the Client with specific API versioning
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not found in .env file.")
    client = None
else:
    # Adding the vertex_ai=False and specific configuration 
    # to ensure it uses the stable Google AI gateway
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={'api_version': 'v1beta'} 
    )

# ── Database Setup ──────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "coral_health_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    filename = Column(String)
    prediction_label = Column(String)
    confidence_score = Column(Float)
    inference_speed_ms = Column(Float)
    model_version = Column(String, default=MODEL_VERSION)
    is_verified = Column(Boolean, default=False)
    verified_label = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ── Config & Model ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
IMG_SIZE = (224, 224)
CLASSES = ["bleached_corals", "healthy_corals"]

class ModelStore:
    model: tf.keras.Model = None

store = ModelStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(MODEL_PATH).exists():
        logging.error(f"Model file {MODEL_PATH} not found!")
    else:
        store.model = tf.keras.models.load_model(MODEL_PATH)
    yield
    if store.model:
        del store.model

app = FastAPI(title="Coral AI Consultant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    confidence_pct: str
    probabilities: dict[str, float]
    status: str
    urgency: str
    actions: list[str]
    inference_ms: float
    num_classes: int

# ── Updated Gemini Logic ──────────────────────────────────────────
async def get_gemini_recommendations(status_label: str):
    if not client:
        return f"Fallback: {status_label}", "low", ["Check API Key in .env"]

    prompt = f"""
    Acting as a Marine Biologist, provide a brief health status and 3 actionable 
    recommendations for a coral reef that has been classified as '{status_label.replace('_', ' ')}'.
    The recommendations should be professional, scientific, and concise.
    Format your response exactly like this:
    STATUS: [One sentence status]
    URGENCY: [low/high/critical]
    ACTIONS:
    - [Action 1]
    - [Action 2]
    - [Action 3]
    """
    try:
        # Use the new generate_content syntax

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt
        )
        text = response.text
        status = text.split("STATUS:")[1].split("URGENCY:")[0].strip()
        urgency = text.split("URGENCY:")[1].split("ACTIONS:")[0].strip().lower()
        actions = [a.strip("- ").strip() for a in text.split("ACTIONS:")[1].strip().split("\n") if a.strip()]
        return status, urgency, actions[:3]
    except Exception as e:
        logging.error(f"Gemini SDK Error: {e}")
        # Recruiter-ready professional fallback logic
        if "bleached" in status_label:
            return ("Thermal stress detected. Coral tissues showing significant pigment loss.", 
                    "high", 
                    ["Report to monitoring authority", "Identify cooling zones", "Minimize contact"])
        return ("Coral appears healthy. Maintaining standard baseline conditions.", 
                "low", 
                ["Document GPS coordinates", "Schedule quarterly re-survey", "Inspect for cryptic stressors"])

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    
    if not store.model:
        raise HTTPException(503, "Model not loaded on server.")

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)

    t0 = time.perf_counter()
    raw_preds = store.model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(raw_preds))
    label = CLASSES[idx]
    ms = (time.perf_counter() - t0) * 1000

    status_msg, urgency, actions = await get_gemini_recommendations(label)

    # Save to Database with new location fields
    new_record = PredictionRecord(
        filename=file.filename,
        prediction_label=label,
        confidence_score=round(float(raw_preds[idx]), 4),
        inference_speed_ms=round(ms, 2),
    )
    db.add(new_record)
    db.commit()

    return PredictionResponse(
        label=label,
        confidence=round(float(raw_preds[idx]), 4),
        confidence_pct=f"{float(raw_preds[idx]) * 100:.1f}%",
        probabilities={CLASSES[i]: float(raw_preds[i]) for i in range(len(CLASSES))},
        status=status_msg,
        urgency=urgency,
        actions=actions,
        inference_ms=round(ms, 2),
        num_classes=len(CLASSES)
    )

@app.get("/analytics/stats")
def get_stats(db: Session = Depends(get_db)):
    total_scans = db.query(PredictionRecord).count()
    stats = db.query(PredictionRecord.prediction_label, func.count(PredictionRecord.id)).group_by(PredictionRecord.prediction_label).all()
    stats_dict = {label: count for label, count in stats}
    bleached_count = stats_dict.get("bleached_corals", 0)
    bleaching_rate = (bleached_count / total_scans * 100) if total_scans > 0 else 0
    return {
        "total_scans": total_scans,
        "bleaching_rate": f"{bleaching_rate:.1f}%",
        "health_distribution": stats_dict,
        "last_updated": datetime.utcnow().isoformat()
    }

# Fetch the last 10 records for the UI History Log
@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(10).all()

# Validation Loop: Update a record with the correct label
@app.patch("/verify/{record_id}")
async def verify_record(record_id: int, correct_label: str, db: Session = Depends(get_db)):
    # Guard: only accept valid class names
    if correct_label not in ["healthy_corals", "bleached_corals"]:
        raise HTTPException(400, f"Invalid label '{correct_label}'. Must be one of: healthy_corals, bleached_corals")
    
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(404, "Record not found")
    
    record.is_verified = True
    record.verified_label = correct_label
    db.commit()

    # Tell the frontend whether the model was right or wrong
    was_correct = record.prediction_label == correct_label
    return {"status": "success", "message": f"Record {record_id} verified as {correct_label}", "was_correct": was_correct}