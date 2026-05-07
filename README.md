# AquaVision 🪸
### Coral Reef Health Monitoring System

> Hybrid AI pipeline combining a fine-tuned MobileNetV2 vision model with Gemini-2.5-Flash-Lite ecological reasoning — built for real-world reef monitoring.

![AquaVision UI](./screenshot.png)

---

## What It Does

AquaVision lets a researcher or diver upload a coral reef image and instantly receive:

- A **binary classification** — Healthy or Bleached — from a fine-tuned MobileNetV2 model
- **Per-class confidence probabilities** from the vision model
- **AI-generated ecological recommendations** from Gemini-2.5-Flash-Lite (urgency level + 3 expert actions)
- **GPS coordinates** captured from the browser at scan time
- A **persistent log** of all scans with a human-in-the-loop verification workflow
- A live **analytics dashboard** showing total scans, bleaching rate, and verified count

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                    │
│  Upload → Analyse → Results + Geospatial History   │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP (FormData + lat/lon)
                        ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)               │
│                                                     │
│  /predict       → TF model inference + Gemini call  │
│  /analytics     → Bleaching rate, scan counts       │
│  /history       → Last 10 records from DB           │
│  /verify/{id}   → Human validation loop (PATCH)     │
└──────────┬───────────────────────┬──────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────┐   ┌───────────────────────────┐
│  TF/Keras Model  │   │  SQLite (coral_data.db)   │
│  MobileNetV2     │   │  Predictions + GPS + Tags │
│  .keras file     │   └───────────────────────────┘
└──────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Google Gemini 2.5-Flash-Lite │
│  Ecological recommendations  │
└──────────────────────────────┘
```

---

## Project Structure

```
aquavision/
│
├── training/                        # Google Colab training pipeline
│   └── coral_training_pipeline.py   # Full MobileNetV2 fine-tune script
│
├── backend/
│   ├── main.py                      # FastAPI app — all endpoints
│   ├── coral_classification_final.keras  # Trained model (not in git)
│   ├── coral_data.db                # SQLite database (auto-created)
│   └── .env                         # API keys & config (not in git)
│
└── frontend/
    └── src/
        ├── App.jsx                  # Main UI — upload, analyse, results
        └── components/
            ├── Header.jsx
            ├── Footer.jsx
            └── CoralLogo.jsx
```

---

## Model Training

The model was trained on the [Corals Classification dataset](https://www.kaggle.com/datasets/aneeshdighe/corals-classification) from Kaggle.

**Pipeline (Google Colab):**

| Stage | Detail |
|---|---|
| Base model | MobileNetV2 (ImageNet weights, top removed) |
| Head | GlobalAveragePooling → BatchNorm → Dense(256) → Dropout(0.4) → Softmax(2) |
| Stage A | Head-only training, base frozen — 3 epochs, lr=1e-3 |
| Stage B | Fine-tune top 30 layers — 5 epochs, lr=1e-5 |
| Augmentation | Flip, rotation, zoom, brightness shift (3× augment per image) |
| Class weighting | `sklearn` balanced weights to handle class imbalance |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

**Dataset split used from source:**

| Split | Healthy | Bleached |
|---|---|---|
| Train | 3,504 | 3,880 |
| Val | 500 | 485 |
| Test | 438 | 485 |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Google AI Studio](https://aistudio.google.com/) API key for Gemini

---

### 1. Train the Model (Google Colab)

Open a Colab notebook and run the two cells:

**Cell 1 — Download dataset:**
```python
import kagglehub, os
path = kagglehub.dataset_download("aneeshdighe/corals-classification")
KAGGLE_INPUT = path
```

**Cell 2 — Full training pipeline:**
```python
# paste coral_training_pipeline.py contents here
```

The trained model saves to `/content/coral_model.h5`. Download it from the Colab Files panel.

---

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install fastapi uvicorn tensorflow pillow sqlalchemy \
            python-dotenv google-genai pydantic python-multipart
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
MODEL_PATH=coral_classification_final.keras
MODEL_VERSION=1.0-MobileNetV2
LOCATION_TAG=SEC-A REEF
DATABASE_URL=sqlite:///./coral_data.db
```

Place your downloaded `.keras` (or `.h5`) model file in the `backend/` folder, then start the server:

```bash
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

App runs at: `http://localhost:3000`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload image (+ optional lat/lon) → classification + Gemini recommendations |
| `GET` | `/analytics/stats` | Total scans, bleaching rate, health distribution |
| `GET` | `/history` | Last 10 prediction records |
| `PATCH` | `/verify/{id}` | Mark a record as verified with correct label |

**Example `/predict` request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@coral_image.jpg" \
  -F "lat=18.4655" \
  -F "lon=-66.1057"
```

**Example response:**
```json
{
  "label": "healthy_corals",
  "confidence": 0.9998,
  "confidence_pct": "99.98%",
  "probabilities": { "bleached_corals": 0.0002, "healthy_corals": 0.9998 },
  "status": "The coral reef exhibits robust coral cover and minimal signs of stress.",
  "urgency": "low",
  "actions": ["Document GPS coordinates", "Schedule quarterly re-survey", "Inspect for cryptic stressors"],
  "inference_ms": 318.44,
  "num_classes": 2
}
```

---

## Key Features

**Hybrid AI** — Vision classification and LLM reasoning run in the same request. Gemini provides marine biology context that a CNN alone cannot.

**Geospatial logging** — The browser captures GPS coordinates at scan time. Every prediction is stored with its real-world location for longitudinal reef tracking.

**Human-in-the-loop validation** — Any scan can be marked as verified via the UI, enabling future supervised retraining with real-world corrections.

**Graceful fallback** — If the Gemini API is unavailable, the backend returns pre-written professional fallback recommendations so the UI never breaks.

**Persistent analytics** — All predictions are stored in SQLite. The dashboard shows live bleaching rate and verified scan count across the full scan history.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio key | required |
| `MODEL_PATH` | Path to `.keras` model file | `coral_classification_final.keras` |
| `MODEL_VERSION` | Version tag stored in DB | `1.0-MobileNetV2` |
| `LOCATION_TAG` | Default reef location label | `Unknown Reef` |
| `DATABASE_URL` | SQLAlchemy DB URL | `sqlite:///./coral_data.db` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | TensorFlow / Keras, MobileNetV2, Google Colab |
| Backend | FastAPI, SQLAlchemy, SQLite |
| AI recommendations | Google Gemini 2.5-Flash-Lite (`google-genai` SDK) |
| Frontend | React, Vite |
| Styling | Inline CSS with monospace dark-mode aesthetic |

---

## Roadmap

- [ ] Multi-class expansion (dead coral, soft coral, algae overgrowth)
- [ ] Interactive map view for geospatial scan history
- [ ] Retraining trigger using verified labels from the validation loop
- [ ] Docker Compose setup for one-command deployment
- [ ] Export scan history as CSV / GeoJSON

---

## Author

Designed and developed by **Piyali Ghosh**

- GitHub: [@Piyali439](https://github.com/Piyali439)

---

## License

MIT License — free to use, modify, and distribute with attribution.
