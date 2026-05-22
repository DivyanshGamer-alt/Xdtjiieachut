#!/usr/bin/env python3
"""
Reshmi Voice Clone Server - Production Ready
Auto file cleanup + GET endpoint + Memory optimized
"""

import os
import uuid
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    VOICE_DIR = Path("voices")
    CACHE_DIR = Path("cache")
    TEMPERATURE = 0.65
    LENGTH_PENALTY = 0.78
    REPETITION_PENALTY = 2.5
    TOP_K = 50
    TOP_P = 0.92
    HIGH_PASS_FREQ = 75
    LOW_PASS_FREQ = 8500
    
    for d in [VOICE_DIR, CACHE_DIR]:
        d.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading XTTS-v2 on [{device.upper()}]...")
    os.environ["COQUI_TOS_AGREED"] = "1"
    app.state.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    logger.info("✅ Model Ready!")
    yield
    if hasattr(app.state, "tts_model"):
        del app.state.tts_model

app = FastAPI(title="Reshmi Voice API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def master_audio(audio: AudioSegment) -> AudioSegment:
    audio = high_pass_filter(audio, Config.HIGH_PASS_FREQ)
    audio = low_pass_filter(audio, Config.LOW_PASS_FREQ)
    audio = normalize(audio, headroom=0.5)
    audio = audio.set_channels(1).set_frame_rate(22050)
    return audio

def delete_file(file_path: str):
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"🗑️ Deleted: {file_path}")
    except Exception as e:
        logger.error(f"Delete failed: {e}")

@app.get("/speak")
async def speak_get(
    background_tasks: BackgroundTasks,
    text: str = Query(..., description="Text to speak"),
    emotion: str = Query("vulnerable", description="vulnerable, sad, happy"),
    speed: float = Query(0.85, ge=0.5, le=1.5)
):
    voice_path = Config.VOICE_DIR / "reshmi_default.wav"
    
    if not voice_path.exists():
        raise HTTPException(404, "Voice not found. Run: python3 preprocess_audio.py")
    
    if emotion == "vulnerable":
        text = text.replace(".", "...").replace("?", "...?") + "..."
    
    unique_id = uuid.uuid4().hex
    raw_wav = Config.CACHE_DIR / f"raw_{unique_id}.wav"
    final_mp3 = Config.CACHE_DIR / f"final_{unique_id}.mp3"
    
    try:
        await asyncio.to_thread(
            app.state.tts_model.tts_to_file,
            text=text,
            speaker_wav=str(voice_path),
            language="en",
            file_path=str(raw_wav),
            temperature=Config.TEMPERATURE,
            length_penalty=Config.LENGTH_PENALTY,
            repetition_penalty=Config.REPETITION_PENALTY,
            top_k=Config.TOP_K,
            top_p=Config.TOP_P
        )
        
        audio = AudioSegment.from_wav(raw_wav)
        audio = master_audio(audio)
        audio.export(final_mp3, format="mp3", bitrate="192k")
        
        background_tasks.add_task(delete_file, str(raw_wav))
        background_tasks.add_task(delete_file, str(final_mp3))
        
        return FileResponse(
            final_mp3,
            media_type="audio/mpeg",
            filename="reshmi_voice.mp3"
        )
        
    except Exception as e:
        delete_file(str(raw_wav))
        delete_file(str(final_mp3))
        logger.error(f"Failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_loaded": hasattr(app.state, "tts_model"),
        "voice_ready": (Config.VOICE_DIR / "reshmi_default.wav").exists(),
        "gpu": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🎤 Reshmi Voice API Server")
    print("=" * 50)
    print("Usage: http://localhost:8000/speak?text=Hello")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
