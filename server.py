#!/usr/bin/env python3
"""
Reshmi Voice Clone Production Server - FIXED VERSION
Optimized for vulnerable, high-pitched female voice with emotional prosody
"""

import os
import re
import uuid
import logging
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter, low_pass_filter
from pydub.silence import split_on_silence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    VOICE_DIR = Path("voices")
    OUTPUT_DIR = Path("outputs")
    UPLOAD_DIR = Path("uploads")
    CACHE_DIR = Path("cache")
    
    TEMPERATURE = 0.65
    LENGTH_PENALTY_NORMAL = 0.85
    LENGTH_PENALTY_VULNERABLE = 0.78
    REPETITION_PENALTY = 2.5
    TOP_K = 50
    TOP_P = 0.92
    
    TARGET_SAMPLE_RATE = 22050
    HIGH_PASS_FREQ = 75
    LOW_PASS_FREQ = 8500
    CROSSFADE_DURATION_MS = 150
    MIN_CHUNK_CHARS = 100
    MAX_CHUNK_CHARS = 250
    
    DEFAULT_VOICE_ID = "reshmi_default"
    DEFAULT_SPEED = 0.85
    DEFAULT_VULNERABILITY = 9
    DEFAULT_EMOTION = "vulnerable"
    
    for d in [VOICE_DIR, OUTPUT_DIR, UPLOAD_DIR, CACHE_DIR]:
        d.mkdir(exist_ok=True)

class ProsodyEngine:
    def __init__(self):
        self.vulnerability_markers = [
            (r'\b(please)\b', r'\g<0>...'),
            (r'\b(just)\b', r'\g<0>...'),
            (r'\b(maybe)\b', r'\g<0>...'),
            (r'\b(perhaps)\b', r'\g<0>...'),
            (r'\b(honestly)\b', r'\g<0>...'),
            (r'\b(truly)\b', r'\g<0>...'),
            (r'\b(really)\b', r'\g<0>...'),
            (r'\b(i feel)\b', r'\g<0>...'),
            (r'\b(i think)\b', r'\g<0>...'),
            (r'\b(i believe)\b', r'\g<0>...'),
            (r'\b(i wish)\b', r'\g<0>...'),
            (r'\b(i hope)\b', r'\g<0>...'),
            (r'\b(you know)\b', r'\g<0>...'),
            (r'\b(like)\b', r'\g<0>...'),
            (r'\b(well)\b', r'\g<0>...'),
            (r'\b(so)\b', r'\g<0>...'),
            (r'\b(actually)\b', r'\g<0>...')
        ]
        
        self.emotion_profiles = {
            "vulnerable": {"ellipses_rate": 0.75, "breath_pauses": True, "slow_factor": 0.82},
            "pleading": {"ellipses_rate": 0.85, "breath_pauses": True, "slow_factor": 0.78},
            "sad": {"ellipses_rate": 0.70, "breath_pauses": True, "slow_factor": 0.85},
            "happy": {"ellipses_rate": 0.30, "breath_pauses": False, "slow_factor": 0.95},
            "neutral": {"ellipses_rate": 0.20, "breath_pauses": False, "slow_factor": 1.0}
        }
    
    def process_emotional_text(self, text: str, emotion: Optional[str] = None, speed: float = 0.85, vulnerability_level: int = 9) -> str:
        original_text = text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        profile = self.emotion_profiles.get(emotion.lower() if emotion else "neutral", self.emotion_profiles["neutral"])
        
        if vulnerability_level >= 5:
            marker_count = min(len(self.vulnerability_markers), vulnerability_level)
            for i in range(marker_count):
                pattern, replacement = self.vulnerability_markers[i]
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if profile["ellipses_rate"] > 0.3:
            sentences = re.split(r'([.!?])', text)
            processed_sentences = []
            for i in range(0, len(sentences), 2):
                sent = sentences[i].strip()
                punct = sentences[i+1] if i+1 < len(sentences) else ''
                if sent and len(sent) > 15:
                    sent = sent.rstrip() + '...'
                processed_sentences.append(sent + punct)
            text = ' '.join(processed_sentences)
        
        text = re.sub(r'([.!?])\s+(?=[A-Z])', r'\1\n\n', text)
        text = re.sub(r'([.!?])$', r'\1...', text)
        
        if speed < 0.9 or profile["slow_factor"] < 0.9:
            text = re.sub(r'\.', '...', text)
            text = re.sub(r'\!', '...!', text)
            text = re.sub(r'\?', '...?', text)
            text = re.sub(r',', ', ', text)
        
        if profile["breath_pauses"] and vulnerability_level >= 7:
            text = re.sub(r'\n\n', '\n\n*breath*\n\n', text)
        
        logger.debug(f"Prosody: {len(original_text)} -> {len(text)} chars")
        return text
    
    def get_length_penalty(self, emotion: Optional[str]) -> float:
        if emotion and emotion.lower() in ["vulnerable", "pleading"]:
            return Config.LENGTH_PENALTY_VULNERABLE
        return Config.LENGTH_PENALTY_NORMAL

class TextChunker:
    def __init__(self, max_chunk_chars: int = 250, min_chunk_chars: int = 100):
        self.max_chunk = max_chunk_chars
        self.min_chunk = min_chunk_chars
    
    def split_by_semantic_boundaries(self, text: str) -> List[str]:
        text = text.replace('\n', ' ').replace('\r', ' ')
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if len(test_chunk) <= self.max_chunk:
                current_chunk = test_chunk
            else:
                if current_chunk and len(current_chunk) >= self.min_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk and len(current_chunk) >= self.min_chunk:
            chunks.append(current_chunk.strip())
        elif current_chunk and chunks:
            chunks[-1] = chunks[-1] + " " + current_chunk
        
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk:
                split_point = chunk.rfind(' ', 0, self.max_chunk)
                if split_point == -1:
                    split_point = self.max_chunk
                final_chunks.append(chunk[:split_point])
                final_chunks.append(chunk[split_point:])
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Chunked text into {len(final_chunks)} parts")
        return final_chunks
    
    def crossfade_audio_segments(self, segments: List[AudioSegment], crossfade_ms: int = 150) -> AudioSegment:
        if not segments:
            return AudioSegment.empty()
        if len(segments) == 1:
            return segments[0]
        
        merged = segments[0]
        for next_segment in segments[1:]:
            tail = merged[-crossfade_ms:]
            head = next_segment[:crossfade_ms]
            tail = tail.fade_out(crossfade_ms)
            head = head.fade_in(crossfade_ms)
            overlapped = tail.overlay(head, position=len(merged) - crossfade_ms)
            merged = merged[:-crossfade_ms] + overlapped + next_segment[crossfade_ms:]
        return merged

def master_audio_for_reshmi(audio: AudioSegment) -> AudioSegment:
    audio = high_pass_filter(audio, Config.HIGH_PASS_FREQ)
    audio = low_pass_filter(audio, Config.LOW_PASS_FREQ)
    audio = normalize(audio, headroom=0.5)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(Config.TARGET_SAMPLE_RATE)
    return audio

def preprocess_uploaded_voice(file_path: Path, output_path: Path) -> bool:
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(Config.TARGET_SAMPLE_RATE)
        audio = master_audio_for_reshmi(audio)
        chunks = split_on_silence(audio, min_silence_len=150, silence_thresh=-45, keep_silence=100)
        if chunks:
            audio = sum(chunks)
        if len(audio) > 15000:
            audio = audio[:15000]
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        logger.error(f"Preprocess failed: {e}")
        return False

# FIXED: Proper lifespan signature with correct typing
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Safely load and unload XTTS model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing XTTS-v2 on {device.upper()}...")
    
    try:
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        app.state.tts_model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(device == "cuda")
        )
        logger.info("✅ XTTS-v2 model loaded successfully")
        
        app.state.prosody_engine = ProsodyEngine()
        app.state.text_chunker = TextChunker()
        
    except Exception as e:
        logger.critical(f"❌ Failed to load model: {str(e)}")
        raise e
    
    yield
    
    if hasattr(app.state, "tts_model"):
        del app.state.tts_model
        logger.info("Model unloaded")

app = FastAPI(
    title="Reshmi Voice Clone API",
    description="Production voice cloning for high-pitched, vulnerable female voice",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = Config.DEFAULT_VOICE_ID
    language: str = "en"
    speed: float = Field(Config.DEFAULT_SPEED, ge=0.5, le=1.5)
    emotion: Optional[str] = Config.DEFAULT_EMOTION
    vulnerability_level: int = Field(Config.DEFAULT_VULNERABILITY, ge=1, le=10)
    response_format: str = Field("mp3")

class VoiceUploadResponse(BaseModel):
    voice_id: str
    message: str
    duration_seconds: float
    sample_rate: int

def cleanup_temp_files(files: List[Path]):
    for file in files:
        try:
            if file and file.exists():
                file.unlink()
        except Exception:
            pass

@app.post("/v1/tts/ultimate")
async def ultimate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    voice_path = Config.VOICE_DIR / f"{request.voice_id}.wav"
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice_id}' not found")
    
    processed_text = app.state.prosody_engine.process_emotional_text(
        text=request.text,
        emotion=request.emotion,
        speed=request.speed,
        vulnerability_level=request.vulnerability_level
    )
    
    chunks = app.state.text_chunker.split_by_semantic_boundaries(processed_text)
    
    length_penalty = app.state.prosody_engine.get_length_penalty(request.emotion)
    
    inference_params = {
        'temperature': Config.TEMPERATURE,
        'length_penalty': length_penalty,
        'repetition_penalty': Config.REPETITION_PENALTY,
        'top_k': Config.TOP_K,
        'top_p': Config.TOP_P
    }
    
    logger.info(f"Generating {len(chunks)} chunks")
    
    audio_segments = []
    temp_files = []
    
    try:
        model = app.state.tts_model
        
        for i, chunk in enumerate(chunks):
            temp_file = Config.CACHE_DIR / f"temp_{uuid.uuid4().hex}_{i}.wav"
            temp_files.append(temp_file)
            
            await asyncio.to_thread(
                model.tts_to_file,
                text=chunk,
                speaker_wav=str(voice_path),
                language=request.language,
                file_path=str(temp_file),
                **inference_params
            )
            
            audio = AudioSegment.from_wav(temp_file)
            audio = master_audio_for_reshmi(audio)
            audio_segments.append(audio)
        
        if len(audio_segments) > 1:
            merged_audio = app.state.text_chunker.crossfade_audio_segments(audio_segments)
        else:
            merged_audio = audio_segments[0]
        
        merged_audio = master_audio_for_reshmi(merged_audio)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.response_format}") as tmp:
            if request.response_format == "mp3":
                merged_audio.export(tmp.name, format="mp3", bitrate="192k")
            else:
                merged_audio.export(tmp.name, format="wav")
            output_path = Path(tmp.name)
        
        background_tasks.add_task(cleanup_temp_files, temp_files)
        background_tasks.add_task(lambda: output_path.unlink() if output_path.exists() else None)
        
        def iterfile():
            with open(output_path, "rb") as f:
                yield from f
        
        media_type = "audio/mpeg" if request.response_format == "mp3" else "audio/wav"
        
        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers={
                "X-Duration-Seconds": str(len(merged_audio) / 1000),
                "X-Chunks-Processed": str(len(chunks))
            }
        )
        
    except Exception as e:
        background_tasks.add_task(cleanup_temp_files, temp_files)
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/voices/upload", response_model=VoiceUploadResponse)
async def upload_voice(audio_file: UploadFile = File(...)):
    allowed = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    ext = Path(audio_file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Use: {', '.join(allowed)}")
    
    temp_input = Config.UPLOAD_DIR / f"raw_{uuid.uuid4().hex}{ext}"
    content = await audio_file.read()
    with open(temp_input, "wb") as f:
        f.write(content)
    
    voice_id = f"reshmi_{uuid.uuid4().hex[:8]}"
    output_path = Config.VOICE_DIR / f"{voice_id}.wav"
    
    success = preprocess_uploaded_voice(temp_input, output_path)
    temp_input.unlink()
    
    if not success:
        raise HTTPException(400, "Failed to process audio")
    
    audio = AudioSegment.from_file(output_path)
    duration = len(audio) / 1000.0
    
    return VoiceUploadResponse(
        voice_id=voice_id,
        message="Voice uploaded successfully",
        duration_seconds=round(duration, 2),
        sample_rate=audio.frame_rate
    )

@app.get("/v1/voices/list")
async def list_voices():
    voices = []
    for f in Config.VOICE_DIR.glob("*.wav"):
        try:
            audio = AudioSegment.from_file(f)
            voices.append({
                "voice_id": f.stem,
                "duration_seconds": round(len(audio)/1000, 2),
                "file_size_mb": round(f.stat().st_size/(1024*1024), 2)
            })
        except Exception:
            continue
    return {"voices": voices, "total": len(voices)}

@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    if voice_id == Config.DEFAULT_VOICE_ID:
        raise HTTPException(400, "Cannot delete default voice")
    path = Config.VOICE_DIR / f"{voice_id}.wav"
    if not path.exists():
        raise HTTPException(404, "Voice not found")
    path.unlink()
    return {"success": True}

@app.get("/v1/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": hasattr(app.state, "tts_model"),
        "gpu_available": torch.cuda.is_available(),
        "voices_available": len(list(Config.VOICE_DIR.glob("*.wav"))),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.get("/")
async def root():
    return {
        "service": "Reshmi Voice Clone API",
        "docs": "/docs",
        "health": "/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎤 RESHMI VOICE CLONE SERVER")
    print("=" * 60)
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"High-Pass: {Config.HIGH_PASS_FREQ}Hz")
    print(f"Low-Pass: {Config.LOW_PASS_FREQ}Hz")
    print("=" * 60)
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )