import os
import torch
import uuid
import re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import shutil
import soundfile as sf
import numpy as np

# Set memory optimization for limited VRAM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

app = FastAPI(title="Qwen3-TTS GPU Worker")

# Global model state
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model
    if model is None:
        print(f"Loading Qwen3-TTS (0.6B) on {device}...")
        from qwen_tts import Qwen3TTSModel
        # Using 0.6B to fit in 6GB VRAM along with other processes
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        print("Model loaded successfully.")
    return model

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("russian"),
    reference_audio: UploadFile = File(...)
):
    try:
        # Normalize language codes
        lang_map = {
            "ru": "russian", "en": "english", "zh": "chinese", "de": "german",
            "ja": "japanese", "ko": "korean", "es": "spanish", "fr": "french",
            "it": "italian", "pt": "portuguese"
        }
        normalized_lang = lang_map.get(language.lower(), language.lower())
        
        temp_dir = Path("temp_worker")
        temp_dir.mkdir(exist_ok=True)
        
        # Save reference audio
        ref_path = temp_dir / f"ref_{uuid.uuid4().hex[:8]}.wav"
        with ref_path.open("wb") as buffer:
            shutil.copyfileobj(reference_audio.file, buffer)
            
        output_path = temp_dir / f"out_{uuid.uuid4().hex[:8]}.wav"
        
        tts = load_model()
        
        # Split text into chunks. Prioritize '|' from GPT, then fall back to punctuation.
        # This ensures natural pauses and avoids CUDA OOM.
        segments = [s.strip() for s in text.split('|') if s.strip()]
        sentences = []
        for seg in segments:
            # Further split very long segments by punctuation if necessary
            raw_sentences = re.split(r'(?<=[.!?])\s+', seg)
            sentences.extend([s.strip() for s in raw_sentences if s.strip()])
        
        # If text is long but has no punctuation, split by length
        if not sentences and text.strip():
            sentences = [text.strip()]
        
        all_audio_chunks = []
        final_sample_rate = 24000 # Default for Qwen3-TTS
        
        print(f"Synthesizing {len(sentences)} chunks...")
        
        for i, sentence in enumerate(sentences):
            print(f"Processing chunk {i+1}/{len(sentences)}: {sentence[:30]}...")
            
            # Synthesize single chunk
            audio_list, sample_rate = tts.generate_voice_clone(
                text=sentence,
                language=normalized_lang,
                ref_audio=str(ref_path),
                x_vector_only_mode=True
            )
            final_sample_rate = sample_rate
            
            if isinstance(audio_list, list) and len(audio_list) > 0:
                chunk_audio = np.concatenate(audio_list)
            else:
                chunk_audio = audio_list
            
            all_audio_chunks.append(chunk_audio)
            
            # Clear cache to free up memory
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Merge all chunks
        if not all_audio_chunks:
            raise ValueError("No audio was generated")
            
        full_audio = np.concatenate(all_audio_chunks)
            
        # Save to WAV
        sf.write(str(output_path), full_audio, final_sample_rate)
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"synthesis_{uuid.uuid4().hex[:8]}.wav"
        )
        
    except Exception as e:
        print(f"Synthesis error: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup ref audio if needed, though we might want to keep it
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
