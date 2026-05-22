#!/usr/bin/env python3
"""
Reshmi Voice Preprocessing - Production Optimized
Finds original audio clips and masters them at 8500Hz for high-pitch clarity.
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter, low_pass_filter
from pydub.silence import split_on_silence

def prepare_reshmi_audio(output_file="voices/reshmi_default.wav"):
    print("=" * 60)
    print("🎤 RESHMI VOICE PREPROCESSING PIPELINE")
    print("=" * 60)
    
    target_files = ["reshmi.mp3", "reshmi.wav", "AUD-20260519-WA0000.mp3", "AUD-20260519-WA0001.mp3"]
    valid_files = [f for f in target_files if Path(f).exists()]
    
    if not valid_files:
        for ext in ['.mp3', '.wav', '.m4a']:
            for f in Path(".").glob(f"*{ext}"):
                if not any(x in f.name.lower() for x in ["temp", "cache", "output", "default"]):
                    valid_files.append(str(f))
                    
    if not valid_files:
        print("\n❌ Error: No valid audio source files found!")
        print("   Place reshmi.mp3 or AUD-20260519-WA0000.mp3 in this folder")
        return None
        
    print(f"📊 Found {len(valid_files)} file(s) for building profile...")
    cleaned_segments = []
    
    for idx, input_file in enumerate(valid_files):
        try:
            print(f"   [{idx+1}] Processing: {input_file}")
            audio = AudioSegment.from_file(input_file)
            
            if len(audio) > 500:
                audio = audio[500:]
                
            audio = audio.set_channels(1).set_frame_rate(22050)
            audio = high_pass_filter(audio, 75)
            audio = low_pass_filter(audio, 8500)
            audio = normalize(audio)
            
            chunks = split_on_silence(audio, min_silence_len=150, silence_thresh=-45, keep_silence=100)
            if chunks:
                audio = sum(chunks)
            if idx > 0:
                audio = audio.fade_in(50)
            cleaned_segments.append(audio)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    if not cleaned_segments:
        return None
        
    combined = cleaned_segments[0]
    for segment in cleaned_segments[1:]:
        combined += AudioSegment.silent(duration=300) + segment
        
    duration = len(combined) / 1000.0
    if duration > 15.0:
        combined = combined[:15000]
    elif duration < 8.0:
        combined = (combined * 2)[:12000]
        
    combined = high_pass_filter(combined, 75)
    combined = low_pass_filter(combined, 8500)
    combined = normalize(combined, headroom=0.5)
    
    Path("voices").mkdir(exist_ok=True)
    combined.export(output_file, format="wav")
    
    print(f"\n✅ SUCCESS! Saved: {output_file} ({len(combined)/1000:.1f}s)")
    return output_file

if __name__ == "__main__":
    prepare_reshmi_audio()
