#!/usr/bin/env python3
"""
Reshmi Voice Preprocessing Pipeline - FIXED VERSION
Optimized for high-pitched, vulnerable female voice
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter, low_pass_filter
from pydub.silence import split_on_silence

def prepare_reshmi_audio(
    input_files: List[str],
    output_file: str = "voices/reshmi_default.wav",
    target_duration_sec: float = 12.0
) -> Optional[str]:
    """
    Professional audio preprocessing for Reshmi's voice profile
    """
    
    print("=" * 60)
    print("🎤 RESHMI VOICE PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Filter to only specific known files - ignore temp/cache files
    valid_files = []
    allowed_patterns = ["reshmi.mp3", "reshmi.wav", "AUD-20260519-WA0000.mp3", "AUD-20260519-WA0001.mp3"]
    
    for file in input_files:
        file_path = Path(file)
        if file_path.exists():
            # Only include specific files or files that look like original recordings
            if file_path.name in allowed_patterns or not any(x in file_path.name.lower() for x in ["temp", "cache", "output", "reshmi_default"]):
                valid_files.append(file)
                print(f"✓ Found: {file}")
            else:
                print(f"⚠️ Skipping: {file} (possible temp/cache file)")
    
    if not valid_files:
        # Manual fallback - ask user for specific files
        print("\n⚠️ No specific files found. Searching for any audio files...")
        for ext in ['.mp3', '.wav', '.m4a']:
            for f in Path(".").glob(f"*{ext}"):
                if not any(x in f.name.lower() for x in ["temp", "cache", "output", "default"]):
                    valid_files.append(str(f))
                    print(f"✓ Found: {f}")
    
    if not valid_files:
        print("\n❌ ERROR: No valid input files found!")
        print("   Please ensure your audio files are named:")
        print("   - reshmi.mp3")
        print("   - AUD-20260519-WA0000.mp3")
        print("   - AUD-20260519-WA0001.mp3")
        return None
    
    print(f"\n📊 Processing {len(valid_files)} file(s)...")
    
    cleaned_segments = []
    
    for idx, input_file in enumerate(valid_files):
        print(f"\n[{idx + 1}/{len(valid_files)}] Processing: {input_file}")
        
        try:
            audio = AudioSegment.from_file(input_file)
            original_duration = len(audio) / 1000.0
            print(f"   Original duration: {original_duration:.1f}s")
            
            if len(audio) > 500:
                audio = audio[500:]
                print(f"   Trimmed first 500ms")
            
            if audio.channels != 1:
                audio = audio.set_channels(1)
                print(f"   Converted to mono")
            
            if audio.frame_rate != 22050:
                audio = audio.set_frame_rate(22050)
                print(f"   Resampled to 22050Hz")
            
            audio = high_pass_filter(audio, 75)
            print(f"   Applied high-pass filter @ 75Hz")
            
            audio = low_pass_filter(audio, 8500)
            print(f"   Applied low-pass filter @ 8500Hz")
            
            audio = normalize(audio)
            
            chunks = split_on_silence(
                audio,
                min_silence_len=150,
                silence_thresh=-45,
                keep_silence=100
            )
            if chunks:
                audio = sum(chunks)
                print(f"   Removed silence gaps")
            
            if idx > 0:
                audio = audio.fade_in(50)
            
            cleaned_segments.append(audio)
            print(f"   ✅ Cleaned duration: {len(audio)/1000:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error processing {input_file}: {str(e)}")
            continue
    
    if not cleaned_segments:
        print("\n❌ ERROR: No segments were successfully processed!")
        return None
    
    print(f"\n🔗 Stitching {len(cleaned_segments)} segments together...")
    
    combined = cleaned_segments[0]
    for segment in cleaned_segments[1:]:
        natural_breath = AudioSegment.silent(duration=300)
        combined += natural_breath + segment
    
    total_duration = len(combined) / 1000.0
    print(f"   Stitched duration: {total_duration:.1f}s")
    
    if total_duration > 15.0:
        print(f"   ⚠️ Duration too long ({total_duration:.1f}s), trimming to 15s...")
        combined = combined[:15000]
    elif total_duration < 8.0:
        print(f"   ⚠️ Duration too short ({total_duration:.1f}s), applying gentle loop...")
        loop_count = int(12.0 / total_duration) + 1
        combined = combined * loop_count
        combined = combined[:12000]
        print(f"   Looped to {len(combined)/1000:.1f}s")
    
    print(f"\n🎛️ Final mastering...")
    
    combined = high_pass_filter(combined, 75)
    combined = low_pass_filter(combined, 8500)
    combined = normalize(combined, headroom=0.5)
    combined = combined.set_channels(1)
    combined = combined.set_frame_rate(22050)
    
    Path("voices").mkdir(exist_ok=True)
    combined.export(output_file, format="wav")
    
    final_duration = len(combined) / 1000.0
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"📁 Output: {output_file}")
    print(f"⏱️ Duration: {final_duration:.1f} seconds")
    print(f"💾 File size: {file_size_mb:.2f} MB")
    print(f"🎚️ Sample rate: {combined.frame_rate} Hz")
    print(f"🔊 Channels: {combined.channels}")
    print(f"📊 Loudness: {combined.dBFS:.1f} dB")
    print("=" * 60)
    
    return output_file

if __name__ == "__main__":
    # Look for specific files only, not temp/cache files
    target_files = ["reshmi.mp3", "reshmi.wav", "AUD-20260519-WA0000.mp3", "AUD-20260519-WA0001.mp3"]
    audio_files = []
    
    for target in target_files:
        if Path(target).exists():
            audio_files.append(target)
    
    if not audio_files:
        print("\n📁 Searching for audio files in current directory...")
        for ext in ['.mp3', '.wav', '.m4a']:
            for f in Path(".").glob(f"*{ext}"):
                if not any(x in f.name.lower() for x in ["temp", "cache", "output", "default"]):
                    if f.name not in audio_files:
                        audio_files.append(str(f))
    
    if not audio_files:
        print("\n❌ No audio files found!")
        print("\n   Please place your Reshmi audio files in this folder with names like:")
        print("   - reshmi.mp3")
        print("   - AUD-20260519-WA0000.mp3")
        print("   - AUD-20260519-WA0001.mp3")
        print("\n   Or any .mp3/.wav file that doesn't contain 'temp' or 'cache' in name")
        sys.exit(1)
    
    print(f"\n📁 Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"   - {f}")
    
    print("\n" + "-" * 40)
    result = prepare_reshmi_audio(audio_files)
    
    if result:
        print("\n🎤 Ready to start the API server!")
        print("   Run: python server.py")
    else:
        print("\n❌ Preprocessing failed. Check your audio files.")
        sys.exit(1)