import os
import time
import subprocess
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
from datetime import timedelta

def extract_audio_to_wav(video_path):
    output_wav = "temp_audio_medium.wav"
    cmd = f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_wav}" -y -loglevel error'
    print(f"🎬 Ekstrakcja audio...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return output_wav
    except:
        print("❌ Błąd FFmpeg")
        return None

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(result, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(result["chunks"], start=1):
            start, end = chunk["timestamp"]
            if end is None: end = start + 2.0
            text = chunk["text"].strip()
            f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n")

def run_gpu_medium(video_path):
    # Zmieniona ścieżka na folder z modelem MEDIUM
    local_model_path = "model_onnx_medium"
    
    if not os.path.exists(local_model_path):
        print("❌ Nie znaleziono folderu 'model_onnx_medium'. Uruchom export_medium.py!")
        return

    audio_path = extract_audio_to_wav(video_path)
    if not audio_path: return

    print(f"--- START NA AMD (MEDIUM): {video_path} ---")
    
    try:
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            local_model_path, 
            provider="DmlExecutionProvider"
        )
        processor = AutoProcessor.from_pretrained(local_model_path)
    except Exception as e:
        print(f"❌ Błąd: {e}")
        return

    print("2. Transkrypcja...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=[6, 0]
    )

    start_time = time.time()
    result = pipe(audio_path, generate_kwargs={"language": "polish"})
    end_time = time.time()

    srt_file = os.path.splitext(video_path)[0] + ".srt"
    generate_srt(result, srt_file)
    
    if os.path.exists(audio_path): os.remove(audio_path)
    print(f"✨ GOTOWE! Czas: {int(end_time - start_time)}s. Plik: {srt_file}")

if __name__ == "__main__":
    PLIK = "Wojciech Cejrowski i woda gazowana [HBfpyArVyOI].mp4"
    run_gpu_medium(PLIK)