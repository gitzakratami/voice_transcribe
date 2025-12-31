import os
import torch
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
from datetime import timedelta

def format_timestamp(seconds):
    """Zamienia sekundy na format SRT (00:00:00,000)"""
    td = timedelta(seconds=seconds)
    # timedelta może zwrócić dni, musimy to obsłużyć
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(result, output_file):
    """Tworzy plik SRT z wyników pipeline'u"""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(result["chunks"], start=1):
            # Czasem pipeline zwraca timestampy jako tuple (start, end)
            start, end = chunk["timestamp"]
            
            # Zabezpieczenie na wypadek braku końca timestampu (ostatnie zdanie)
            if end is None:
                end = start + 2.0 # Dajemy domyślnie 2 sekundy
                
            text = chunk["text"].strip()
            
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

def run_onnx_amd(video_path):
    if not os.path.exists(video_path):
        print("❌ Brak pliku!")
        return

    print(f"--- START NA AMD (ONNX DirectML): {video_path} ---")
    
    model_id = "openai/whisper-large-v2"
    
    print("1. Ładowanie i konwersja modelu na ONNX (to zajmie chwilę za pierwszym razem)...")
    
    try:
        # To jest kluczowy moment. export=True konwertuje model na format zrozumiały dla AMD.
        # provider="DmlExecutionProvider" wymusza użycie Radeona.
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            export=True, 
            provider="DmlExecutionProvider"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        return

    print("2. Uruchamianie pipeline'u na GPU...")
    # Tworzymy potok przetwarzania
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=[6, 0]
    )

    print("3. Rozpoznawanie mowy (Transkrypcja)...")
    # generate_kwargs wymusza język polski
    result = pipe(video_path, generate_kwargs={"language": "polish"})

    # 4. Zapis do SRT
    srt_file = os.path.splitext(video_path)[0] + ".srt"
    print(f"4. Zapisywanie napisów do: {srt_file}")
    
    generate_srt(result, srt_file)
    print("✨ GOTOWE! Sprawdź plik .srt")

if __name__ == "__main__":
    PLIK = "WorldOfTanks_replay_2025.11.11-22.58.mp4"
    run_onnx_amd(PLIK)