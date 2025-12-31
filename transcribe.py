import os
import time
from faster_whisper import WhisperModel
from datetime import timedelta
import multiprocessing

# ================= KONFIGURACJA TURBO =================
FOLDER_WEJSCIOWY = "nagrania" 
ROZSZERZENIA = ('.mp3', '.wav', '.mp4', '.m4a', '.flac', '.mov', '.mkv', '.avi', '.wma', '.aac')

# ZMIANA 1: Model 'medium' zamiast 'large-v2'. 
# Dla CPU to najlepszy kompromis. Jeśli chcesz, zmień z powrotem na 'large-v2'.
# Możesz też spróbować nowego: "deepdml/faster-whisper-large-v3-turbo-ct2"
MODEL_SIZE = "medium" 

# ZMIANA 2: Liczba wątków procesora (Używamy wszystkich dostępnych rdzeni)
CPU_THREADS = multiprocessing.cpu_count()
# ======================================================

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def process_folder_turbo():
    if not os.path.exists(FOLDER_WEJSCIOWY):
        os.makedirs(FOLDER_WEJSCIOWY)
        print(f"📁 Stworzono folder '{FOLDER_WEJSCIOWY}'. Wrzuć tam pliki!")
        return

    pliki = [f for f in os.listdir(FOLDER_WEJSCIOWY) if f.lower().endswith(ROZSZERZENIA)]
    
    if not pliki:
        print(f"❌ Pusto w folderze '{FOLDER_WEJSCIOWY}'.")
        return

    print(f"🚀 Ładowanie modelu {MODEL_SIZE} (INT8) na {CPU_THREADS} wątkach...")
    
    try:
        model = WhisperModel(
            MODEL_SIZE, 
            device="cpu", 
            compute_type="int8", 
            cpu_threads=CPU_THREADS,
            num_workers=1
        )
    except Exception as e:
        print(f"❌ Błąd modelu: {e}")
        return

    print(f"🔥 TRYB TURBO WŁĄCZONY (Beam Size = 1)")
    print(f"📂 Znaleziono {len(pliki)} plików.")

    for i, plik in enumerate(pliki, 1):
        pelna_sciezka = os.path.join(FOLDER_WEJSCIOWY, plik)
        srt_sciezka = os.path.join(FOLDER_WEJSCIOWY, os.path.splitext(plik)[0] + ".srt")
        
        if os.path.exists(srt_sciezka):
            print(f"⏭️  Pominięto: {plik}")
            continue

        print(f"\n[{i}/{len(pliki)}] 🎙️  Przetwarzanie: {plik} ...")
        start_time = time.time()
        
        try:
            # ZMIANA 3: beam_size=1 (Klucz do szybkości!)
            # best_of=1 wyłącza szukanie alternatyw
            # temperature=0 zmniejsza halucynacje
            segments, info = model.transcribe(
                pelna_sciezka, 
                beam_size=1,        # <--- TU JEST PRZYSPIESZENIE
                best_of=1,
                temperature=0,
                language="pl", 
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500) # Ignoruj ciszę krótszą niż 0.5s
            )
            
            with open(srt_sciezka, "w", encoding="utf-8") as f:
                count = 1
                for segment in segments:
                    start = format_timestamp(segment.start)
                    end = format_timestamp(segment.end)
                    text = segment.text.strip()
                    
                    # Wyświetlamy tylko kropkę co segment, żeby nie śmiecić w konsoli, ale widzieć postęp
                    print(".", end="", flush=True)
                    
                    f.write(f"{count}\n{start} --> {end}\n{text}\n\n")
                    count += 1
            
            duration = int(time.time() - start_time)
            print(f"\n✅ Gotowe w {duration}s.")

        except Exception as e:
            print(f"\n❌ Błąd: {e}")

    print("\n✨ ZAKOŃCZONO! ✨")

if __name__ == "__main__":
    process_folder_turbo()