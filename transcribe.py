import os
import time
from faster_whisper import WhisperModel
from datetime import timedelta

# ================= KONFIGURACJA =================
# Nazwa folderu, w którym trzymasz pliki do przerobienia
FOLDER_WEJSCIOWY = "nagrania" 

# Jakie formaty ma łapać program?
ROZSZERZENIA = ('.mp3', '.wav', '.mp4', '.m4a', '.flac', '.mov', '.mkv', '.avi', '.wma', '.aac')

# Model: 'large-v2' (dokładny) lub 'medium' (szybki)
MODEL_SIZE = "large-v2"
# ================================================

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def process_folder():
    # 1. Sprawdzamy czy folder istnieje
    if not os.path.exists(FOLDER_WEJSCIOWY):
        os.makedirs(FOLDER_WEJSCIOWY)
        print(f"📁 Stworzono folder '{FOLDER_WEJSCIOWY}'. Wrzuć tam pliki i uruchom program ponownie!")
        return

    # 2. Szukamy plików
    pliki = [f for f in os.listdir(FOLDER_WEJSCIOWY) if f.lower().endswith(ROZSZERZENIA)]
    
    if not pliki:
        print(f"❌ Pusto w folderze '{FOLDER_WEJSCIOWY}'. Wrzuć jakieś mp3/mp4/wav.")
        return

    print(f"📂 Znaleziono {len(pliki)} plików do przetworzenia.")
    print(f"🚀 Ładowanie modelu {MODEL_SIZE} (INT8)... (Tylko raz)")
    
    try:
        # Ładujemy model przed pętlą, żeby nie tracić czasu przy każdym pliku
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        return

    # 3. Pętla po plikach
    for i, plik in enumerate(pliki, 1):
        pelna_sciezka = os.path.join(FOLDER_WEJSCIOWY, plik)
        srt_sciezka = os.path.join(FOLDER_WEJSCIOWY, os.path.splitext(plik)[0] + ".srt")
        
        # Sprawdzamy, czy napisów już nie ma (żeby nie robić 2 razy tego samego)
        if os.path.exists(srt_sciezka):
            print(f"⏭️  Pominięto: {plik} (plik .srt już istnieje)")
            continue

        print(f"\n[{i}/{len(pliki)}] 🎙️  Przetwarzanie: {plik} ...")
        start_time = time.time()
        
        try:
            # Transkrypcja
            segments, info = model.transcribe(pelna_sciezka, beam_size=5, language="pl", vad_filter=True)
            
            # Zapis do pliku
            with open(srt_sciezka, "w", encoding="utf-8") as f:
                count = 1
                for segment in segments:
                    start = format_timestamp(segment.start)
                    end = format_timestamp(segment.end)
                    text = segment.text.strip()
                    
                    # Wypisz w konsoli (żebyś wiedział że żyje)
                    print(f"   -> {text}")
                    
                    f.write(f"{count}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
                    count += 1
            
            duration = int(time.time() - start_time)
            print(f"✅ Gotowe w {duration}s. Zapisano: {srt_sciezka}")

        except Exception as e:
            print(f"❌ Błąd przy pliku {plik}: {e}")

    print("\n✨ WSZYSTKIE ZADANIA UKOŃCZONE! ✨")

if __name__ == "__main__":
    process_folder()