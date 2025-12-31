import stable_whisper
import torch_directml
import os

def run_on_amd(video_path):
    if not os.path.exists(video_path):
        print("❌ Brak pliku!")
        return

    print(f"--- START NA GPU AMD (Radeon): {video_path} ---")

    # 1. Konfiguracja DirectML
    try:
        dml = torch_directml.device()
        print("✅ Wykryto GPU AMD przez DirectML.")
    except:
        print("❌ Błąd: Nie wykryto DirectML. Upewnij się, że masz Python 3.10.")
        return

    # 2. Ładowanie modelu
    # Masz 16GB VRAM, więc 'large-v2' wejdzie bez problemu.
    # Jest dużo dokładniejszy niż medium.
    print("1. Ładowanie modelu large-v2 do VRAM...")
    model = stable_whisper.load_model('large-v2', device=dml)

    # 3. Transkrypcja
    print("2. Rozpoznawanie mowy...")
    # WAŻNE: fp16=False. Karty AMD na Windows przez DirectML często
    # wyrzucają błędy przy FP16. FP32 (False) jest stabilne.
    result = model.transcribe(video_path, language='pl', vad=True, fp16=False)

    # 4. Zapis
    srt_file = os.path.splitext(video_path)[0] + ".srt"
    result.to_srt_vtt(srt_file, word_level=False)
    print(f"✨ GOTOWE! Zapisano: {srt_file}")

if __name__ == "__main__":
    # Wpisz nazwę pliku
    PLIK = "videos/WorldOfTanks_replay_2025.11.11-22.58.mp4" 
    run_on_amd(PLIK)