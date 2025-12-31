import stable_whisper
import os

def generate_subtitles_only(video_path, model_size='medium'):
    # Sprawdzanie czy plik istnieje
    if not os.path.exists(video_path):
        print(f"❌ Błąd: Nie znaleziono pliku '{video_path}'")
        return

    print(f"--- ANALIZA PLIKU: {video_path} ---")
    
    # 1. Ładowanie modelu
    # 'medium' to dobry balans między szybkością a dokładnością dla polskiego
    # 'large-v2' jest najdokładniejszy, ale wolniejszy
    print(f"1. Ładuję model AI ({model_size})...")
    model = stable_whisper.load_model(model_size)
    
    # 2. Transkrypcja
    print("2. Rozpoznaję mowę (AI słucha)...")
    
    # vad=True usuwa ciszę (Voice Activity Detection)
    # language='pl' wymusza polski
    result = model.transcribe(video_path, language='pl', vad=True)
    
    # 3. Zapis do pliku .srt
    # Tworzymy nazwę pliku wyjściowego taką samą jak wideo, ale z końcówką .srt
    srt_filename = os.path.splitext(video_path)[0] + ".srt"
    
    print(f"3. Zapisuję napisy do: {srt_filename}")
    
    # word_level=False sprawia, że napisy wyglądają "normalnie" (całe zdania).
    # Jeśli dasz True, będzie to wyglądać jak karaoke (pojedyncze słowa).
    result.to_srt_vtt(srt_filename, word_level=False) 
    
    print("✨ GOTOWE! Możesz otworzyć film w VLC.")

# Uruchomienie
if __name__ == "__main__":
    # Wpisz nazwę swojego pliku wideo
    MOJE_WIDEO = "videos/WorldOfTanks_replay_2025.11.11-22.58.mp4"
    
    generate_subtitles_only(MOJE_WIDEO, model_size='medium')