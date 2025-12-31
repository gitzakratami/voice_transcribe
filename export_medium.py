from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor
import shutil
import os

print("--- EKSPORTOWANIE MODELU MEDIUM DO ONNX ---")

# Zmieniamy na medium - jest lżejszy i nie wywali sterownika AMD
model_id = "openai/whisper-medium"
output_dir = "model_onnx_medium"

# Sprzątanie poprzedniego folderu jeśli istnieje
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print(f"Pobieranie i konwersja {model_id}...")
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
processor = AutoProcessor.from_pretrained(model_id)

print(f"Zapisywanie do folderu: {output_dir}...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("✅ GOTOWE! Model Medium przygotowany.")