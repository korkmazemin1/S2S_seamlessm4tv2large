from datasets import load_dataset
from pydub import AudioSegment
import io


cv_12 = load_dataset("mozilla-foundation/common_voice_12_0", "tr", split="train", streaming=True)

# İlk örneği al
sample = next(iter(cv_12))

# Ses verisini al
audio_bytes = sample["audio"]["array"]
sample_rate = sample["audio"]["sampling_rate"]

# Pydub ile ses formatına dönüştür
audio = AudioSegment(
    audio_bytes.tobytes(),
    frame_rate=sample_rate,
    sample_width=audio_bytes.dtype.itemsize,
    channels=1
)

# İlk 5 saniyeyi kes
snippet = audio[:5000]  # 5000 ms = 5 saniye

# Dosya adı ve format belirle
file_path = "snippet.wav"
snippet.export(file_path, format="wav")

# Dosya boyutunu hesapla
import os
file_size = os.path.getsize(file_path) / 1024  # KB cinsinden

print(f"Ses dosyası kaydedildi: {file_path}")
print(f"Dosya boyutu: {file_size:.2f} KB")
