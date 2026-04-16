import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch
import time
import json
import os
import numpy as np
import librosa

# =========================
# DEVICE
# =========================

def detectar_device():
    if torch.cuda.is_available():
        nombre = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {nombre} ({vram:.1f} GB VRAM)")
        return "cuda"
    print("  GPU no disponible, usando CPU")
    return "cpu"

print("\n=== Detectando hardware ===")
DEVICE = detectar_device()

# =========================
# MODELOS
# =========================

from transformers import pipeline as hf_pipeline
from pysentimiento import create_analyzer

print("\n=== Cargando modelos ===")

# Whisper via transformers (compatible ROCm/CUDA/CPU)
print("Cargando Whisper (transformers)...")
t0 = time.time()
whisper_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=DEVICE,
    chunk_length_s=30,
    return_timestamps=True,
    generate_kwargs={"language": "spanish", "task": "transcribe"}
)
print(f"  [OK] {time.time() - t0:.1f}s")

# Voz: modelo de emocion SER
print("Cargando modelo emocion de voz (SER)...")
t0 = time.time()
emotion_pipe = hf_pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=DEVICE
)
print(f"  [OK] {time.time() - t0:.1f}s")

# Texto: pysentimiento
print("Cargando pysentimiento...")
t0 = time.time()
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
print(f"  [OK] {time.time() - t0:.1f}s")

# =========================
# MAPEOS
# =========================

# SER: superb/wav2vec2-base-superb-er devuelve ang/hap/neu/sad
VOZ_A_SCORE = {
    "ang": -1.5,   # enojado
    "sad": -0.8,   # triste
    "neu":  0.0,   # neutral
    "hap":  1.0,   # feliz
}

def score_voz(predicciones: list) -> tuple[float, str]:
    top = max(predicciones, key=lambda x: x["score"])
    label = top["label"]
    score = VOZ_A_SCORE.get(label, 0.0) * top["score"]
    return round(score, 3), label

def score_texto(texto: str) -> tuple[float, str, dict]:
    res = sentiment_analyzer.predict(texto)
    label = res.output
    probas = res.probas

    if label == "NEG":
        return round(-probas.get("NEG", 0), 3), label, probas
    if label == "POS":
        return round(probas.get("POS", 0), 3), label, probas
    return 0.0, "NEU", probas

def fusionar(s_texto: float, s_voz: float) -> tuple[float, str]:
    """
    Texto aporta QUE se dice (60%).
    Voz aporta COMO se dice (40%).
    Si discrepan, se marca como ambiguo.
    """
    final = round(0.6 * s_texto + 0.4 * s_voz, 3)

    mismo_signo = (s_texto < 0 and s_voz < 0) or (s_texto > 0 and s_voz > 0)
    alguno_neutro = s_texto == 0 or s_voz == 0

    if mismo_signo:
        confianza = "alta"
    elif alguno_neutro:
        confianza = "media"
    else:
        confianza = "baja"   # posible sarcasmo o tension

    return final, confianza

def clasificar(score: float) -> str:
    if score <= -1.2:   return "muy negativo"
    if score <= -0.4:   return "negativo"
    if score >= 1.2:    return "muy positivo"
    if score >= 0.4:    return "positivo"
    return "neutral"

# =========================
# PROCESAMIENTO
# =========================

def procesar_audio(ruta_audio: str) -> dict:

    print(f"\n=== Procesando: {ruta_audio} ===")

    # Cargar audio a 16kHz (requerido por wav2vec2)
    print("Cargando audio...")
    t0 = time.time()
    y, sr = librosa.load(ruta_audio, sr=16000, mono=True)
    duracion = len(y) / sr
    print(f"  [OK] {duracion:.1f}s de audio, cargado en {time.time()-t0:.1f}s")

    # Transcripcion
    print("Transcribiendo...")
    t0 = time.time()
    resultado_whisper = whisper_pipe(ruta_audio)
    chunks = resultado_whisper.get("chunks", [])
    print(f"  [OK] {len(chunks)} segmentos en {time.time()-t0:.1f}s")

    if not chunks:
        print("  Sin segmentos detectados.")
        return {}

    # Procesar cada segmento
    print(f"Analizando {len(chunks)} segmentos...")
    t0 = time.time()
    segmentos = []

    for i, chunk in enumerate(chunks):
        texto = chunk["text"].strip()
        if not texto:
            continue

        ts = chunk.get("timestamp", (0, duracion))
        inicio = ts[0] or 0
        fin = ts[1] or duracion

        # Audio del segmento
        i_start = int(inicio * sr)
        i_end = int(fin * sr)
        audio_seg = y[i_start:i_end]

        if len(audio_seg) < sr * 0.5:   # segmentos muy cortos: saltar SER
            s_voz, label_voz = 0.0, "neu"
        else:
            preds = emotion_pipe({"array": audio_seg, "sampling_rate": sr})
            s_voz, label_voz = score_voz(preds)

        s_txt, label_txt, probas = score_texto(texto)
        s_final, confianza = fusionar(s_txt, s_voz)
        emocion = clasificar(s_final)

        segmentos.append({
            "inicio": round(inicio, 2),
            "fin": round(fin, 2),
            "texto": texto,
            "score_texto": s_txt,
            "label_texto": label_txt,
            "score_voz": s_voz,
            "label_voz": label_voz,
            "score_final": s_final,
            "confianza": confianza,
            "emocion": emocion
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(chunks)} segmentos procesados...")

    t_analisis = time.time() - t0
    print(f"  [OK] Analisis en {t_analisis:.1f}s")

    # Emocion global
    scores = [s["score_final"] for s in segmentos]
    score_global = round(float(np.mean(scores)), 3) if scores else 0.0
    emocion_global = clasificar(score_global)

    discrepancias = sum(1 for s in segmentos if s["confianza"] == "baja")

    return {
        "metadata": {
            "duracion_audio": round(duracion, 2),
            "segmentos_totales": len(segmentos),
            "discrepancias_texto_voz": discrepancias,
            "device": DEVICE
        },
        "score_global": score_global,
        "emocion_global": emocion_global,
        "segmentos": segmentos
    }

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    ruta_audio = "audios/audio_largo.mp3"

    t_total = time.time()
    resultado = procesar_audio(ruta_audio)
    t_total = time.time() - t_total

    if resultado:
        os.makedirs("outputs", exist_ok=True)
        nombre = os.path.splitext(os.path.basename(ruta_audio))[0]
        salida = f"outputs/{nombre}_gpu_test.json"

        with open(salida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=4, ensure_ascii=False)

        print(f"\n=== Resultado ===")
        print(f"  Emocion global : {resultado['emocion_global']} (score: {resultado['score_global']})")
        print(f"  Segmentos      : {resultado['metadata']['segmentos_totales']}")
        print(f"  Discrepancias  : {resultado['metadata']['discrepancias_texto_voz']} (texto vs voz)")
        print(f"  Tiempo total   : {t_total:.1f}s")
        print(f"  Guardado en    : {salida}")
