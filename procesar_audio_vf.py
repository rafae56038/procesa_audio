import json
import os
import numpy as np
import librosa
from faster_whisper import WhisperModel
from pysentimiento import create_analyzer
from dataclasses import dataclass

# =========================
# ⚙️ CONFIG
# =========================

@dataclass
class Config:
    whisper_model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    idiomas_soportados: tuple = ("es", "ca", "pt")

    # Umbrales texto
    confianza_minima_neg: float = 0.45
    confianza_minima_pos: float = 0.60

config = Config()

# =========================
# 🚀 MODELOS (lazy load)
# =========================

_whisper_model = None
_sentiment_analyzer = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("🔄 Cargando Whisper...")
        _whisper_model = WhisperModel(
            config.whisper_model_size,
            device=config.device,
            compute_type=config.compute_type
        )
    return _whisper_model

def get_sentiment():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        print("🔄 Cargando pysentimiento...")
        _sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
    return _sentiment_analyzer

# =========================
# 🧠 SCORE TEXTO
# =========================

def score_texto(texto: str, idioma: str = "es") -> tuple[float, str, dict]:
    analyzer = get_sentiment()
    resultado = analyzer.predict(texto)

    label = resultado.output
    probas = resultado.probas

    factor_idioma = 1.0 if idioma in config.idiomas_soportados else 0.35

    confianza_neg = probas.get("NEG", 0) * factor_idioma
    confianza_pos = probas.get("POS", 0) * factor_idioma

    if label == "NEG" and confianza_neg >= config.confianza_minima_neg:
        return round(-1.5 * confianza_neg, 3), label, probas

    if label == "POS" and confianza_pos >= config.confianza_minima_pos:
        return round(1.2 * confianza_pos, 3), label, probas

    return 0.0, "NEU", probas

# =========================
# 🎧 FEATURES AUDIO
# =========================

def extraer_features_audio(y, sr):
    energia = float(np.mean(librosa.feature.rms(y=y)))

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    variabilidad_pitch = float(np.std(pitch_vals)) if len(pitch_vals) else 0.0

    return energia, variabilidad_pitch

# =========================
# 🎯 SCORING AUDIO
# =========================

def score_audio(energia: float, variabilidad: float, energia_global: float) -> float:
    if energia_global == 0:
        return 0.0

    intensidad = energia / energia_global
    score = 0.0

    if intensidad > 1.4:
        score += 2.0
    elif intensidad > 1.2:
        score += 1.2
    elif intensidad < 0.7:
        score -= 0.5

    if variabilidad > 50:
        score += 1.5
    elif variabilidad > 30:
        score += 0.8

    return round(min(max(score, 0.0), 2.5), 3)

# =========================
# ⚖️ FUSIÓN (FIX CLAVE)
# =========================

def fusion_score(s_texto: float, s_audio: float) -> float:
    """
    Audio amplifica la emoción del texto (no la contradice)
    """

    if s_texto == 0:
        return round(s_audio * 0.3, 3)

    signo = 1 if s_texto > 0 else -1

    if abs(s_texto) >= 1.0:
        factor = 0.7
    elif abs(s_texto) >= 0.4:
        factor = 0.5
    else:
        factor = 0.3

    return round(s_texto + (signo * s_audio * factor), 3)

# =========================
# 🔥 BOOST AGRESIVIDAD
# =========================

PALABRAS_AGRESIVAS = [
    "maldito",
    "no me hables",
    "qué te pasa",
    "me estás escuchando",
    "no me importa",
]

def ajustar_por_agresividad(texto: str, score: float) -> float:
    texto_l = texto.lower()

    if any(p in texto_l for p in PALABRAS_AGRESIVAS):
        return round(score - 0.6, 3)

    return score

# =========================
# 🧠 CLASIFICACIÓN
# =========================

def clasificar_score(score: float) -> str:
    if score >= 2.2:
        return "enojo"
    elif score >= 1.2:
        return "estresado"
    elif score <= -2.0:
        return "triste"
    elif score <= -0.8:
        return "negativo"
    elif score >= 0.5:
        return "positivo"
    return "neutral"

# =========================
# 🌍 EMOCIÓN GLOBAL
# =========================

def calcular_emocion_global(segmentos: list) -> str:
    if not segmentos:
        return "neutral"

    scores = [s["score_final"] for s in segmentos]

    mean_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    varianza = float(np.std(scores))

    negativos_reales = sum(1 for s in scores if s < -0.3)
    ratio_neg = negativos_reales / len(scores)

    pico_alto = max_score >= 2.0
    pico_bajo = min_score <= -1.2

    hay_conflicto = (
        (pico_alto or pico_bajo) or
        (varianza > 1.0 and ratio_neg > 0.2)
    )

    if hay_conflicto and negativos_reales < 2:
        hay_conflicto = False

    if hay_conflicto:
        return "conflicto"

    if ratio_neg > 0.3:
        return "negativo"

    if mean_score > 0.6:
        return "positivo"

    if mean_score < -0.3:
        return "negativo"

    return "neutral"

# =========================
# 🎤 PROCESAMIENTO PRINCIPAL
# =========================

def procesar_audio(ruta_audio: str) -> dict:

    whisper = get_whisper()

    segments, info = whisper.transcribe(
        ruta_audio,
        beam_size=3,
        best_of=3,
        vad_filter=True
    )

    y, sr = librosa.load(ruta_audio, sr=None)
    y = librosa.util.normalize(y)
    energia_global = float(np.mean(librosa.feature.rms(y=y)))

    idioma = info.language
    resultados = []
    texto_completo = ""

    for seg in segments:

        texto = seg.text.strip()
        if not texto:
            continue

        texto_completo += texto + " "

        # --- TEXTO ---
        s_txt, label, probas = score_texto(texto, idioma)

        # --- AUDIO ---
        start = int(seg.start * sr)
        end = int(seg.end * sr)
        y_seg = y[start:end]

        energia, var_pitch = extraer_features_audio(y_seg, sr)
        s_aud = score_audio(energia, var_pitch, energia_global)

        # --- FUSIÓN ---
        s_final = fusion_score(s_txt, s_aud)

        # --- BOOST SEMÁNTICO ---
        s_final = ajustar_por_agresividad(texto, s_final)

        emocion = clasificar_score(s_final)

        resultados.append({
            "inicio": round(seg.start, 2),
            "fin": round(seg.end, 2),
            "texto": texto,
            "score_texto": s_txt,
            "score_audio": s_aud,
            "score_final": s_final,
            "probas": {k: round(v, 3) for k, v in probas.items()},
            "emocion_final": emocion
        })

    return {
        "metadata": {
            "duracion_total": round(info.duration, 2),
            "idioma": idioma,
            "modelo": "pipeline_v7_fusion_inteligente"
        },
        "texto_completo": texto_completo.strip(),
        "emocion_global": calcular_emocion_global(resultados),
        "segmentos": resultados
    }

# =========================
# 💾 MAIN
# =========================

if __name__ == "__main__":
    import time

    ruta_audio = "audios/audio_largo.mp3"

    print("\n🔄 Procesando...\n")

    t_inicio = time.time()

    t0 = time.time()
    get_whisper()
    print(f"   [Whisper cargado en {time.time() - t0:.1f}s]")

    t0 = time.time()
    get_sentiment()
    print(f"   [pysentimiento cargado en {time.time() - t0:.1f}s]\n")

    t0 = time.time()
    resultado = procesar_audio(ruta_audio)
    print(f"   [Audio procesado en {time.time() - t0:.1f}s]")

    os.makedirs("outputs", exist_ok=True)
    nombre = os.path.splitext(os.path.basename(ruta_audio))[0]
    salida = f"outputs/{nombre}.json"

    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False)

    t_total = time.time() - t_inicio
    print(f"\n✅ Guardado en: {salida}")
    print(f"   Emoción global: {resultado['emocion_global']}")
    print(f"   Segmentos procesados: {len(resultado['segmentos'])}")
    print(f"   Tiempo total: {t_total:.1f}s")