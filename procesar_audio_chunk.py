import os
import json
import time
from faster_whisper import WhisperModel
from transformers import pipeline

# =========================
# 🔧 Configuración
# =========================

CHUNK_SEGUNDOS = 60  # 👈 puedes cambiar a 30, 45, etc.

print("🔄 Cargando modelo Whisper...")
whisper_model = WhisperModel("base", device="cpu")

print("🔄 Cargando modelo de sentimiento...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# =========================
# 🧠 Mapeos
# =========================

MAPEO_NUMERICO = {
    "1 star": 1,
    "2 stars": 2,
    "3 stars": 3,
    "4 stars": 4,
    "5 stars": 5
}


def normalizar_sentimiento(label):
    return {
        "1 star": "muy negativo",
        "2 stars": "negativo",
        "3 stars": "neutral",
        "4 stars": "positivo",
        "5 stars": "muy positivo"
    }.get(label, label)


def interpretar_promedio(valor):
    if valor <= 1.5:
        return "muy negativo"
    elif valor <= 2.5:
        return "negativo"
    elif valor <= 3.5:
        return "neutral"
    elif valor <= 4.5:
        return "positivo"
    else:
        return "muy positivo"


# =========================
# 📊 Sentimiento global
# =========================

def calcular_sentimiento_global(items):
    suma_ponderada = 0
    suma_scores = 0

    for item in items:
        valor = MAPEO_NUMERICO.get(item["label_original"], 3)
        score = item["score"]

        suma_ponderada += valor * score
        suma_scores += score

    if suma_scores == 0:
        return {"label": "neutral", "score": 3}

    promedio = suma_ponderada / suma_scores

    return {
        "label": interpretar_promedio(promedio),
        "score": round(promedio, 2)
    }


# =========================
# 🔥 Crear bloques
# =========================

def crear_bloques(segmentos, ventana=60):
    bloques = []
    bloque_actual = []
    inicio_bloque = 0

    for seg in segmentos:
        if not bloque_actual:
            inicio_bloque = seg["inicio"]

        bloque_actual.append(seg)

        duracion = seg["fin"] - inicio_bloque

        if duracion >= ventana:
            bloques.append(procesar_bloque(bloque_actual))
            bloque_actual = []

    # último bloque
    if bloque_actual:
        bloques.append(procesar_bloque(bloque_actual))

    return bloques


def procesar_bloque(segmentos_bloque):
    texto = " ".join([s["texto"] for s in segmentos_bloque])

    sentimiento = sentiment_pipeline(texto)[0]

    return {
        "inicio": segmentos_bloque[0]["inicio"],
        "fin": segmentos_bloque[-1]["fin"],
        "duracion": round(segmentos_bloque[-1]["fin"] - segmentos_bloque[0]["inicio"], 2),
        "texto": texto,
        "sentimiento": normalizar_sentimiento(sentimiento["label"]),
        "label_original": sentimiento["label"],
        "score": round(sentimiento["score"], 4)
    }


# =========================
# 🎤 Procesamiento principal
# =========================

def procesar_audio(ruta_audio):
    segments, info = whisper_model.transcribe(ruta_audio)

    segmentos = []
    texto_completo = ""

    for segment in segments:
        texto = segment.text.strip()

        if not texto:
            continue

        texto_completo += texto + " "

        sentimiento = sentiment_pipeline(texto)[0]

        segmentos.append({
            "inicio": round(segment.start, 2),
            "fin": round(segment.end, 2),
            "duracion": round(segment.end - segment.start, 2),
            "texto": texto,
            "sentimiento": normalizar_sentimiento(sentimiento["label"]),
            "label_original": sentimiento["label"],
            "score": round(sentimiento["score"], 4)
        })

    texto_completo = texto_completo.strip()

    # 🔥 Crear bloques
    bloques = crear_bloques(segmentos, CHUNK_SEGUNDOS)

    # 📊 Global (puedes usar bloques o segmentos)
    sentimiento_global = calcular_sentimiento_global(bloques)

    eventos_clave = extraer_eventos_clave(segmentos)

    metadata = {
        "duracion_total": round(info.duration, 2) if info.duration else None,
        "idioma": info.language,
        "chunk_segundos": CHUNK_SEGUNDOS
    }

    return {
        "metadata": metadata,
        "sentimiento_global": sentimiento_global,
        "bloques": bloques,
        "eventos_clave": eventos_clave
    }


# =========================
# 💾 Guardar JSON
# =========================

def generar_nombre_salida(ruta_audio):
    base = os.path.splitext(os.path.basename(ruta_audio))[0]
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", f"{base}.json")

def es_evento_clave(segmento):
    return segmento["sentimiento"] in ["muy negativo", "muy positivo"]

def extraer_eventos_clave(segmentos):
    eventos = []

    for seg in segmentos:
        if es_evento_clave(seg):
            eventos.append({
                "timestamp": seg["inicio"],
                "texto": seg["texto"],
                "sentimiento": seg["sentimiento"]
            })

    return eventos

# =========================
# 🚀 MAIN
# =========================

if __name__ == "__main__":

    ruta_audio = "audios/cliente_nocturno.mp3.mpeg"

    print("\n🔄 Procesando audio...\n")

    t_inicio = time.time()

    resultado = procesar_audio(ruta_audio)

    archivo = generar_nombre_salida(ruta_audio)

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False)

    t_total = time.time() - t_inicio
    print(f"\n✅ Guardado en: {archivo}")
    print(f"   Tiempo total: {t_total:.1f}s")