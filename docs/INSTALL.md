# Guía de instalación

Este proyecto puede correr en tres configuraciones de hardware distintas.
Sigue **solo la sección que corresponde a tu caso**.

---

## Requisitos previos (todos los casos)

- Python 3.10 o superior
- `git` instalado
- Al menos 6 GB de RAM libre

```bash
git clone <url-del-repo>
cd procesa_audio
python -m venv venv
```

---

## Opción 1 — CPU (Windows o Linux)

Sin GPU. Más lento pero funciona en cualquier máquina.

**Tiempo estimado por audio:** ~50-90 segundos

```bash
# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate

pip install torch torchvision torchaudio
pip install faster-whisper transformers librosa pysentimiento
```

Scripts disponibles: `procesar_audio_vf.py`, `procesar_audio_chunk.py`

---

## Opción 2 — NVIDIA GPU + CUDA (Windows o Linux)

Requiere GPU NVIDIA con al menos 6 GB VRAM.
Recomendado: RTX 3060 12 GB o superior.

**Tiempo estimado por audio:** ~5-10 segundos

### Paso 1 — Verificar que CUDA está instalado

```bash
nvidia-smi
```

Debes ver el driver y la versión de CUDA. Si no aparece, instala los drivers desde el sitio de NVIDIA.

### Paso 2 — Instalar PyTorch con CUDA

Elige según la versión de CUDA que reportó `nvidia-smi`:

```bash
# CUDA 12.1 (más común en drivers recientes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Paso 3 — Resto de dependencias

```bash
pip install faster-whisper transformers librosa pysentimiento
```

### Paso 4 — Verificar

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Debe imprimir `True` y el nombre de tu GPU.

Scripts disponibles: todos, incluyendo `test_gpu.py`

---

## Opción 3 — AMD GPU + ROCm (solo Linux)

Requiere GPU AMD RDNA2 o superior (RX 6000 series en adelante).
La RX 6650 XT es compatible (gfx1032).

> **Windows con AMD no está soportado** por estas librerías.
> Usar Linux (Ubuntu 22.04 recomendado).

**Tiempo estimado por audio:** ~8-15 segundos

### Paso 1 — Instalar ROCm

Sigue la guía oficial según tu distribución:
https://rocm.docs.amd.com/en/latest/deploy/linux/index.html

Para Ubuntu 22.04:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $LOGNAME
# Reiniciar sesión
```

### Paso 2 — Verificar ROCm

```bash
rocm-smi
```

Debes ver tu GPU listada.

### Paso 3 — Instalar PyTorch con ROCm

```bash
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Paso 4 — Resto de dependencias

```bash
pip install transformers librosa pysentimiento
```

> `faster-whisper` usa CTranslate2 que no soporta ROCm.
> El script `test_gpu.py` usa Whisper vía `transformers` que sí es compatible.

### Paso 5 — Verificar

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

En ROCm, PyTorch expone la GPU AMD bajo el nombre `cuda`. Debe imprimir `True`.

Scripts disponibles: `test_gpu.py`

---

## Tabla resumen

| | CPU | NVIDIA + CUDA | AMD + ROCm |
|---|---|---|---|
| Sistema operativo | Windows / Linux | Windows / Linux | **Solo Linux** |
| Script principal | `procesar_audio_vf.py` | `procesar_audio_vf.py` | `test_gpu.py` |
| Whisper backend | faster-whisper | faster-whisper | transformers |
| Velocidad | Lenta | Rapida | Rapida |
| VRAM mínima | — | 6 GB | 6 GB |

---

## Verificación general

Independientemente de la opción elegida, puedes verificar que todo funciona con:

```bash
python test_gpu.py
```

El script detecta automáticamente si hay GPU disponible e informa qué hardware está usando.
