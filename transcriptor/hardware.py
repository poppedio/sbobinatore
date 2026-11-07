"""Rilevamento hardware e stime dei tempi: servono a dire all'utente cosa aspettarsi prima di premere Trascrivi."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

_CUDA: Optional[bool] = None
_LOCK = threading.Lock()


def ram_gb() -> Optional[float]:
    """Memoria fisica in GB (None se non determinabile)."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemStatus()
            status.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 2 ** 30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except Exception:
        return None


def cpu_cores() -> int:
    return os.cpu_count() or 4


def cuda_available() -> bool:
    """True se ctranslate2 vede almeno una GPU CUDA (faster-whisper può usarla)."""
    global _CUDA
    with _LOCK:
        if _CUDA is None:
            try:
                import ctranslate2

                _CUDA = ctranslate2.get_cuda_device_count() > 0
            except Exception:
                _CUDA = False
        return _CUDA


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def accelerator_key(backend_resolved: str, device: str) -> str:
    """'mlx' | 'cuda' | 'cpu': la classe di velocità con cui gira Whisper."""
    if backend_resolved == "mlx-whisper":
        return "mlx"
    if device == "cuda":
        return "cuda"
    return "cpu"


def accelerator_label(key: str) -> str:
    return {"mlx": "GPU Apple Silicon", "cuda": "GPU NVIDIA", "cpu": "solo CPU"}.get(key, key)


# Fattori "tempo reale" indicativi (durata audio / tempo di calcolo). cpu = 8 core moderni, int8.
SPEED_FACTORS: Dict[str, Dict[str, float]] = {
    "mlx": {"tiny": 90, "base": 70, "small": 50, "medium": 25, "large-v3": 10, "large-v3-turbo": 22},
    "cuda": {"tiny": 120, "base": 100, "small": 70, "medium": 35, "large-v3": 14, "large-v3-turbo": 30},
    "cpu": {"tiny": 14, "base": 9, "small": 4.5, "medium": 1.6, "large-v3": 0.6, "large-v3-turbo": 1.2},
}

MODEL_QUALITY = {
    "tiny": "qualità bassa", "base": "qualità modesta", "small": "qualità discreta",
    "medium": "qualità buona", "large-v3": "qualità massima", "large-v3-turbo": "qualità ottima",
}


def _speeds_path() -> Path:
    from .cache import cache_root

    return cache_root() / "speeds.json"


def _load_speeds() -> Dict[str, float]:
    try:
        with open(_speeds_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        return {}


def record_speed(meta: Dict) -> None:
    """Memorizza il fattore tempo reale misurato (media mobile) per stime future più precise."""
    rtf = meta.get("realtime_factor")
    model = meta.get("model_size") or meta.get("model")
    if not rtf or not model:
        return
    key = f"{accelerator_key(meta.get('backend', ''), meta.get('device', 'cpu'))}|{_short_model(model)}"
    with _LOCK:
        speeds = _load_speeds()
        old = speeds.get(key)
        speeds[key] = round(float(rtf) if old is None else 0.6 * old + 0.4 * float(rtf), 2)
        try:
            path = _speeds_path()
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(speeds, handle)
            os.replace(tmp, path)
        except Exception:
            pass


def _short_model(model: str) -> str:
    tail = model.split("/")[-1]
    for prefix in ("faster-whisper-", "whisper-"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
    return tail


def speed_factor(model_size: str, accel: str) -> float:
    """Fattore tempo reale: misurato se disponibile, altrimenti tabellare (CPU scalata sui core)."""
    learned = _load_speeds().get(f"{accel}|{_short_model(model_size)}")
    if learned:
        return learned
    base = SPEED_FACTORS.get(accel, SPEED_FACTORS["cpu"]).get(_short_model(model_size), 1.0)
    if accel == "cpu":
        base *= min(2.0, max(0.5, (cpu_cores() / 8.0) ** 0.7))
    return base


def estimate_seconds(duration_seconds: float, model_size: str, accel: str) -> float:
    return float(duration_seconds) / max(0.05, speed_factor(model_size, accel))


def format_duration_rough(seconds: float) -> str:
    """'circa 2 min', 'circa 45 min', 'circa 1 h 20 min', pensato per stime."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "meno di 1 min"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"circa {minutes} min"
    hours, rem = divmod(minutes, 60)
    return f"circa {hours} h" if rem < 5 else f"circa {hours} h {rem} min"


def media_duration_seconds(path: str) -> Optional[float]:
    """Durata di un file audio/video senza convertirlo: ffprobe se c'è, altrimenti PyAV."""
    probe = shutil.which("ffprobe")
    if probe:
        try:
            out = subprocess.run(
                [probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out:
                return float(out)
        except Exception:
            pass
    try:
        import av

        with av.open(path) as container:
            if container.duration:
                return container.duration / 1_000_000.0
    except Exception:
        pass
    return None


# --- Ollama e memoria ---
def ollama_models_that_fit(models: List[Dict], ram: Optional[float], share: float = 0.7) -> List[str]:
    """Nomi dei modelli il cui file sta nella quota di RAM utilizzabile (unificata su Apple Silicon)."""
    if ram is None:
        return [m["name"] for m in models]
    budget = ram * share * 2 ** 30
    return [m["name"] for m in models if not m.get("size") or m["size"] <= budget]


def suggest_ollama_pull(ram: Optional[float]) -> str:
    if ram is None or ram >= 28:
        return "qwen3:30b"
    if ram >= 14:
        return "gemma3:12b"
    return "qwen3:8b"


def hardware_summary() -> str:
    parts = []
    if is_apple_silicon():
        parts.append("Mac Apple Silicon")
    elif cuda_available():
        parts.append("GPU NVIDIA")
    else:
        parts.append(f"CPU {cpu_cores()} core")
    ram = ram_gb()
    if ram:
        parts.append(f"{ram:g} GB di memoria")
    return ", ".join(parts)


