"""Trascrizione a blocchi con mlx-whisper o faster-whisper, con progresso, annullamento e ripresa."""
from __future__ import annotations

import platform
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .audio import AudioChunk, detect_speech_regions, fixed_chunks, plan_chunks, SAMPLE_RATE

Segment = Dict
ProgressFn = Callable[[str, float, float, str], None]

FASTER_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
MODEL_OPTIONS = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]

HALLUCINATION_PATTERNS = [
    r"sottotitol", r"amara\.org", r"www\.", r"grazie per aver guardato", r"iscriviti al canale",
    r"subtitles by", r"thanks for watching", r"trascrizione a cura", r"a cura di qtss",
]
_HALLUCINATION_RE = re.compile("|".join(HALLUCINATION_PATTERNS), flags=re.IGNORECASE)


class TranscriptionCancelled(Exception):
    pass


def is_apple_silicon_mac() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    if not is_apple_silicon_mac():
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "mlx-whisper" if mlx_available() else "faster-whisper"
    return backend


def mlx_repo_for(model_size: str) -> str:
    if model_size.startswith("mlx-community/"):
        return model_size
    return MLX_MODELS.get(model_size, MLX_MODELS["large-v3-turbo"])


def faster_model_for(model_size: str) -> str:
    if model_size in FASTER_WHISPER_MODELS:
        return model_size
    tail = model_size.split("/")[-1].replace("whisper-", "")
    return tail if tail in FASTER_WHISPER_MODELS else "large-v3-turbo"


def choose_default_model() -> str:
    return "large-v3-turbo" if is_apple_silicon_mac() else "medium"


# --- cache dei modelli (un caricamento per processo) ---
_MODEL_CACHE: Dict[tuple, object] = {}
_MODEL_LOCK = threading.Lock()


def get_faster_whisper_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    key = (model_size, device, compute_type)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            _MODEL_CACHE[key] = model
    return model


# --- filtro allucinazioni ---
def _norm(text: str) -> str:
    return re.sub(r"[^\wà-ÿ]+", " ", text.lower()).strip()


def is_hallucination(text: str) -> bool:
    if _HALLUCINATION_RE.search(text):
        return True
    words = _norm(text).split()
    if len(words) >= 8:
        counts: Dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        if max(counts.values()) / len(words) > 0.5:
            return True
    return False


def filter_segments(segments: List[Segment]) -> List[Segment]:
    kept: List[Segment] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text or is_hallucination(text):
            continue
        norm = _norm(text)
        if len(kept) >= 2 and _norm(kept[-1]["text"]) == norm and _norm(kept[-2]["text"]) == norm:
            continue
        kept.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})
    return kept


# --- backend ---
def _transcribe_chunk_mlx(audio: np.ndarray, repo: str, language: Optional[str]) -> Tuple[List[Segment], Optional[str]]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=repo,
        language=language,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=False,
        verbose=None,
    )
    segments = [
        {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", "")).strip()}
        for s in (result.get("segments") or [])
    ]
    return segments, result.get("language")


def _transcribe_chunk_faster(model, audio: np.ndarray, language: Optional[str], vad_filter: bool) -> Tuple[List[Segment], Optional[str]]:
    iterator, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in iterator]
    return segments, getattr(info, "language", None)


def transcribe_audio(
    audio: np.ndarray,
    *,
    backend: str = "auto",
    model_size: str = "large-v3-turbo",
    language: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8",
    vad_filter: bool = True,
    chunk_target_seconds: float = 480.0,
    on_progress: Optional[ProgressFn] = None,
    cancel_event: Optional[threading.Event] = None,
    chunk_store=None,
    allow_cpu_fallback: bool = False,
) -> Tuple[List[Segment], Dict]:
    """Trascrive l'intero audio (float32, 16 kHz) a blocchi e restituisce segmenti con tempi assoluti."""

    def progress(stage: str, done: float, total: float, message: str) -> None:
        if on_progress:
            on_progress(stage, done, total, message)

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled("Trascrizione annullata.")

    started = time.time()
    duration = len(audio) / SAMPLE_RATE
    language = None if language in (None, "", "auto") else language

    progress("vad", 0, duration, "Rilevo le pause per dividere l'audio in blocchi…")
    vad_used = True
    try:
        regions = detect_speech_regions(audio) if vad_filter else [(0.0, duration)]
        chunks = plan_chunks(regions, duration, target_seconds=chunk_target_seconds, max_seconds=chunk_target_seconds * 1.5)
    except Exception:
        vad_used = False
        chunks = fixed_chunks(duration, chunk_target_seconds)
    if not chunks:
        return [], {"duration": duration, "chunks": 0, "backend": backend, "warning": "Nessun parlato rilevato."}

    speech_seconds = sum(c.duration for c in chunks)
    resolved = resolve_backend(backend)
    fallback_reason = None
    fw_model = None
    if resolved == "faster-whisper":
        progress("load", 0, duration, f"Carico faster-whisper {faster_model_for(model_size)} ({device}/{compute_type})…")
        fw_model = get_faster_whisper_model(faster_model_for(model_size), device, compute_type)

    plan = [c.to_dict() for c in chunks]
    if chunk_store is not None:
        chunk_store.ensure_plan(plan)

    all_segments: List[Segment] = []
    detected_language = language
    done_seconds = 0.0
    chunk_times: List[float] = []

    for chunk in chunks:
        check_cancel()
        cached = chunk_store.load(chunk.index) if chunk_store is not None else None
        if cached is not None:
            all_segments.extend(cached)
            done_seconds += chunk.duration
            progress("transcribe", done_seconds, speech_seconds,
                     f"Blocco {chunk.index + 1}/{len(chunks)} ripreso dalla cache.")
            continue

        eta = ""
        if chunk_times:
            per_second = sum(chunk_times) / max(1e-6, done_seconds)
            remaining = (speech_seconds - done_seconds) * per_second
            eta = f" · stimati {int(remaining // 60)} min {int(remaining % 60)} s"
        progress("transcribe", done_seconds, speech_seconds,
                 f"Blocco {chunk.index + 1}/{len(chunks)} ({_fmt(chunk.start)}–{_fmt(chunk.end)}) con {resolved}{eta}")

        piece = chunk.slice(audio)
        t0 = time.time()
        try:
            if resolved == "mlx-whisper":
                segs, lang = _transcribe_chunk_mlx(piece, mlx_repo_for(model_size), detected_language)
            else:
                segs, lang = _transcribe_chunk_faster(fw_model, piece, detected_language, vad_filter)
        except TranscriptionCancelled:
            raise
        except Exception as exc:
            if resolved == "mlx-whisper" and allow_cpu_fallback:
                fallback_reason = f"{type(exc).__name__}: {exc}"
                resolved = "faster-whisper"
                progress("load", done_seconds, speech_seconds, "MLX ha fallito, passo a faster-whisper su CPU…")
                fw_model = get_faster_whisper_model(faster_model_for(model_size), "cpu", "int8")
                segs, lang = _transcribe_chunk_faster(fw_model, piece, detected_language, vad_filter)
            else:
                raise RuntimeError(
                    f"Errore nel blocco {chunk.index + 1} con {resolved}: {type(exc).__name__}: {exc}"
                ) from exc
        chunk_times.append(time.time() - t0)

        if detected_language is None and lang:
            detected_language = lang
        for seg in segs:
            seg["start"] = round(seg["start"] + chunk.start, 3)
            seg["end"] = round(seg["end"] + chunk.start, 3)
        segs = filter_segments(segs)
        if chunk_store is not None:
            chunk_store.save(chunk.index, segs)
        all_segments.extend(segs)
        done_seconds += chunk.duration

    all_segments = filter_segments(all_segments)
    elapsed = time.time() - started
    meta = {
        "backend": resolved,
        "model": mlx_repo_for(model_size) if resolved == "mlx-whisper" else faster_model_for(model_size),
        "language": detected_language,
        "duration": round(duration, 2),
        "speech_seconds": round(speech_seconds, 2),
        "chunks": len(chunks),
        "vad": vad_used,
        "device": "apple-silicon" if resolved == "mlx-whisper" else device,
        "compute_type": "mlx" if resolved == "mlx-whisper" else compute_type,
        "elapsed_seconds": round(elapsed, 1),
        "realtime_factor": round(duration / elapsed, 1) if elapsed > 0 else None,
        "fallback_reason": fallback_reason,
        "segments": len(all_segments),
    }
    progress("done", speech_seconds, speech_seconds, "Trascrizione completata.")
    return all_segments, meta


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
