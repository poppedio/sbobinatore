"""Preparazione audio: conversione con ffmpeg, caricamento, VAD e suddivisione in blocchi."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000


def file_sha256(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def ffmpeg_binary() -> Optional[str]:
    return shutil.which("ffmpeg")


_FFMPEG_OK: Optional[bool] = None


def ffmpeg_works() -> bool:
    """True se il binario ffmpeg esiste e parte davvero (una libreria mancante lo fa fallire all'avvio)."""
    global _FFMPEG_OK
    if _FFMPEG_OK is None:
        binary = ffmpeg_binary()
        if not binary:
            _FFMPEG_OK = False
        else:
            try:
                proc = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=15)
                _FFMPEG_OK = proc.returncode == 0
            except Exception:
                _FFMPEG_OK = False
    return _FFMPEG_OK


def _write_wav16k(dst: str, audio: np.ndarray) -> None:
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    tmp = dst + ".part"
    with wave.open(tmp, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    os.replace(tmp, dst)


def _convert_with_pyav(src: str, dst: str) -> None:
    """Fallback senza binario ffmpeg: PyAV (dipendenza di faster-whisper) decodifica e ricampiona."""
    from faster_whisper.audio import decode_audio

    audio = decode_audio(src, sampling_rate=SAMPLE_RATE, split_stereo=False)
    _write_wav16k(dst, np.asarray(audio, dtype=np.float32))


def convert_to_wav16k(src: str, dst: str) -> str:
    """Converte qualsiasi audio/video in WAV mono 16 kHz 16 bit (il formato nativo di Whisper).

    Usa ffmpeg se funziona, altrimenti PyAV. Ritorna il nome del convertitore usato.
    """
    if ffmpeg_works():
        tmp = dst + ".part"
        cmd = [
            ffmpeg_binary(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", src, "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", tmp,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            os.replace(tmp, dst)
            return "ffmpeg"
        try:
            os.remove(tmp)
        except OSError:
            pass
        ffmpeg_error = proc.stderr.strip()[:500]
    else:
        ffmpeg_error = "binario ffmpeg assente o non avviabile"
    try:
        _convert_with_pyav(src, dst)
        return "pyav"
    except Exception as exc:
        raise RuntimeError(
            f"Impossibile convertire il file. ffmpeg: {ffmpeg_error}. PyAV: {type(exc).__name__}: {exc}"
        ) from exc


def wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def load_wav16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != SAMPLE_RATE or handle.getsampwidth() != 2:
            raise ValueError("Atteso un WAV mono a 16 kHz e 16 bit.")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


@dataclass
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def slice(self, audio: np.ndarray) -> np.ndarray:
        return audio[int(self.start * SAMPLE_RATE): int(self.end * SAMPLE_RATE)]

    def to_dict(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3), "end": round(self.end, 3)}


def detect_speech_regions(audio: np.ndarray, min_silence_ms: int = 800, threshold: float = 0.5) -> List[Tuple[float, float]]:
    """Regioni di parlato (secondi) tramite Silero VAD, distribuito con faster-whisper."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=250,
        max_speech_duration_s=float("inf"),
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=200,
    )
    stamps = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def plan_chunks(
    regions: Sequence[Tuple[float, float]],
    total_duration: float,
    target_seconds: float = 480.0,
    max_seconds: float = 720.0,
    max_gap_seconds: float = 8.0,
    pad_seconds: float = 0.25,
) -> List[AudioChunk]:
    """Raggruppa le regioni di parlato in blocchi contigui tagliati sulle pause.

    - una pausa più lunga di max_gap_seconds chiude sempre il blocco (il silenzio non viene trascritto);
    - raggiunto target_seconds il blocco si chiude alla pausa più lunga trovata prima di max_seconds;
    - le regioni troppo lunghe vengono spezzate a target_seconds;
    - i blocchi non si sovrappongono mai: il padding entra solo nelle pause.
    """
    pieces: List[Tuple[float, float]] = []
    for start, end in regions:
        start = max(0.0, float(start))
        end = min(float(total_duration), float(end))
        if end <= start:
            continue
        while end - start > max_seconds:
            pieces.append((start, start + target_seconds))
            start += target_seconds
        pieces.append((start, end))

    merged: List[Tuple[float, float]] = []
    current: List[Tuple[float, float]] = []

    def close(upto: int) -> None:
        nonlocal current
        merged.append((current[0][0], current[upto][1]))
        current = current[upto + 1:]

    for piece in pieces:
        if not current:
            current.append(piece)
            continue
        gap = piece[0] - current[-1][1]
        if gap > max_gap_seconds:
            close(len(current) - 1)
            current.append(piece)
            continue
        if piece[1] - current[0][0] > max_seconds:
            # scegli la pausa più lunga tra quelle successive al 60% del target
            best_idx, best_gap = len(current) - 1, -1.0
            for idx in range(len(current) - 1):
                if current[idx][1] - current[0][0] < target_seconds * 0.6:
                    continue
                g = current[idx + 1][0] - current[idx][1]
                if g > best_gap:
                    best_idx, best_gap = idx, g
            close(best_idx)
            current.append(piece)
            continue
        current.append(piece)
        if current[-1][1] - current[0][0] >= target_seconds and gap >= 1.5:
            close(len(current) - 1)
    if current:
        close(len(current) - 1)

    chunks: List[AudioChunk] = []
    for i, (start, end) in enumerate(merged):
        prev_end = merged[i - 1][1] if i > 0 else 0.0
        next_start = merged[i + 1][0] if i + 1 < len(merged) else total_duration
        pad_before = pad_seconds if i == 0 else min(pad_seconds, max(0.0, (start - prev_end) / 2))
        pad_after = pad_seconds if i + 1 == len(merged) else min(pad_seconds, max(0.0, (next_start - end) / 2))
        start = max(0.0, start - pad_before)
        end = min(total_duration, end + pad_after)
        chunks.append(AudioChunk(i, start, end))
    return chunks


def fixed_chunks(total_duration: float, seconds: float = 480.0) -> List[AudioChunk]:
    chunks: List[AudioChunk] = []
    start = 0.0
    index = 0
    while start < total_duration:
        end = min(total_duration, start + seconds)
        chunks.append(AudioChunk(index, start, end))
        start = end
        index += 1
    return chunks
