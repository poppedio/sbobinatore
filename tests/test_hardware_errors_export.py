"""Stime dei tempi, errori comprensibili, export DOCX, titoli generici."""
import zipfile
import io

from transcriptor import hardware
from transcriptor.errors import friendly_error
from transcriptor.export import build_docx
from transcriptor.textproc import looks_generic_title


def test_estimates_are_ordered_by_model_and_accelerator(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    cpu_turbo = hardware.estimate_seconds(3600, "large-v3-turbo", "cpu")
    cpu_small = hardware.estimate_seconds(3600, "small", "cpu")
    mlx_turbo = hardware.estimate_seconds(3600, "large-v3-turbo", "mlx")
    assert cpu_small < cpu_turbo
    assert mlx_turbo < cpu_turbo
    assert hardware.format_duration_rough(30) == "meno di 1 min"
    assert hardware.format_duration_rough(600) == "circa 10 min"
    assert hardware.format_duration_rough(4800) == "circa 1 h 20 min"
    assert hardware.format_duration_rough(3660) == "circa 1 h"


def test_record_speed_is_used_for_later_estimates(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    before = hardware.speed_factor("medium", "cpu")
    # clip breve: ignorata (il caricamento del modello falserebbe la misura)
    hardware.record_speed({"backend": "faster-whisper", "device": "cpu", "model": "medium", "realtime_factor": 0.2, "duration": 15.0})
    assert hardware.speed_factor("medium", "cpu") == before
    hardware.record_speed({"backend": "faster-whisper", "device": "cpu", "model": "medium", "realtime_factor": 3.0, "duration": 1800.0})
    assert hardware.speed_factor("medium", "cpu") == 3.0
    hardware.record_speed({"backend": "faster-whisper", "device": "cpu", "model": "Systran/faster-whisper-medium", "realtime_factor": 5.0, "duration": 1800.0})
    assert abs(hardware.speed_factor("medium", "cpu") - 3.8) < 0.01  # media mobile 0.6*3 + 0.4*5


def test_ollama_fit_and_suggestion():
    models = [{"name": "qwen3:30b", "size": 19 * 2 ** 30}, {"name": "gemma3:12b", "size": 8 * 2 ** 30}, {"name": "qwen3:8b", "size": 5 * 2 ** 30}]
    assert hardware.ollama_models_that_fit(models, 16) == ["gemma3:12b", "qwen3:8b"]
    assert hardware.ollama_models_that_fit(models, None) == ["qwen3:30b", "gemma3:12b", "qwen3:8b"]
    assert hardware.suggest_ollama_pull(8) == "qwen3:8b"
    assert hardware.suggest_ollama_pull(16) == "gemma3:12b"
    assert hardware.suggest_ollama_pull(32) == "qwen3:30b"


def test_friendly_errors():
    assert friendly_error("ConnectionError: HTTPSConnectionPool(host='huggingface.co')")[0].startswith("Connessione")
    assert friendly_error("RuntimeError: CUDA failed with error out of memory")[0] in ("Memoria insufficiente", "Librerie o driver NVIDIA non disponibili")
    assert friendly_error("OllamaError: HTTPConnectionPool(host='localhost', port=11434)")[0] == "Ollama non risponde"
    assert friendly_error("RuntimeError: File sorgente non trovato: ricarica")[0] == "File non trovato"
    title, hint, detail = friendly_error("ZeroDivisionError: division by zero")
    assert title == "Qualcosa è andato storto" and "division" in detail


def test_build_docx_contains_sections():
    paragraphs = [{"start": 0.0, "end": 5.0, "text": "Prima frase della lezione."}]
    pack = {"summary": ["Punto uno"], "key_concepts": [{"term": "Domanda", "explanation": "spiegazione"}],
            "chapters": [{"start": 0, "end": 5, "title": "Introduzione", "content": "Testo del capitolo"}],
            "glossary": [{"term": "PIL", "definition": "prodotto interno lordo"}], "review_questions": ["Perché?"]}
    data = build_docx("Titolo prova", paragraphs, pack, with_timestamps=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    for needle in ("Titolo prova", "Sintesi", "Punto uno", "Domanda", "Introduzione", "PIL", "Perché?", "[00:00]", "Prima frase"):
        assert needle in xml


def test_looks_generic_title():
    assert looks_generic_title("REC0042")
    assert looks_generic_title("Nuova registrazione 12")
    assert looks_generic_title("2024-10-01 10.30.15")
    assert looks_generic_title("")
    assert not looks_generic_title("Analisi matematica 2 - limiti")
    assert not looks_generic_title("Keynes e la domanda aggregata")
