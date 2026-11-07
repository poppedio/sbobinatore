"""Avvia l'app Streamlit in modalità headless e verifica che non sollevi eccezioni."""
import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_renders_without_source(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(os.path.dirname(APP))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert "Carica file" in labels and "Lezioni già elaborate" in labels
    assert "Nessuna lezione in cache." in [c.value for c in at.caption]


def test_app_rejects_invalid_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    [t for t in at.text_input if t.label == "Percorso del file"][0].set_value("/non/esiste.mp3")
    [b for b in at.button if b.label == "Usa questo file"][0].click()
    at.run()
    assert not at.exception
    assert any("non valido" in e.value for e in at.error)


def _seed_lesson(root, file_hash="abc123def456"):
    from transcriptor.cache import LessonCache

    cache = LessonCache(file_hash, root)
    cache.save_source_info("REC0042.m4a", 1234, None)
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Oggi parliamo di Cheynes e della domanda aggregata."},
        {"start": 4.0, "end": 9.0, "text": "La domanda aggregata dipende dal reddito disponibile."},
        {"start": 9.0, "end": 15.0, "text": "Cheynes propone l'intervento pubblico nelle recessioni."},
    ]
    meta = {"backend": "faster-whisper", "model": "small", "model_size": "small", "language": "it", "duration": 15.0,
            "elapsed_seconds": 3.0, "realtime_factor": 5.0, "transcribed_at": 1700000000.0}
    cache.save_transcript(segments, meta)
    return file_hash


def _open_cached(at):
    [b for b in at.button if b.label == "Apri"][0].click()
    at.run()
    return at


def test_app_full_flow_with_cached_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    _seed_lesson(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    _open_cached(at)
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    for expected in ("Più lezioni in coda", "Pacchetto studio", "Correggi la trascrizione", "Testo completo"):
        assert expected in labels
    buttons = [b.label for b in at.button]
    assert "Trascrivi" in buttons or "Ritrascrivi" in buttons
    assert any("Genera" in b for b in buttons)
    # export Word presente
    assert any("Word" in d.label for d in at.get("download_button"))
    # il titolo generico produce un suggerimento solo con un pacchetto: senza pacchetto nessun crash
    assert [t for t in at.text_input if t.label == "Titolo lezione"][0].value == "REC0042"


def test_find_replace_updates_transcript(tmp_path, monkeypatch):
    from transcriptor.cache import LessonCache

    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    file_hash = _seed_lesson(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    _open_cached(at)
    [t for t in at.text_input if t.label == "Trova"][0].set_value("Cheynes")
    [t for t in at.text_input if t.label == "Sostituisci con"][0].set_value("Keynes")
    [b for b in at.button if b.label == "Sostituisci ovunque"][0].click()
    at.run()
    assert not at.exception, at.exception
    segments, meta = LessonCache(file_hash, tmp_path).load_transcript()
    assert all("Cheynes" not in s["text"] for s in segments)
    assert sum(s["text"].count("Keynes") for s in segments) == 2
    assert meta.get("edited_at")
    assert any("2 occorrenze" in s.value for s in at.success)


def test_queue_tab_reports_missing_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    [t for t in at.text_area if t.label.startswith("Percorsi dei file")][0].set_value("/non/esiste/a.mp3\n/non/esiste/b.mp4")
    [b for b in at.button if b.label == "Aggiungi alla coda"][0].click()
    at.run()
    assert not at.exception
    assert any("Non trovati" in e.value for e in at.error)


def test_advanced_toggle_shows_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOR_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    labels = [s.label for s in at.sidebar.selectbox]
    assert "Backend" not in labels and "Modello Whisper" in labels
    at.sidebar.toggle[0].set_value(True)
    at.run()
    assert not at.exception
    labels = [s.label for s in at.sidebar.selectbox]
    assert "Backend" in labels and "Livello sintesi" in labels
