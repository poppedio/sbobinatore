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
