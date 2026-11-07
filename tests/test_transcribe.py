from transcriptor.transcribe import faster_model_for, filter_segments, is_hallucination, mlx_repo_for


def test_hallucination_filter():
    assert is_hallucination("Sottotitoli creati dalla comunità Amara.org")
    assert is_hallucination("si si si si si si si si si si")
    assert not is_hallucination("La derivata di una funzione misura la pendenza della tangente.")


def test_filter_segments_drops_triple_repeats_and_empty():
    segs = [
        {"start": 0, "end": 1, "text": "ciao"},
        {"start": 1, "end": 2, "text": "ciao"},
        {"start": 2, "end": 3, "text": "ciao"},
        {"start": 3, "end": 4, "text": "  "},
        {"start": 4, "end": 5, "text": "fine"},
    ]
    out = filter_segments(segs)
    assert [s["text"] for s in out] == ["ciao", "ciao", "fine"]


def test_model_name_mapping():
    assert mlx_repo_for("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert mlx_repo_for("mlx-community/whisper-small") == "mlx-community/whisper-small"
    assert faster_model_for("mlx-community/whisper-large-v3") == "large-v3"
    assert faster_model_for("medium") == "medium"


def test_model_is_downloaded_uses_hf_cache(tmp_path, monkeypatch):
    from transcriptor.transcribe import model_is_downloaded, model_repo_for

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert model_is_downloaded("medium", "faster-whisper") is False
    repo = model_repo_for("medium", "faster-whisper")
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert model_is_downloaded("medium", "faster-whisper") is False  # snapshot vuoto
    (snap / "model.bin").write_bytes(b"x")
    assert model_is_downloaded("medium", "faster-whisper") is True
