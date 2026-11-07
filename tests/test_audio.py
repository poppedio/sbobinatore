from transcriptor.audio import AudioChunk, fixed_chunks, plan_chunks


def test_plan_chunks_cuts_on_long_gaps_and_target():
    regions = [(0, 100), (101, 250), (262, 300), (300.5, 900), (905, 950)]
    chunks = plan_chunks(regions, total_duration=1000, target_seconds=200, max_seconds=300, max_gap_seconds=8, pad_seconds=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # la pausa 250->262 (12 s) chiude il blocco
    assert chunks[0].start == 0 and chunks[0].end == 250
    # la regione lunga 300.5->900 viene spezzata
    assert all(c.duration <= 300 for c in chunks)
    assert chunks[-1].end == 950
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start


def test_plan_chunks_pad_and_clamp():
    chunks = plan_chunks([(0.1, 10)], total_duration=10, pad_seconds=0.5)
    assert chunks[0].start == 0.0 and chunks[0].end == 10


def test_fixed_chunks():
    chunks = fixed_chunks(1000, 480)
    assert len(chunks) == 3
    assert chunks[-1].end == 1000


def test_chunk_slice():
    import numpy as np

    audio = np.arange(16000 * 3, dtype=np.float32)
    piece = AudioChunk(0, 1.0, 2.0).slice(audio)
    assert len(piece) == 16000 and piece[0] == 16000


def test_plan_chunks_never_overlap_and_prefer_longest_gap():
    # parlato quasi continuo con una pausa netta di 3 s a 310 s e pause brevi altrove
    regions = []
    t = 0.0
    while t < 600:
        regions.append((t, t + 29.0))
        t += 30.0 if abs(t - 300) > 1 else 33.0
    chunks = plan_chunks(regions, total_duration=600, target_seconds=240, max_seconds=360, max_gap_seconds=8, pad_seconds=0.5)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start
    assert any(abs(c.end - 329.0) < 1.0 or abs(c.start - 333.0) < 1.0 for c in chunks)
