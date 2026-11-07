from transcriptor.textproc import (
    build_windows,
    clean_text,
    extractive_summary,
    format_ts,
    highlight_terms_in_markdown,
    segments_to_paragraphs,
    slugify,
)


def test_clean_text_removes_only_non_lexical_fillers():
    text = "ehm allora la derivata, cioè il limite , è importante mh ok"
    cleaned = clean_text(text)
    assert "ehm" not in cleaned and "mh" not in cleaned
    assert "allora" in cleaned.lower() and "cioè" in cleaned and "ok" in cleaned
    assert "limite, è" in cleaned
    assert cleaned[0].isupper()


def test_format_ts_and_slugify():
    assert format_ts(65) == "01:05"
    assert format_ts(3725) == "1:02:05"
    assert slugify("Lezione 3: Analisi (parte 1)") == "lezione_3_analisi_parte_1"
    assert slugify("") == "lezione"


def _segments(n, seconds_each=10.0, text="frase numero {i} della lezione."):
    return [{"start": i * seconds_each, "end": (i + 1) * seconds_each - 1, "text": text.format(i=i)} for i in range(n)]


def test_paragraphs_break_on_long_gap():
    segs = _segments(4)
    segs[2]["start"] = 100.0  # pausa lunga prima del terzo segmento
    segs[2]["end"] = 109.0
    segs[3]["start"], segs[3]["end"] = 110.0, 119.0
    for s in segs:
        s["text"] = "Questa è una frase abbastanza lunga da superare la soglia minima di duecento caratteri " * 3
    paras = segments_to_paragraphs(segs)
    assert len(paras) == 2
    assert paras[0]["start"] == 0.0 and paras[1]["start"] == 100.0


def test_windows_cover_all_text_in_order():
    segs = _segments(200, text="Parola " * 20 + "{i}.")
    windows = build_windows(segs, target_chars=2000, max_chars=2500)
    assert len(windows) > 1
    assert [w["index"] for w in windows] == list(range(len(windows)))
    assert windows[0]["start"] == 0.0
    assert windows[-1]["end"] == segs[-1]["end"]
    for a, b in zip(windows, windows[1:]):
        assert a["end"] <= b["start"]
    assert all(len(w["text"]) <= 2500 for w in windows)


def test_extractive_summary_keeps_sentence_order():
    text = "La derivata misura la pendenza. Il limite definisce la derivata. Oggi piove. La derivata e il limite sono legati."
    summary = extractive_summary(text, max_sentences=2)
    assert len(summary) == 2
    assert text.index(summary[0]) < text.index(summary[1])


def test_highlight_does_not_double_wrap():
    out = highlight_terms_in_markdown("La derivata della derivata.", ["derivata", "Derivata"])
    assert out == "La **derivata** della **derivata**."
