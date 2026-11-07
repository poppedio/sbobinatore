from pathlib import Path

from transcriptor.cache import LessonCache, list_cached_lessons, studypack_key
from transcriptor.export import build_markdown, build_rtf, build_srt, rtf_escape


PARAS = [{"start": 0.0, "end": 5.0, "text": "Prima frase."}, {"start": 65.0, "end": 70.0, "text": "Seconda, con è."}]
PACK = {
    "summary": ["punto"],
    "key_concepts": [{"term": "Derivata", "explanation": "pendenza"}],
    "chapters": [{"title": "Intro", "content": "testo", "start": 0.0, "end": 70.0}],
    "glossary": [{"term": "T", "definition": "d"}],
    "review_questions": ["Perché?"],
}


def test_markdown_has_sections_and_timestamps():
    md = build_markdown("Lez", PARAS, PACK)
    assert md.startswith("# Lez")
    assert "### [00:00] Intro" in md
    assert "**[01:05]** Seconda, con è." in md
    md2 = build_markdown("Lez", PARAS, PACK, with_timestamps=False)
    assert "[01:05]" not in md2


def test_rtf_escape_and_document():
    assert rtf_escape("a{b}\\") == r"a\{b\}\\"
    assert rtf_escape("è") == r"\u232?"
    assert rtf_escape("😀") == r"\u-10179?\u-8704?"
    doc = build_rtf("Lez", PARAS, PACK).decode("utf-8")
    assert doc.startswith(r"{\rtf1") and doc.endswith("}")
    assert r"\u232?" in doc


def test_srt():
    srt = build_srt([{"start": 0.0, "end": 1.5, "text": "ciao"}, {"start": 3661.25, "end": 3662.0, "text": "fine"}])
    assert "1\n00:00:00,000 --> 00:00:01,500\nciao" in srt
    assert "01:01:01,250 --> 01:01:02,000" in srt


def test_lesson_cache_roundtrip(tmp_path: Path):
    cache = LessonCache("abc123", root=tmp_path)
    assert not cache.has_transcript()
    cache.save_source_info("lez.mp4", 10, "/x/lez.mp4")
    cache.save_transcript([{"start": 0, "end": 1, "text": "x"}], {"duration": 1})
    segments, meta = cache.load_transcript()
    assert segments[0]["text"] == "x" and meta["duration"] == 1
    key = studypack_key(generator="ollama", model="m")
    assert key == studypack_key(model="m", generator="ollama")
    cache.save_studypack(key, {"summary": ["s"]})
    assert cache.load_studypack(key) == {"summary": ["s"]}
    lessons = list_cached_lessons(tmp_path)
    assert lessons[0]["name"] == "lez.mp4" and lessons[0]["has_transcript"] and lessons[0]["studypacks"] == 1

    store = cache.chunk_store()
    store.ensure_plan([{"index": 0}])
    store.save(0, [{"start": 0, "end": 1, "text": "a"}])
    assert store.load(0)[0]["text"] == "a"
    store.ensure_plan([{"index": 0}, {"index": 1}])  # piano diverso: azzera
    assert store.load(0) is None
