import pytest

from transcriptor import ollama
from transcriptor.studypack import build_notes_markdown, chapters_from_reduce, generate_heuristic


def test_parse_json_object_tolerates_wrappers():
    raw = "<think>ragiono</think>```json\n{\"a\": 1, \"b\": [1,2]}\n```"
    assert ollama.parse_json_object(raw) == {"a": 1, "b": [1, 2]}
    assert ollama.parse_json_object('testo prima {"x": "y}"} dopo') == {"x": "y}"}
    with pytest.raises(ollama.OllamaError):
        ollama.parse_json_object("")
    with pytest.raises(ollama.OllamaError):
        ollama.parse_json_object("[1,2]")


def test_choose_num_ctx_bounds():
    assert ollama.choose_num_ctx("x" * 100) == ollama.MIN_CTX
    assert ollama.choose_num_ctx("x" * 30000, reserve_output=2048) == 12288
    assert ollama.choose_num_ctx("x" * 10_000_000) == ollama.MAX_CTX


def test_pick_default_model():
    assert ollama.pick_default_model(["gpt-oss:20b", "qwen3:14b", "qwen3:30b"]) == "qwen3:30b"
    assert ollama.pick_default_model(["gemma3:12b", "llama3:8b"]) == "gemma3:12b"
    assert ollama.pick_default_model(["zzz"]) == "zzz"
    assert ollama.pick_default_model([]) is None


def _windows(n):
    return [{"index": i, "start": i * 100.0, "end": (i + 1) * 100.0, "text": f"testo {i}"} for i in range(n)]


def test_chapters_from_reduce_repairs_overlaps_and_gaps():
    windows = _windows(5)
    raw = [
        {"title": "B", "content": "c", "from_block": 3, "to_block": 4},
        {"title": "A", "content": "c", "from_block": 1, "to_block": 3},
        {"title": "ignorato", "content": "", "from_block": 9, "to_block": 9},
    ]
    chapters = chapters_from_reduce(raw, windows, blocks=[{}] * 5)
    assert [c["title"] for c in chapters] == ["A", "B"]
    assert (chapters[0]["from_block"], chapters[0]["to_block"]) == (1, 3)
    assert (chapters[1]["from_block"], chapters[1]["to_block"]) == (4, 5)
    assert chapters[0]["start"] == 0.0 and chapters[1]["end"] == 500.0


def test_chapters_from_reduce_fallback_to_blocks():
    windows = _windows(2)
    blocks = [{"title": "Uno", "summary": ["a", "b"]}, {"title": "", "summary": []}]
    chapters = chapters_from_reduce(None, windows, blocks)
    assert [c["title"] for c in chapters] == ["Uno", "Parte 2"]


def test_notes_markdown_demotes_headings():
    pack = {"chapters": [{"title": "Cap", "start": 0, "from_block": 1, "to_block": 1, "content": "x"}], "summary": ["s"], "review_questions": ["q?"]}
    blocks = [{"notes": "# Titolo\n## Sotto\ntesto"}]
    md = build_notes_markdown("Lezione", pack, blocks)
    assert "## [00:00] Cap" in md
    assert "\n### Titolo\n#### Sotto\n" in md
    assert "## Riassunto finale" in md and "1. q?" in md


def test_generate_heuristic_shape():
    text = ("La derivata misura la pendenza della tangente. Il limite definisce la derivata in modo rigoroso. "
            "La derivata e il limite sono i due concetti centrali della lezione di analisi. ") * 20
    windows = [{"index": i, "start": i * 60.0, "end": (i + 1) * 60.0, "text": text} for i in range(20)]
    pack = generate_heuristic(windows=windows, cleaned_text=text, summary_mode="breve")
    assert pack["generator"] == "heuristic"
    assert 1 <= len(pack["summary"]) <= 5
    assert 3 <= len(pack["chapters"]) <= 8
    assert pack["chapters"][0]["start"] == 0.0 and pack["chapters"][-1]["end"] == 1200.0
    assert pack["key_concepts"][0]["term"].lower() in ("derivata", "limite")
    assert pack["review_questions"]


def test_repair_latex_escapes_and_heading_normalization():
    from transcriptor.studypack import _normalize_headings, repair_latex_escapes

    broken = "$$ \x0crac{a}{b} \times x \to 0 $$\n\tindentato \x08eta \nabla"
    fixed = repair_latex_escapes(broken)
    assert "\\frac{a}{b}" in fixed and "\\times" in fixed and "\\to 0" in fixed
    assert "\n\tindentato" in fixed  # il tab a inizio riga resta
    assert "\\beta" in fixed and "\\nabla" in fixed
    assert _normalize_headings("##### A\n###### B\ntesto") == "### A\n#### B\ntesto"
    assert _normalize_headings("# A\n## B") == "### A\n#### B"
