"""Pacchetto studio: map/reduce con Ollama sui blocchi della lezione, oppure euristiche locali."""
from __future__ import annotations

import re
import threading
from typing import Callable, Dict, List, Optional, Sequence

from . import ollama
from .textproc import (
    extract_keywords,
    extractive_summary,
    find_evidence_sentence,
    format_ts,
    normalize_spaces,
    sentence_split,
)

ProgressFn = Callable[[str, float, float, str], None]

SUMMARY_POINTS = {"breve": "4-6", "standard": "6-8", "dettagliata": "10-14"}

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_TERM_EXPL = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"term": {"type": "string"}, "explanation": {"type": "string"}},
        "required": ["term", "explanation"],
    },
}
_TERM_DEF = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
        "required": ["term", "definition"],
    },
}


def map_schema(want_notes: bool) -> Dict:
    props = {"title": {"type": "string"}, "summary": _STR_LIST, "concepts": _TERM_EXPL, "terms": _TERM_DEF}
    required = ["title", "summary", "concepts", "terms"]
    if want_notes:
        props["notes"] = {"type": "string"}
        required.append("notes")
    return {"type": "object", "properties": props, "required": required}


REDUCE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": _STR_LIST,
        "key_concepts": _TERM_EXPL,
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "from_block": {"type": "integer"},
                    "to_block": {"type": "integer"},
                },
                "required": ["title", "content", "from_block", "to_block"],
            },
        },
        "glossary": _TERM_DEF,
        "review_questions": _STR_LIST,
    },
    "required": ["summary", "key_concepts", "chapters", "glossary", "review_questions"],
}

MAP_SYSTEM = (
    "Sei un assistente che prepara materiale di studio da trascrizioni di lezioni universitarie. "
    "Ricevi UN blocco della lezione alla volta. Usa solo informazioni presenti nel blocco, non inventare nulla. "
    "Scrivi nella stessa lingua della lezione (di norma italiano), in modo chiaro e preciso. "
    "Rispondi esclusivamente con JSON conforme allo schema richiesto."
)

REDUCE_SYSTEM = (
    "Sei un tutor universitario. Ricevi le sintesi dei blocchi consecutivi di una lezione e devi produrre "
    "il pacchetto studio finale. Usa solo informazioni presenti nelle sintesi, non inventare. "
    "Scrivi nella stessa lingua delle sintesi. Rispondi esclusivamente con JSON conforme allo schema richiesto."
)


def _map_prompt(title: str, window: Dict, index: int, total: int, want_notes: bool) -> str:
    lines = [
        f"Lezione: {title}",
        f"Blocco {index}/{total}, dal minuto {format_ts(window['start'])} al minuto {format_ts(window['end'])}.",
        "",
        "Produci:",
        "- title: titolo breve dell'argomento del blocco (massimo 8 parole, niente 'Blocco N')",
        "- summary: da 3 a 6 punti con le idee principali, in ordine",
        "- concepts: da 2 a 6 concetti importanti, ciascuno con spiegazione di 1-2 frasi",
        "- terms: da 0 a 6 termini tecnici con definizione breve (solo se davvero presenti)",
    ]
    if want_notes:
        lines.append(
            "- notes: appunti discorsivi in Markdown del blocco, pronti per studiare: sottotitoli (###), "
            "definizioni evidenziate, formule, esempi fatti a lezione, punti critici. Tra 150 e 400 parole. "
            "Non ripetere il titolo del blocco. Per le formule usa LaTeX tra $...$ scrivendo le barre rovesciate "
            "raddoppiate (\\\\frac, \\\\times) perché il testo è dentro una stringa JSON."
        )
    lines += ["", "Testo del blocco:", window["text"]]
    return "\n".join(lines)


def _reduce_prompt(title: str, summary_mode: str, blocks: List[Dict], windows: Sequence[Dict]) -> str:
    points = SUMMARY_POINTS.get(summary_mode, SUMMARY_POINTS["standard"])
    lines = [f"Lezione: {title}", f"Numero di blocchi: {len(blocks)}", ""]
    for i, (block, window) in enumerate(zip(blocks, windows), start=1):
        lines.append(f"[Blocco {i} | {format_ts(window['start'])}–{format_ts(window['end'])}] {block.get('title', '')}")
        for item in block.get("summary", []):
            lines.append(f"  - {item}")
        concepts = "; ".join(f"{c['term']}: {c['explanation']}" for c in block.get("concepts", []))
        if concepts:
            lines.append(f"  Concetti: {concepts}")
        terms = "; ".join(f"{t['term']}: {t['definition']}" for t in block.get("terms", []))
        if terms:
            lines.append(f"  Termini: {terms}")
        lines.append("")
    lines += [
        "Produci:",
        f"- summary: {points} punti che coprono TUTTA la lezione, in ordine cronologico",
        "- key_concepts: i 6-10 concetti più importanti dell'intera lezione, con spiegazione di 1-3 frasi",
        "- chapters: da 3 a 8 capitoli che raggruppano blocchi CONSECUTIVI per argomento. Per ciascuno: title, "
        "content (paragrafo discorsivo di 3-6 frasi utile per studiare), from_block e to_block (numeri di blocco, "
        "da 1 in su; i capitoli devono essere in ordine, senza sovrapposizioni e coprire tutti i blocchi)",
        "- glossary: da 5 a 12 termini tecnici con definizione",
        "- review_questions: da 6 a 10 domande di ripasso, dalle più semplici alle più difficili",
    ]
    return "\n".join(lines)


def _clean_list(items) -> List[str]:
    if not isinstance(items, list):
        return []
    return [normalize_spaces(str(x)) for x in items if normalize_spaces(str(x))]


def _clean_pairs(items, second: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        term = normalize_spaces(str(item.get("term", "")))
        other = normalize_spaces(str(item.get(second, "")))
        if term:
            out.append({"term": term, second: other})
    return out


def _normalize_block(raw: Dict) -> Dict:
    return {
        "title": normalize_spaces(str(raw.get("title", ""))),
        "summary": _clean_list(raw.get("summary")),
        "concepts": _clean_pairs(raw.get("concepts"), "explanation"),
        "terms": _clean_pairs(raw.get("terms"), "definition"),
        "notes": str(raw.get("notes", "") or "").strip(),
    }


def chapters_from_reduce(raw_chapters, windows: Sequence[Dict], blocks: List[Dict]) -> List[Dict]:
    """Converte from_block/to_block in timestamp reali; ripara indici incoerenti."""
    n = len(windows)
    chapters: List[Dict] = []
    if isinstance(raw_chapters, list):
        for item in raw_chapters:
            if not isinstance(item, dict):
                continue
            title = normalize_spaces(str(item.get("title", "")))
            content = str(item.get("content", "")).strip()
            try:
                a = int(item.get("from_block", 0))
                b = int(item.get("to_block", a))
            except (TypeError, ValueError):
                continue
            a, b = sorted((max(1, min(n, a)), max(1, min(n, b))))
            if title and content:
                chapters.append({"title": title, "content": content, "from_block": a, "to_block": b})
    chapters.sort(key=lambda c: (c["from_block"], c["to_block"]))
    # elimina sovrapposizioni mantenendo l'ordine
    fixed: List[Dict] = []
    next_free = 1
    for chapter in chapters:
        a = max(chapter["from_block"], next_free)
        b = max(chapter["to_block"], a)
        if a > n:
            break
        chapter["from_block"], chapter["to_block"] = a, min(b, n)
        fixed.append(chapter)
        next_free = chapter["to_block"] + 1
    if fixed and next_free <= n:
        fixed[-1]["to_block"] = n
    if not fixed:
        fixed = [
            {
                "title": block.get("title") or f"Parte {i}",
                "content": " ".join(block.get("summary", [])),
                "from_block": i,
                "to_block": i,
            }
            for i, block in enumerate(blocks, start=1)
        ]
    for chapter in fixed:
        chapter["start"] = windows[chapter["from_block"] - 1]["start"]
        chapter["end"] = windows[chapter["to_block"] - 1]["end"]
    return fixed


_LATEX_FIXES = [
    (re.compile(r"\x0c(?=[A-Za-z])"), r"\\f"),           # \frac, \forall
    (re.compile(r"(?<!\n)\t(?=[A-Za-z])"), r"\\t"),      # \times, \to, \theta (non a inizio riga)
    (re.compile(r"\x08(?=[A-Za-z])"), r"\\b"),           # \beta, \bar
    (re.compile(r"\x0b(?=[A-Za-z])"), r"\\v"),           # \vec
    (re.compile(r"\r(?=ight|ho\b|angle)"), r"\\r"),     # \right, \rho
    (re.compile(r"\n(?=abla|eq\b|otin\b)"), r"\\n"),   # \nabla, \neq
]


def repair_latex_escapes(text: str) -> str:
    """I modelli scrivono \\frac dentro JSON come \f: ripristina i comandi LaTeX più comuni."""
    for pattern, repl in _LATEX_FIXES:
        text = pattern.sub(repl, text)
    return text


def _normalize_headings(markdown: str, base_level: int = 3) -> str:
    """Porta il titolo più alto degli appunti di un blocco a base_level, mantenendo la gerarchia relativa."""
    levels = [len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s+", markdown, flags=re.MULTILINE)]
    if not levels:
        return markdown
    shift = base_level - min(levels)

    def repl(match):
        return "#" * max(1, min(6, len(match.group(1)) + shift)) + " "

    return re.sub(r"^(#{1,6})\s+", repl, markdown, flags=re.MULTILINE)


def build_notes_markdown(title: str, pack: Dict, blocks: List[Dict]) -> str:
    lines = [f"# {title}", ""]
    for chapter in pack.get("chapters", []):
        lines.append(f"## [{format_ts(chapter.get('start', 0))}] {chapter['title']}")
        lines.append("")
        for idx in range(chapter.get("from_block", 1), chapter.get("to_block", 0) + 1):
            if 1 <= idx <= len(blocks):
                notes = blocks[idx - 1].get("notes", "")
                if notes:
                    lines.append(_normalize_headings(repair_latex_escapes(notes)))
                    lines.append("")
        if not any(blocks[i - 1].get("notes") for i in range(chapter.get("from_block", 1), chapter.get("to_block", 0) + 1) if 1 <= i <= len(blocks)):
            lines.append(chapter.get("content", ""))
            lines.append("")
    if pack.get("summary"):
        lines.append("## Riassunto finale")
        lines.extend(f"- {item}" for item in pack["summary"])
        lines.append("")
    if pack.get("review_questions"):
        lines.append("## Domande di ripasso")
        lines.extend(f"{i}. {q}" for i, q in enumerate(pack["review_questions"], start=1))
        lines.append("")
    return "\n".join(lines)


def generate_with_ollama(
    *,
    windows: Sequence[Dict],
    title: str,
    model: str,
    base_url: str,
    summary_mode: str = "standard",
    want_notes: bool = False,
    on_progress: Optional[ProgressFn] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict:
    def progress(stage, done, total, message):
        if on_progress:
            on_progress(stage, done, total, message)

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Generazione annullata.")

    if not windows:
        raise ValueError("Nessun testo da elaborare.")

    total_steps = len(windows) + 1
    blocks: List[Dict] = []
    warnings: List[str] = []
    stats_all: List[Dict] = []
    schema = map_schema(want_notes)
    for i, window in enumerate(windows, start=1):
        check_cancel()
        progress("map", i - 1, total_steps, f"Ollama: blocco {i}/{len(windows)} ({format_ts(window['start'])}–{format_ts(window['end'])})")
        raw, stats = ollama.chat_json(
            base_url=base_url, model=model, system=MAP_SYSTEM,
            user=_map_prompt(title, window, i, len(windows), want_notes),
            schema=schema, reserve_output=3000 if want_notes else 1500,
        )
        if stats.get("truncated"):
            warnings.append(f"Blocco {i}: il prompt potrebbe essere stato troncato (contesto {stats['num_ctx']}).")
        stats_all.append(stats)
        blocks.append(_normalize_block(raw))

    check_cancel()
    progress("reduce", len(windows), total_steps, "Ollama: unisco i blocchi nel pacchetto studio finale…")
    raw_final, stats = ollama.chat_json(
        base_url=base_url, model=model, system=REDUCE_SYSTEM,
        user=_reduce_prompt(title, summary_mode, blocks, windows),
        schema=REDUCE_SCHEMA, reserve_output=4000,
    )
    if stats.get("truncated"):
        warnings.append("Fase finale: il prompt potrebbe essere stato troncato.")
    stats_all.append(stats)

    pack = {
        "summary": _clean_list(raw_final.get("summary")),
        "key_concepts": _clean_pairs(raw_final.get("key_concepts"), "explanation"),
        "chapters": chapters_from_reduce(raw_final.get("chapters"), windows, blocks),
        "glossary": _clean_pairs(raw_final.get("glossary"), "definition"),
        "review_questions": _clean_list(raw_final.get("review_questions")),
        "blocks": blocks,
        "generator": "ollama",
        "model": model,
        "warnings": warnings,
        "ollama_stats": {
            "calls": len(stats_all),
            "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in stats_all),
            "output_tokens": sum(s.get("output_tokens") or 0 for s in stats_all),
            "seconds": round(sum(s.get("total_seconds") or 0 for s in stats_all), 1),
        },
    }
    if not pack["glossary"]:
        seen = set()
        for block in blocks:
            for term in block["terms"]:
                if term["term"].lower() not in seen:
                    seen.add(term["term"].lower())
                    pack["glossary"].append(term)
    for chapter in pack["chapters"]:
        chapter["content"] = repair_latex_escapes(chapter["content"])
    for item in pack["key_concepts"]:
        item["explanation"] = repair_latex_escapes(item["explanation"])
    pack["notes_md"] = build_notes_markdown(title, pack, blocks) if want_notes else ""
    progress("done", total_steps, total_steps, "Pacchetto studio pronto.")
    return pack


def generate_heuristic(
    *,
    windows: Sequence[Dict],
    cleaned_text: str,
    summary_mode: str = "standard",
    language: str = "it",
) -> Dict:
    lang = (language or "it")[:2]
    n_points = {"breve": 5, "standard": 7, "dettagliata": 12}.get(summary_mode, 7)
    summary = extractive_summary(cleaned_text, max_sentences=n_points, language=lang)

    sentences = sentence_split(cleaned_text)
    key_concepts: List[Dict[str, str]] = []
    for kw in extract_keywords(cleaned_text, top_n=8, language=lang):
        evidence = find_evidence_sentence(kw, sentences)
        key_concepts.append({"term": kw.capitalize(), "explanation": evidence or "Concetto ricorrente nella lezione."})

    # capitoli: raggruppa finestre consecutive fino a un massimo di 8
    groups: List[List[Dict]] = []
    per_group = max(1, -(-len(windows) // 8))
    for i in range(0, len(windows), per_group):
        groups.append(list(windows[i: i + per_group]))
    chapters: List[Dict] = []
    for i, group in enumerate(groups, start=1):
        text = " ".join(w["text"] for w in group)
        keywords = extract_keywords(text, top_n=3, language=lang)
        title = f"Parte {i}" + (f": {', '.join(k.capitalize() for k in keywords)}" if keywords else "")
        content = " ".join(extractive_summary(text, max_sentences=4, language=lang)) or text[:600]
        chapters.append({
            "title": title,
            "content": content,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "from_block": group[0]["index"] + 1,
            "to_block": group[-1]["index"] + 1,
        })

    questions: List[str] = []
    for item in key_concepts[:5]:
        questions.append(f"Che cos'è {item['term']} e perché è importante nella lezione?")
    if len(key_concepts) >= 2:
        questions.append(f"Come si collegano {key_concepts[0]['term']} e {key_concepts[1]['term']} nel ragionamento della lezione?")
    for chapter in chapters[:3]:
        questions.append(f"Quali sono le idee principali della sezione \"{chapter['title']}\"?")

    return {
        "summary": summary,
        "key_concepts": key_concepts,
        "chapters": chapters,
        "glossary": [{"term": c["term"], "definition": c["explanation"]} for c in key_concepts[:6]],
        "review_questions": questions[:8],
        "blocks": [],
        "generator": "heuristic",
        "model": None,
        "warnings": ["Pacchetto generato con euristiche locali: attiva Ollama per capitoli e sintesi di qualità."],
        "notes_md": "",
    }
