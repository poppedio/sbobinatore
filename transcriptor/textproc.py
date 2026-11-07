"""Pulizia testo, euristiche e suddivisione in paragrafi/finestre."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

Segment = Dict

STOPWORDS_IT = {
    "a", "ad", "al", "alla", "alle", "allo", "ai", "agli", "all", "anche", "ancora",
    "avere", "aveva", "avevano", "basta", "bene", "che", "chi", "ci", "cioè", "come",
    "con", "contro", "cosa", "cui", "da", "dagli", "dai", "dal", "dalla", "dalle",
    "dallo", "de", "dei", "del", "della", "delle", "dello", "di", "dopo", "dove",
    "due", "e", "ed", "era", "erano", "essere", "fa", "faccio", "fare", "fatto",
    "fra", "gli", "ha", "hai", "hanno", "ho", "i", "il", "in", "io", "la", "le",
    "lei", "li", "lo", "loro", "lui", "ma", "mi", "mia", "mie", "miei", "mio",
    "molto", "nei", "nel", "nella", "nelle", "nello", "noi", "non", "nostro", "o",
    "oggi", "ogni", "per", "perché", "più", "poi", "poco", "può", "quale", "quali",
    "quello", "questa", "queste", "questi", "questo", "qui", "quindi", "sarà", "se",
    "sei", "senza", "si", "sia", "siamo", "sono", "sopra", "sotto", "sua", "sue",
    "suoi", "sul", "sulla", "sulle", "sullo", "su", "tra", "tre", "tu", "tua", "tuo",
    "un", "una", "uno", "va", "voi", "vostra", "vostro", "è", "eh", "allora",
    "diciamo", "praticamente", "tipo", "ok", "cioe", "insomma", "dunque", "adesso",
    "proprio", "stesso", "stessa", "quando", "mentre", "perche", "però", "ecco",
    "quindi", "cosi", "così", "qualche", "altro", "altra", "altri", "tutto", "tutti",
    "fatto", "detto", "vedere", "dire", "nostra", "vostre", "sempre", "mai", "già",
}

STOPWORDS_EN = {
    "the", "and", "that", "this", "with", "for", "are", "was", "were", "have", "has",
    "had", "not", "but", "you", "your", "they", "them", "their", "from", "which",
    "what", "when", "where", "who", "how", "why", "can", "could", "would", "should",
    "will", "just", "about", "into", "than", "then", "there", "here", "also", "very",
    "some", "more", "most", "like", "okay", "yeah", "really", "because", "these",
    "those", "been", "being", "does", "did", "doing", "its", "it's", "our", "out",
}

STOPWORDS_BY_LANG = {"it": STOPWORDS_IT, "en": STOPWORDS_EN}

# Solo riempitivi non lessicali: parole come "allora", "cioè", "ok" possono avere valore logico.
FILLERS = [
    r"\b(?:ehm+|ehh+|em+|mh+|mm+|hmm+|uhm+|uh+|ah+)\b",
]


def normalize_spaces(text: str) -> str:
    text = (text or "").replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    cleaned = text or ""
    for pattern in FILLERS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([A-Za-zÀ-ÿ])", r"\1 \2", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str, language: str = "it") -> List[str]:
    stop = STOPWORDS_BY_LANG.get((language or "it")[:2], STOPWORDS_IT)
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']+", (text or "").lower())
    return [w for w in words if w not in stop and len(w) > 2]


def extract_keywords(text: str, top_n: int = 12, language: str = "it") -> List[str]:
    counts = Counter(tokenize(text, language))
    return [w for w, _ in counts.most_common(top_n)]


def extractive_summary(text: str, max_sentences: int = 8, language: str = "it") -> List[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    freq = Counter(tokenize(text, language))
    scored: List[Tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        words = tokenize(sent, language)
        if len(words) < 3:
            continue
        score = sum(freq[w] for w in words) / math.sqrt(len(words))
        scored.append((score, idx, sent))
    top = sorted(scored, reverse=True)[:max_sentences]
    return [s for _, _, s in sorted(top, key=lambda x: x[1])]


def find_evidence_sentence(term: str, sentences: Sequence[str]) -> str:
    pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    for sent in sentences:
        if pattern.search(sent):
            return sent
    return ""


def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9àèéìòù_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "lezione"


def segments_to_text(segments: Sequence[Segment]) -> str:
    return normalize_spaces(" ".join(s.get("text", "") for s in segments if s.get("text")))


def segments_to_paragraphs(
    segments: Sequence[Segment],
    max_chars: int = 1000,
    gap_seconds: float = 2.5,
) -> List[Dict]:
    """Paragrafi con timestamp: nuovo paragrafo su pausa lunga o quando si supera max_chars."""
    paragraphs: List[Dict] = []
    current: List[str] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_len = 0
    prev_end = None

    def flush():
        if current:
            text = clean_text(" ".join(current))
            if text:
                paragraphs.append({"start": cur_start, "end": cur_end, "text": text})

    for seg in segments:
        text = normalize_spaces(seg.get("text", ""))
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        long_gap = prev_end is not None and (start - prev_end) >= gap_seconds
        ends_sentence = bool(current) and current[-1].rstrip()[-1:] in ".!?"
        if current and ((cur_len + len(text) + 1 > max_chars and ends_sentence) or (long_gap and cur_len > 200)):
            flush()
            current = []
            cur_len = 0
        if not current:
            cur_start = start
        current.append(text)
        cur_len += len(text) + 1
        cur_end = end
        prev_end = end
    flush()
    return paragraphs


def build_windows(
    segments: Sequence[Segment],
    target_chars: int = 9000,
    max_chars: int = 12000,
    max_seconds: float = 720.0,
    gap_break_seconds: float = 4.0,
) -> List[Dict]:
    """Finestre di testo consecutive (con start/end) da passare una alla volta al modello linguistico."""
    windows: List[Dict] = []
    current: List[str] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_len = 0
    prev_end = None

    def flush():
        if current:
            windows.append({
                "index": len(windows),
                "start": cur_start,
                "end": cur_end,
                "text": clean_text(" ".join(current)),
            })

    for seg in segments:
        text = normalize_spaces(seg.get("text", ""))
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if current:
            gap = (start - prev_end) if prev_end is not None else 0.0
            too_long = cur_len + len(text) + 1 > max_chars or (end - cur_start) > max_seconds
            natural_break = cur_len >= target_chars * 0.6 and (gap >= gap_break_seconds or cur_len >= target_chars)
            if too_long or natural_break:
                flush()
                current = []
                cur_len = 0
        if not current:
            cur_start = start
        current.append(text)
        cur_len += len(text) + 1
        cur_end = end
        prev_end = end
    flush()
    return windows


def highlight_terms_in_markdown(text: str, terms: Sequence[str], max_terms: int = 40) -> str:
    unique: List[str] = []
    seen = set()
    for term in terms:
        key = (term or "").lower().strip()
        if key and len(key) >= 3 and key not in seen:
            unique.append(term.strip())
            seen.add(key)
    highlighted = text
    for term in sorted(unique[:max_terms], key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w*])({re.escape(term)})(?![\w*])", flags=re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted


_GENERIC_TITLE = re.compile(
    r"^(rec|recording|registrazione|nuova registrazione|new recording|audio|voice|memo|video|vid|img|mov|zoom|teams|meet|"
    r"untitled|senza titolo|lezione|lesson|track|traccia|file|whatsapp audio|ptt)?[\s_\-\.\d:()]*$",
    re.IGNORECASE,
)


def looks_generic_title(title: str) -> bool:
    """True per titoli tipo 'REC0042', 'Nuova registrazione 3', '2024-10-01 10.30' che non dicono nulla della lezione."""
    return bool(_GENERIC_TITLE.match((title or "").strip()))
