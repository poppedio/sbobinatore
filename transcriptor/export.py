"""Export: Markdown, RTF, TXT con timestamp, SRT/VTT e JSON."""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

from .textproc import format_ts

Segment = Dict


def build_markdown(title: str, paragraphs: Sequence[Dict], pack: Dict, with_timestamps: bool = True) -> str:
    md: List[str] = [f"# {title}", ""]

    if pack.get("summary"):
        md.append("## Sintesi")
        md.extend(f"- {item}" for item in pack["summary"])
        md.append("")

    if pack.get("key_concepts"):
        md.append("## Concetti chiave")
        for item in pack["key_concepts"]:
            md.append(f"**{item['term']}** — {item.get('explanation', '')}")
            md.append("")

    if pack.get("chapters"):
        md.append("## Capitoli")
        for chapter in pack["chapters"]:
            stamp = f"[{format_ts(chapter.get('start', 0))}] " if with_timestamps else ""
            md.append(f"### {stamp}{chapter['title']}")
            md.append(chapter.get("content", ""))
            md.append("")

    if pack.get("glossary"):
        md.append("## Glossario")
        md.extend(f"- **{item['term']}**: {item.get('definition', '')}" for item in pack["glossary"])
        md.append("")

    if pack.get("review_questions"):
        md.append("## Domande di ripasso")
        md.extend(f"{i}. {q}" for i, q in enumerate(pack["review_questions"], start=1))
        md.append("")

    md.append("## Testo completo")
    md.append("")
    for para in paragraphs:
        stamp = f"**[{format_ts(para['start'])}]** " if with_timestamps else ""
        md.append(f"{stamp}{para['text']}")
        md.append("")
    return "\n".join(md)


# --- RTF ---
def rtf_escape(text: str) -> str:
    parts: List[str] = []
    for ch in text or "":
        if ch == "\\":
            parts.append(r"\\")
        elif ch == "{":
            parts.append(r"\{")
        elif ch == "}":
            parts.append(r"\}")
        elif ch == "\n":
            parts.append(r"\line ")
        else:
            code = ord(ch)
            if code > 0xFFFF:
                code -= 0x10000
                hi, lo = 0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)
                parts.append(f"\\u{hi - 65536}?\\u{lo - 65536}?")
            elif code > 127:
                parts.append(f"\\u{code if code <= 32767 else code - 65536}?")
            else:
                parts.append(ch)
    return "".join(parts)


def build_rtf(title: str, paragraphs: Sequence[Dict], pack: Dict, with_timestamps: bool = True) -> bytes:
    rtf: List[str] = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Helvetica;}}", "\n", r"\fs24", "\n"]

    def heading(text: str, size: int = 30) -> None:
        rtf.append(rf"\pard\sb240\sa120\fs{size}\b " + rtf_escape(text) + r"\b0\fs24\par" + "\n")

    def para(text: str, bold_prefix: str = "") -> None:
        prefix = (r"\b " + rtf_escape(bold_prefix) + r"\b0 ") if bold_prefix else ""
        rtf.append(r"\pard\sa120 " + prefix + rtf_escape(text) + r"\par" + "\n")

    rtf.append(r"\pard\sa240\fs40\b " + rtf_escape(title) + r"\b0\fs24\par" + "\n")

    if pack.get("summary"):
        heading("Sintesi")
        for item in pack["summary"]:
            para("• " + item)
    if pack.get("key_concepts"):
        heading("Concetti chiave")
        for item in pack["key_concepts"]:
            para(item.get("explanation", ""), bold_prefix=item["term"] + ":")
    if pack.get("chapters"):
        heading("Capitoli")
        for chapter in pack["chapters"]:
            stamp = f"[{format_ts(chapter.get('start', 0))}] " if with_timestamps else ""
            heading(stamp + chapter["title"], size=26)
            for piece in chapter.get("content", "").split("\n\n"):
                if piece.strip():
                    para(piece.strip())
    if pack.get("glossary"):
        heading("Glossario")
        for item in pack["glossary"]:
            para(item.get("definition", ""), bold_prefix=item["term"] + ":")
    if pack.get("review_questions"):
        heading("Domande di ripasso")
        for i, q in enumerate(pack["review_questions"], start=1):
            para(f"{i}. {q}")

    heading("Testo completo")
    for p in paragraphs:
        stamp = f"[{format_ts(p['start'])}] " if with_timestamps else ""
        para(p["text"], bold_prefix=stamp.strip())
    rtf.append("}")
    return "".join(rtf).encode("utf-8")


# --- testo e sottotitoli ---
def build_txt(paragraphs: Sequence[Dict], with_timestamps: bool = True) -> str:
    lines = []
    for p in paragraphs:
        stamp = f"[{format_ts(p['start'])}] " if with_timestamps else ""
        lines.append(f"{stamp}{p['text']}")
    return "\n\n".join(lines) + "\n"


def _srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments: Sequence[Segment]) -> str:
    out: List[str] = []
    for i, seg in enumerate(segments, start=1):
        out.append(str(i))
        out.append(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def build_vtt(segments: Sequence[Segment]) -> str:
    out: List[str] = ["WEBVTT", ""]
    for seg in segments:
        out.append(f"{_srt_time(seg['start']).replace(',', '.')} --> {_srt_time(seg['end']).replace(',', '.')}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def build_json(segments: Sequence[Segment], meta: Dict, pack: Dict = None) -> str:
    payload = {"meta": meta, "segments": list(segments)}
    if pack:
        payload["study_pack"] = {k: v for k, v in pack.items() if k != "blocks"}
    return json.dumps(payload, ensure_ascii=False, indent=1)
