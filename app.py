"""Sbobinatore lezioni: interfaccia Streamlit."""
from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import streamlit as st

from transcriptor import __version__, ollama
from transcriptor.audio import bytes_sha256, convert_to_wav16k, ffmpeg_works, file_sha256, load_wav16k
from transcriptor.cache import (
    LessonCache,
    cache_root,
    cache_size_bytes,
    clear_cache,
    list_cached_lessons,
    studypack_key,
)
from transcriptor.export import build_json, build_markdown, build_rtf, build_srt, build_txt
from transcriptor.jobs import Job, find_running, get_job, running_jobs, start_job
from transcriptor.studypack import generate_heuristic, generate_with_ollama
from transcriptor.textproc import (
    build_windows,
    clean_text,
    format_ts,
    highlight_terms_in_markdown,
    segments_to_paragraphs,
    segments_to_text,
    slugify,
)
from transcriptor.transcribe import (
    MODEL_OPTIONS,
    choose_default_model,
    is_apple_silicon_mac,
    mlx_available,
    transcribe_audio,
)

st.set_page_config(page_title="Sbobinatore lezioni", page_icon="🧠", layout="wide")

_WIDE = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}
SUPPORTED_TYPES = ["mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm", "aac", "flac", "ogg", "opus", "aiff"]


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------
@st.cache_data(ttl=20, show_spinner=False)
def ollama_status(base_url: str):
    online = ollama.is_available(base_url)
    return online, (ollama.list_models(base_url) if online else [])


@st.cache_data(show_spinner=False)
def hash_local_file(path: str, size: int, mtime: float) -> str:
    return file_sha256(path)


def human_size(num_bytes) -> str:
    if not num_bytes:
        return "n/d"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def state_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


for _key, _value in {
    "source": None,
    "transcript": None,
    "pack": None,
    "transcribe_job_id": None,
    "pack_job_id": None,
    "upload_uid": None,
    "notice": None,
}.items():
    state_default(_key, _value)


def set_source(file_hash: str, name: str, size: int, path):
    st.session_state["source"] = {"hash": file_hash, "name": name, "size": size, "path": path}
    st.session_state["transcript"] = None
    st.session_state["pack"] = None
    st.session_state["pack_job_id"] = None
    running = find_running("transcribe", file_hash)
    st.session_state["transcribe_job_id"] = running.id if running else None
    loaded = LessonCache(file_hash).load_transcript()
    if loaded:
        st.session_state["transcript"] = {"segments": loaded[0], "meta": loaded[1]}
    st.session_state["lesson_title"] = Path(name).stem.replace("_", " ").strip() or "Lezione"


# ---------------------------------------------------------------------------
# lavori in background
# ---------------------------------------------------------------------------
def run_transcription(job: Job, source: dict, opts: dict) -> dict:
    cache = LessonCache(source["hash"])
    if not cache.wav_path.exists():
        src = source.get("path")
        if not src or not os.path.exists(src):
            raise RuntimeError("File sorgente non trovato: ricarica il file e riprova.")
        job.set_progress("convert", 0, 1, "Converto l'audio in WAV 16 kHz mono con ffmpeg…")
        converter = convert_to_wav16k(src, str(cache.wav_path))
        job.set_progress("convert", 1, 1, f"Audio convertito con {converter}.")
        copy = cache.find_source_copy()
        if copy is not None and Path(src).resolve() == copy.resolve():
            copy.unlink(missing_ok=True)  # la copia dell'upload non serve più
            cache.save_source_info(source["name"], source["size"], None)
    job.set_progress("load", 0, 1, "Carico l'audio in memoria…")
    audio = load_wav16k(str(cache.wav_path))
    segments, meta = transcribe_audio(
        audio,
        backend=opts["backend"],
        model_size=opts["model_size"],
        language=opts["language"],
        device=opts["device"],
        compute_type=opts["compute_type"],
        vad_filter=opts["vad_filter"],
        chunk_target_seconds=opts["chunk_minutes"] * 60.0,
        on_progress=job.set_progress,
        cancel_event=job.cancel_event,
        chunk_store=cache.chunk_store(),
        allow_cpu_fallback=opts["allow_cpu_fallback"],
    )
    meta["source_name"] = source["name"]
    meta["source_size"] = source["size"]
    meta["transcribed_at"] = time.time()
    cache.save_transcript(segments, meta)
    cache.chunk_store().clear()
    return {"segments": segments, "meta": meta}


def run_studypack(job: Job, file_hash: str, segments: list, meta: dict, opts: dict) -> dict:
    windows = build_windows(segments)
    if not windows:
        raise RuntimeError("La trascrizione è vuota: niente da riassumere.")
    if opts["use_ollama"]:
        pack = generate_with_ollama(
            windows=windows,
            title=opts["title"],
            model=opts["model"],
            base_url=opts["base_url"],
            summary_mode=opts["summary_mode"],
            want_notes=opts["want_notes"],
            on_progress=job.set_progress,
            cancel_event=job.cancel_event,
        )
    else:
        job.set_progress("heuristic", 0, 1, "Genero il pacchetto studio con euristiche locali…")
        pack = generate_heuristic(
            windows=windows,
            cleaned_text=clean_text(segments_to_text(segments)),
            summary_mode=opts["summary_mode"],
            language=meta.get("language") or "it",
        )
    if not opts["include_glossary"]:
        pack["glossary"] = []
    if not opts["include_questions"]:
        pack["review_questions"] = []
    pack["title"] = opts["title"]
    pack["key"] = opts["key"]
    pack["created_at"] = time.time()
    LessonCache(file_hash).save_studypack(opts["key"], pack)
    return pack


def render_job_monitor(state_key: str, label: str) -> None:
    @st.fragment(run_every=1.0)
    def monitor():
        job = get_job(st.session_state.get(state_key))
        if job is None:
            return
        if job.running:
            st.progress(job.fraction, text=f"{label} · {job.message} · {format_ts(job.elapsed)} trascorsi")
            cols = st.columns([1, 5])
            if cols[0].button("Annulla", key=f"cancel_{job.id}"):
                job.cancel()
                cols[1].caption("Annullamento richiesto: si ferma alla fine del blocco corrente.")
            with st.expander("Dettagli", expanded=False):
                st.code("\n".join(job.log[-20:]) or "…")
        else:
            st.rerun()

    monitor()


def consume_job(state_key: str, on_done) -> None:
    """Se il job è terminato, sposta il risultato in sessione e libera il riferimento."""
    job = get_job(st.session_state.get(state_key))
    if job is None or job.running:
        return
    st.session_state[state_key] = None
    if job.status == "done":
        on_done(job.result)
    elif job.status == "cancelled":
        st.session_state["notice"] = ("info", "Operazione annullata. I blocchi già completati restano in cache e verranno ripresi.")
    else:
        st.session_state["notice"] = ("error", f"Errore: {job.error}")


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
apple_mac = is_apple_silicon_mac()
with st.sidebar:
    st.header("Impostazioni")
    st.subheader("Trascrizione")
    backend_options = ["auto", "mlx-whisper", "faster-whisper"] if apple_mac else ["faster-whisper"]
    backend = st.selectbox(
        "Backend", backend_options, index=0,
        help="Su Apple Silicon 'auto' usa mlx-whisper (GPU). Se MLX non è disponibile usa faster-whisper.",
    )
    model_size = st.selectbox(
        "Modello Whisper", MODEL_OPTIONS, index=MODEL_OPTIONS.index(choose_default_model()),
        help="large-v3-turbo è il miglior compromesso qualità/velocità per l'italiano.",
    )
    language = st.selectbox("Lingua", ["auto", "it", "en", "es", "fr", "de"], index=1)
    chunk_minutes = st.slider("Durata blocchi (minuti)", min_value=3, max_value=15, value=8,
                              help="L'audio viene tagliato sulle pause in blocchi di questa durata: progresso, annullamento e ripresa lavorano per blocco.")
    vad_filter = st.checkbox("Rileva pause e salta i silenzi (VAD)", value=True)
    with st.expander("Avanzate faster-whisper"):
        device = st.selectbox("Device", ["cpu"] if apple_mac else ["cpu", "cuda"], index=0)
        compute_type = st.selectbox("Compute type", ["int8", "float16", "float32"], index=0)
        allow_cpu_fallback = st.checkbox(
            "Se MLX fallisce, continua con faster-whisper su CPU", value=False,
            help="Disattivato: un errore MLX viene mostrato subito invece di partire con una trascrizione CPU molto più lenta.",
        )
    if apple_mac and backend in ("auto", "mlx-whisper") and not mlx_available():
        st.warning("mlx-whisper non è installato: verrà usato faster-whisper su CPU (molto più lento).")

    st.divider()
    st.subheader("Pacchetto studio")
    summary_mode = st.selectbox("Livello sintesi", ["breve", "standard", "dettagliata"], index=1)
    include_glossary = st.checkbox("Includi glossario", value=True)
    include_questions = st.checkbox("Includi domande di ripasso", value=True)

    st.divider()
    st.subheader("Ollama")
    ollama_base_url = st.text_input("URL", value=ollama.DEFAULT_URL)
    online, ollama_models = ollama_status(ollama_base_url)
    if st.button("Ricontrolla Ollama", key="refresh_ollama"):
        ollama_status.clear()
        online, ollama_models = ollama_status(ollama_base_url)
    if online and ollama_models:
        st.caption(f"✅ Ollama attivo, {len(ollama_models)} modelli")
    elif online:
        st.caption("⚠️ Ollama attivo ma senza modelli: `ollama pull qwen3:30b`")
    else:
        st.caption("⚠️ Ollama non raggiungibile: verrà usata la generazione euristica.")
    use_ollama = st.checkbox("Usa Ollama per sintesi, capitoli e glossario", value=bool(online and ollama_models),
                             disabled=not (online and ollama_models))
    default_model = ollama.pick_default_model(ollama_models)
    ollama_model = st.selectbox(
        "Modello", ollama_models or ["nessuno"],
        index=(ollama_models.index(default_model) if default_model in ollama_models else 0),
        help="Consigliato: qwen3:30b (MoE, veloce su Apple Silicon). Alternativa più leggera: gemma3:12b.",
    )
    want_notes = st.checkbox("Genera anche appunti completi (Markdown)", value=False, disabled=not use_ollama,
                             help="Per ogni blocco il modello scrive appunti discorsivi: più lento ma molto più utile per studiare.")

    st.divider()
    st.subheader("Cache")
    st.caption(f"{human_size(cache_size_bytes())} in `{cache_root()}`")
    if st.checkbox("Mostra opzioni di pulizia", value=False):
        if st.button("Svuota tutta la cache", type="secondary"):
            clear_cache()
            st.session_state["source"] = None
            st.session_state["transcript"] = None
            st.session_state["pack"] = None
            st.rerun()
    st.caption(f"Versione {__version__}")


# ---------------------------------------------------------------------------
# pagina
# ---------------------------------------------------------------------------
st.title("🧠 Sbobinatore lezioni in locale")
st.caption("Trascrizione locale con Whisper, a blocchi e con ripresa; pacchetto studio con Ollama o euristiche; tutto resta sul tuo Mac.")

if not ffmpeg_works():
    st.warning("ffmpeg non è disponibile o non parte (libreria mancante?): la conversione userà PyAV, più lenta. "
               "Su Mac: `brew install ffmpeg` oppure `brew upgrade ffmpeg`.")

notice = st.session_state.pop("notice", None)
if notice:
    getattr(st, notice[0])(notice[1])

# --- 1. sorgente ---
st.subheader("1. Scegli la registrazione")
tab_path, tab_upload, tab_cache = st.tabs(["File sul Mac (consigliato per file grandi)", "Carica file", "Lezioni già elaborate"])

with tab_path:
    st.caption("Incolla il percorso del file: nessun upload, nessuna copia in memoria. Trascina il file nel terminale per ottenere il percorso.")
    local_path = st.text_input("Percorso del file", placeholder="/Users/tuonome/Downloads/lezione.mp4")
    if st.button("Usa questo file", key="use_local"):
        candidate = os.path.expanduser(local_path.strip().strip("'\""))
        if not candidate or not os.path.isfile(candidate):
            st.error("Percorso non valido o file non trovato.")
        else:
            stat = os.stat(candidate)
            with st.spinner("Calcolo l'impronta del file…"):
                file_hash = hash_local_file(candidate, stat.st_size, stat.st_mtime)
            cache = LessonCache(file_hash)
            cache.save_source_info(os.path.basename(candidate), stat.st_size, candidate)
            set_source(file_hash, os.path.basename(candidate), stat.st_size, candidate)
            st.rerun()

with tab_upload:
    uploaded = st.file_uploader("Carica audio o video", type=SUPPORTED_TYPES, accept_multiple_files=False)
    if uploaded is not None:
        uid = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_uid") != uid:
            with st.spinner("Calcolo l'impronta del file e lo salvo in cache…"):
                data = uploaded.getbuffer()
                file_hash = bytes_sha256(data)
                cache = LessonCache(file_hash)
                copy = cache.find_source_copy()
                if copy is None and not cache.wav_path.exists():
                    copy = cache.source_copy_path(Path(uploaded.name).suffix or ".bin")
                    with open(copy, "wb") as handle:
                        handle.write(data)
                cache.save_source_info(uploaded.name, uploaded.size, str(copy) if copy else None)
            st.session_state["upload_uid"] = uid
            set_source(file_hash, uploaded.name, uploaded.size, str(copy) if copy else None)
            st.rerun()

with tab_cache:
    lessons = list_cached_lessons()
    if not lessons:
        st.caption("Nessuna lezione in cache.")
    else:
        labels = [
            f"{item['name']} · {time.strftime('%d/%m/%Y %H:%M', time.localtime(item['saved_at']))} · "
            f"{'trascritta' if item['has_transcript'] else 'non trascritta'} · {item['studypacks']} pacchetti"
            for item in lessons
        ]
        choice = st.selectbox("Lezione", range(len(lessons)), format_func=lambda i: labels[i])
        cols = st.columns([1, 1, 4])
        if cols[0].button("Apri", key="open_cached"):
            item = lessons[choice]
            info = LessonCache(item["hash"]).load_source_info() or {}
            set_source(item["hash"], item["name"], item.get("size") or 0, info.get("path"))
            st.rerun()
        if cols[1].button("Elimina", key="delete_cached"):
            LessonCache(lessons[choice]["hash"]).clear()
            if st.session_state.get("source") and st.session_state["source"]["hash"] == lessons[choice]["hash"]:
                st.session_state["source"] = None
                st.session_state["transcript"] = None
                st.session_state["pack"] = None
            st.rerun()

source = st.session_state.get("source")
if not source:
    others = running_jobs()
    if others:
        st.info(f"{len(others)} elaborazioni in corso in background. Riapri la lezione dalla scheda 'Lezioni già elaborate' per seguirle.")
    st.stop()

cache = LessonCache(source["hash"])
st.info(f"**{source['name']}** · {human_size(source['size'])} · impronta `{source['hash'][:12]}`")

# --- 2. trascrizione ---
st.subheader("2. Trascrizione")


def _on_transcript_done(result):
    st.session_state["transcript"] = result
    st.session_state["pack"] = None


consume_job("transcribe_job_id", _on_transcript_done)
transcribe_job = get_job(st.session_state.get("transcribe_job_id"))
transcript = st.session_state.get("transcript")

transcribe_opts = {
    "backend": backend, "model_size": model_size, "language": language, "device": device,
    "compute_type": compute_type, "vad_filter": vad_filter, "chunk_minutes": chunk_minutes,
    "allow_cpu_fallback": allow_cpu_fallback,
}
cols = st.columns([1, 4])
label = "Ritrascrivi" if transcript else "Trascrivi"
if cols[0].button(label, type="primary", disabled=transcribe_job is not None, **_WIDE):
    if transcript:
        cache.chunk_store().clear()
    job = start_job("transcribe", source["hash"], lambda j, s=dict(source), o=dict(transcribe_opts): run_transcription(j, s, o))
    st.session_state["transcribe_job_id"] = job.id
    st.rerun()
if transcribe_job is not None:
    render_job_monitor("transcribe_job_id", "Trascrizione")
elif transcript:
    meta = transcript["meta"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Durata", format_ts(meta.get("duration") or 0))
    m2.metric("Segmenti", len(transcript["segments"]))
    m3.metric("Lingua", meta.get("language") or "n/d")
    m4.metric("Backend", (meta.get("backend") or "n/d").replace("-whisper", ""))
    rtf = meta.get("realtime_factor")
    m5.metric("Tempo", f"{format_ts(meta.get('elapsed_seconds') or 0)}" + (f" ({rtf}×)" if rtf else ""))
    if meta.get("fallback_reason"):
        st.warning(f"MLX ha fallito e si è usato faster-whisper su CPU: {meta['fallback_reason']}")
else:
    cols[1].caption("Il file verrà convertito con ffmpeg, diviso in blocchi sulle pause e trascritto blocco per blocco. Puoi annullare e riprendere.")

if not transcript:
    st.stop()

segments = transcript["segments"]
meta = transcript["meta"]
paragraphs = segments_to_paragraphs(segments)

# --- 3. pacchetto studio ---
st.subheader("3. Pacchetto studio")
lesson_title = st.text_input("Titolo lezione", key="lesson_title")
pack_opts = {
    "use_ollama": bool(use_ollama and online and ollama_models),
    "model": ollama_model if use_ollama else None,
    "base_url": ollama_base_url,
    "summary_mode": summary_mode,
    "want_notes": bool(want_notes and use_ollama),
    "include_glossary": include_glossary,
    "include_questions": include_questions,
    "title": lesson_title or "Lezione",
}
pack_opts["key"] = studypack_key(
    generator="ollama" if pack_opts["use_ollama"] else "heuristic",
    model=pack_opts["model"], summary_mode=summary_mode, want_notes=pack_opts["want_notes"],
    glossary=include_glossary, questions=include_questions, title=pack_opts["title"],
    segments=len(segments), transcribed_at=meta.get("transcribed_at"),
)


def _on_pack_done(result):
    st.session_state["pack"] = result


consume_job("pack_job_id", _on_pack_done)
pack_job = get_job(st.session_state.get("pack_job_id"))
cached_pack = cache.load_studypack(pack_opts["key"])

cols = st.columns([1, 1, 3])
gen_label = f"Genera con {ollama_model}" if pack_opts["use_ollama"] else "Genera (euristico)"
if cols[0].button(gen_label, type="primary", disabled=pack_job is not None, **_WIDE):
    job = start_job(
        "studypack", f"{source['hash']}:{pack_opts['key']}",
        lambda j, h=source["hash"], s=segments, m=dict(meta), o=dict(pack_opts): run_studypack(j, h, s, m, o),
    )
    st.session_state["pack_job_id"] = job.id
    st.rerun()
if cached_pack and cols[1].button("Carica dalla cache", **_WIDE):
    st.session_state["pack"] = cached_pack
    st.rerun()
if pack_job is not None:
    render_job_monitor("pack_job_id", "Pacchetto studio")
elif cached_pack and not st.session_state.get("pack"):
    cols[2].caption("Esiste già un pacchetto con queste impostazioni: caricalo dalla cache oppure rigeneralo.")

pack = st.session_state.get("pack")

# --- 4. risultati ---
st.subheader("4. Risultati")
tabs = st.tabs(["Pacchetto studio", "Appunti AI", "Testo completo", "Trascrizione", "Metadati"])

with tabs[0]:
    if not pack:
        st.caption("Genera il pacchetto studio per vedere sintesi, concetti, capitoli, glossario e domande.")
    else:
        if pack.get("warnings"):
            for warning in pack["warnings"]:
                st.warning(warning)
        st.markdown("#### Sintesi")
        for item in pack.get("summary", []):
            st.markdown(f"- {item}")
        st.markdown("#### Concetti chiave")
        for item in pack.get("key_concepts", []):
            st.markdown(f"**{item['term']}** — {item.get('explanation', '')}")
        st.markdown("#### Capitoli")
        for chapter in pack.get("chapters", []):
            with st.expander(f"[{format_ts(chapter.get('start', 0))}–{format_ts(chapter.get('end', 0))}] {chapter['title']}"):
                st.write(chapter.get("content", ""))
        if pack.get("glossary"):
            st.markdown("#### Glossario")
            for item in pack["glossary"]:
                st.markdown(f"- **{item['term']}**: {item.get('definition', '')}")
        if pack.get("review_questions"):
            st.markdown("#### Domande di ripasso")
            for i, question in enumerate(pack["review_questions"], start=1):
                st.markdown(f"{i}. {question}")

with tabs[1]:
    if pack and pack.get("notes_md"):
        st.markdown(pack["notes_md"])
    else:
        st.caption("Attiva 'Genera anche appunti completi' nella sidebar e rigenera il pacchetto con Ollama.")

with tabs[2]:
    show_ts = st.checkbox("Mostra i minuti", value=True, key="show_ts_full")
    terms = [c["term"] for c in (pack or {}).get("key_concepts", [])] + [g["term"] for g in (pack or {}).get("glossary", [])]
    for para in paragraphs:
        text = highlight_terms_in_markdown(para["text"], terms) if terms else para["text"]
        stamp = f"**[{format_ts(para['start'])}]** " if show_ts else ""
        st.markdown(stamp + text)

with tabs[3]:
    st.caption("Segmenti grezzi di Whisper, con tempi di inizio e fine.")
    st.dataframe(
        [{"inizio": format_ts(s["start"]), "fine": format_ts(s["end"]), "testo": s["text"]} for s in segments],
        hide_index=True, **_WIDE,
    )

with tabs[4]:
    st.json({"transcript": meta, "study_pack": {k: v for k, v in (pack or {}).items() if k in ("generator", "model", "ollama_stats", "warnings", "created_at")}})

# --- 5. export ---
st.subheader("5. Esporta")
base = slugify(lesson_title or source["name"])
export_pack = pack or {}
with_ts = st.checkbox("Timestamp negli export", value=True, key="export_ts")
md_doc = build_markdown(pack_opts["title"], paragraphs, export_pack, with_timestamps=with_ts)
rtf_doc = build_rtf(pack_opts["title"], paragraphs, export_pack, with_timestamps=with_ts)
cols = st.columns(3)
cols[0].download_button("Pacchetto studio .rtf", data=rtf_doc, file_name=f"{base}_studio.rtf", mime="application/rtf", **_WIDE)
cols[1].download_button("Pacchetto studio .md", data=md_doc.encode("utf-8"), file_name=f"{base}_studio.md", mime="text/markdown", **_WIDE)
if pack and pack.get("notes_md"):
    cols[2].download_button("Appunti AI .md", data=pack["notes_md"].encode("utf-8"), file_name=f"{base}_appunti.md", mime="text/markdown", **_WIDE)
cols = st.columns(3)
cols[0].download_button("Trascrizione .txt", data=build_txt(paragraphs, with_ts).encode("utf-8"), file_name=f"{base}_trascrizione.txt", mime="text/plain", **_WIDE)
cols[1].download_button("Sottotitoli .srt", data=build_srt(segments).encode("utf-8"), file_name=f"{base}.srt", mime="text/plain", **_WIDE)
cols[2].download_button("Dati completi .json", data=build_json(segments, meta, pack).encode("utf-8"), file_name=f"{base}.json", mime="application/json", **_WIDE)
