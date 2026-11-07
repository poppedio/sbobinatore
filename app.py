"""Sbobinatore: interfaccia Streamlit."""
from __future__ import annotations

import inspect
import os
import platform
import time
from pathlib import Path

import pandas as pd
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
from transcriptor.errors import friendly_error
from transcriptor.export import build_docx, build_json, build_markdown, build_rtf, build_srt, build_txt
from transcriptor.hardware import (
    MODEL_QUALITY,
    accelerator_key,
    accelerator_label,
    cuda_available,
    estimate_seconds,
    format_duration_rough,
    hardware_summary,
    media_duration_seconds,
    ollama_models_that_fit,
    ram_gb,
    record_speed,
    suggest_ollama_pull,
)
from transcriptor.jobs import Job, find_running, get_job, running_jobs, start_job
from transcriptor.studypack import generate_heuristic, generate_with_ollama
from transcriptor.textproc import (
    build_windows,
    clean_text,
    format_ts,
    highlight_terms_in_markdown,
    looks_generic_title,
    segments_to_paragraphs,
    segments_to_text,
    slugify,
)
from transcriptor.transcribe import (
    MODEL_DOWNLOAD_GB,
    MODEL_OPTIONS,
    choose_default_model,
    is_apple_silicon_mac,
    mlx_available,
    model_is_downloaded,
    resolve_backend,
    transcribe_audio,
)

st.set_page_config(page_title="Sbobinatore", layout="wide")

_WIDE = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}
SUPPORTED_TYPES = ["mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm", "aac", "flac", "ogg", "opus", "aiff"]
LANGUAGES = {"it": "Italiano", "en": "Inglese", "es": "Spagnolo", "fr": "Francese", "de": "Tedesco", "auto": "Rileva da sola"}
IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------
@st.cache_data(ttl=20, show_spinner=False)
def ollama_status(base_url: str):
    online = ollama.is_available(base_url)
    return online, (ollama.list_models_detailed(base_url) if online else [])


@st.cache_data(show_spinner=False)
def hash_local_file(path: str, size: int, mtime: float) -> str:
    return file_sha256(path)


@st.cache_data(show_spinner=False)
def probe_duration(path: str, size: int, mtime: float):
    return media_duration_seconds(path)


def human_size(num_bytes) -> str:
    if not num_bytes:
        return "n/d"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def clean_path(raw: str) -> str:
    return os.path.expanduser(raw.strip().strip("'\""))


def file_duration(path) -> float:
    if not path or not os.path.isfile(path):
        return 0.0
    stat = os.stat(path)
    return probe_duration(path, stat.st_size, stat.st_mtime) or 0.0


def register_local_file(path: str) -> dict:
    """Calcola impronta e durata di un file locale e lo registra in cache. Ritorna la sorgente."""
    stat = os.stat(path)
    file_hash = hash_local_file(path, stat.st_size, stat.st_mtime)
    LessonCache(file_hash).save_source_info(os.path.basename(path), stat.st_size, path)
    return {"hash": file_hash, "name": os.path.basename(path), "size": stat.st_size, "path": path,
            "duration": file_duration(path)}


def register_upload(uploaded) -> dict:
    data = uploaded.getbuffer()
    file_hash = bytes_sha256(data)
    cache = LessonCache(file_hash)
    copy = cache.find_source_copy()
    if copy is None and not cache.wav_path.exists():
        copy = cache.source_copy_path(Path(uploaded.name).suffix or ".bin")
        with open(copy, "wb") as handle:
            handle.write(data)
    cache.save_source_info(uploaded.name, uploaded.size, str(copy) if copy else None)
    return {"hash": file_hash, "name": uploaded.name, "size": uploaded.size, "path": str(copy) if copy else None,
            "duration": file_duration(str(copy) if copy else None)}


for _key, _value in {
    "source": None, "transcript": None, "pack": None,
    "transcribe_job_id": None, "pack_job_id": None, "queue_job_id": None,
    "upload_uid": None, "queue_upload_uid": None, "notice": None, "queue": [],
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


def set_source(source: dict) -> None:
    st.session_state["source"] = dict(source)
    st.session_state["transcript"] = None
    st.session_state["pack"] = None
    st.session_state["pack_job_id"] = None
    running = find_running("transcribe", source["hash"])
    st.session_state["transcribe_job_id"] = running.id if running else None
    loaded = LessonCache(source["hash"]).load_transcript()
    if loaded:
        st.session_state["transcript"] = {"segments": loaded[0], "meta": loaded[1]}
    st.session_state["lesson_title"] = Path(source["name"]).stem.replace("_", " ").strip() or "Lezione"


def notify(kind: str, text: str, hint: str = "", detail: str = "") -> None:
    st.session_state["notice"] = (kind, text, hint, detail)


# ---------------------------------------------------------------------------
# lavori in background
# ---------------------------------------------------------------------------
def run_transcription(job, source: dict, opts: dict) -> dict:
    cache = LessonCache(source["hash"])
    if not cache.wav_path.exists():
        src = source.get("path")
        if not src or not os.path.exists(src):
            raise RuntimeError("File sorgente non trovato: ricarica il file e riprova.")
        job.set_progress("convert", 0, 1, "Converto l'audio in WAV 16 kHz mono…")
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
    meta["model_size"] = opts["model_size"]
    meta["transcribed_at"] = time.time()
    cache.save_transcript(segments, meta)
    cache.chunk_store().clear()
    record_speed(meta)
    return {"segments": segments, "meta": meta}


def run_studypack(job, file_hash: str, segments: list, meta: dict, opts: dict) -> dict:
    windows = build_windows(segments)
    if not windows:
        raise RuntimeError("La trascrizione è vuota: niente da riassumere.")
    if opts["use_ollama"]:
        pack = generate_with_ollama(
            windows=windows, title=opts["title"], model=opts["model"], base_url=opts["base_url"],
            summary_mode=opts["summary_mode"], want_notes=opts["want_notes"],
            on_progress=job.set_progress, cancel_event=job.cancel_event,
        )
    else:
        job.set_progress("heuristic", 0, 1, "Genero il pacchetto studio con euristiche locali…")
        pack = generate_heuristic(
            windows=windows, cleaned_text=clean_text(segments_to_text(segments)),
            summary_mode=opts["summary_mode"], language=meta.get("language") or "it",
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


def pack_key_for(pack_opts: dict, segments_count: int, transcribed_at) -> str:
    return studypack_key(
        generator="ollama" if pack_opts["use_ollama"] else "heuristic",
        model=pack_opts["model"], summary_mode=pack_opts["summary_mode"], want_notes=pack_opts["want_notes"],
        glossary=pack_opts["include_glossary"], questions=pack_opts["include_questions"], title=pack_opts["title"],
        segments=segments_count, transcribed_at=transcribed_at,
    )


class _JobView:
    """Vista di un job per una singola lezione della coda: riporta il progresso nel job complessivo."""

    def __init__(self, job: Job, index: int, total: int, name: str, span):
        self.job, self.index, self.total, self.name, self.span = job, index, total, name, span
        self.cancel_event = job.cancel_event

    def set_progress(self, stage, done, total, message):
        frac = (float(done) / float(total)) if total else 0.0
        lo, hi = self.span
        self.job.set_progress(stage, self.index + lo + frac * (hi - lo), self.total,
                              f"Lezione {self.index + 1}/{self.total} · {self.name} · {message}")


def run_queue(job: Job, items: list, opts: dict, pack_opts: dict, make_pack: bool) -> list:
    results = []
    total = len(items)
    for index, src in enumerate(items):
        if job.cancel_event.is_set():
            break
        cache = LessonCache(src["hash"])
        try:
            loaded = cache.load_transcript()
            if loaded:
                segments, meta = loaded
                job.set_progress("skip", index, total, f"Lezione {index + 1}/{total} · {src['name']} · già trascritta.")
            else:
                view = _JobView(job, index, total, src["name"], (0.0, 0.8 if make_pack else 1.0))
                result = run_transcription(view, src, opts)
                segments, meta = result["segments"], result["meta"]
            status = "trascritta"
            if make_pack:
                po = dict(pack_opts)
                po["title"] = Path(src["name"]).stem.replace("_", " ").strip() or "Lezione"
                po["key"] = pack_key_for(po, len(segments), meta.get("transcribed_at"))
                if not cache.load_studypack(po["key"]):
                    run_studypack(_JobView(job, index, total, src["name"], (0.8, 1.0)), src["hash"], segments, meta, po)
                status = "trascritta, pacchetto pronto"
            results.append({"name": src["name"], "status": status, "error": None})
        except Exception as exc:  # noqa: BLE001
            if job.cancel_event.is_set():
                raise
            results.append({"name": src["name"], "status": "errore", "error": f"{type(exc).__name__}: {exc}"})
    job.set_progress("done", total, total, "Coda completata.")
    return results


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
        notify("info", "Operazione annullata. I blocchi già completati restano in cache e verranno ripresi.")
    else:
        title, hint, detail = friendly_error(job.error)
        notify("error", title, hint, detail + "\n\n" + "\n".join(job.log[-5:]))


# ---------------------------------------------------------------------------
# barra laterale
# ---------------------------------------------------------------------------
apple_mac = is_apple_silicon_mac()
current_source = st.session_state.get("source") or {}
source_duration = float(current_source.get("duration") or 0.0)

with st.sidebar:
    st.header("Impostazioni")
    advanced = st.toggle("Modalità avanzata", value=False,
                         help="Mostra tutte le opzioni tecniche. In modalità semplice l'app sceglie da sola i valori adatti al tuo computer.")
    st.caption(hardware_summary())

    # --- trascrizione ---
    st.subheader("Trascrizione")
    default_device = "cuda" if (not apple_mac and cuda_available()) else "cpu"
    backend = "auto" if apple_mac else "faster-whisper"
    device, compute_type = default_device, ("float16" if default_device == "cuda" else "int8")
    chunk_minutes, vad_filter, allow_cpu_fallback = 8, True, False
    if advanced:
        backend_options = ["auto", "mlx-whisper", "faster-whisper"] if apple_mac else ["faster-whisper"]
        backend = st.selectbox(
            "Backend", backend_options, index=0,
            help=("Su Apple Silicon 'auto' usa mlx-whisper (GPU). Se MLX non è disponibile usa faster-whisper."
                  if apple_mac else "Su Windows e Linux si usa faster-whisper, su CPU o su GPU NVIDIA (Device = cuda)."),
        )
        with st.expander("Avanzate faster-whisper", expanded=not apple_mac):
            device_options = ["cpu"] if apple_mac else ["cpu", "cuda"]
            device = st.selectbox("Device", device_options, index=device_options.index(default_device),
                                  help="cuda richiede una GPU NVIDIA con driver aggiornati; su CPU scegli int8.")
            compute_type = st.selectbox("Compute type", ["int8", "float16", "float32"],
                                        index=1 if device == "cuda" else 0,
                                        help="int8 per la CPU; float16 per GPU NVIDIA (cuda).")
            allow_cpu_fallback = st.checkbox(
                "Se MLX fallisce, continua con faster-whisper su CPU", value=False,
                help="Disattivato: un errore MLX viene mostrato subito invece di partire con una trascrizione CPU molto più lenta.",
            )
    accel = accelerator_key(resolve_backend(backend), device)

    def model_label(name: str) -> str:
        if source_duration:
            est = format_duration_rough(estimate_seconds(source_duration, name, accel))
        else:
            est = format_duration_rough(estimate_seconds(3600, name, accel)) + " per ora di lezione"
        extra = "" if model_is_downloaded(name, backend) else " · da scaricare"
        return f"{name} · {MODEL_QUALITY.get(name, '')} · {est}{extra}"

    default_model = choose_default_model() if accel != "cuda" else "large-v3-turbo"
    model_size = st.selectbox(
        "Modello Whisper", MODEL_OPTIONS, index=MODEL_OPTIONS.index(default_model), format_func=model_label,
        help=f"Tempi stimati per questo computer ({accelerator_label(accel)})"
             + (" e per la registrazione scelta." if source_duration else "; diventano precisi dopo la prima trascrizione."),
    )
    lang_codes = list(LANGUAGES)
    language = st.selectbox("Lingua della lezione", lang_codes, index=lang_codes.index("it"), format_func=lambda c: LANGUAGES[c])
    if advanced:
        chunk_minutes = st.slider("Durata blocchi (minuti)", min_value=3, max_value=15, value=8,
                                  help="L'audio viene tagliato sulle pause in blocchi di questa durata: progresso, annullamento e ripresa lavorano per blocco.")
        vad_filter = st.checkbox("Rileva pause e salta i silenzi (VAD)", value=True)
    if apple_mac and backend in ("auto", "mlx-whisper") and not mlx_available():
        st.warning("mlx-whisper non è installato: verrà usato faster-whisper su CPU (molto più lento).")

    # --- pacchetto studio ---
    st.divider()
    st.subheader("Pacchetto studio")
    ollama_base_url = ollama.DEFAULT_URL
    if advanced:
        ollama_base_url = st.text_input("URL di Ollama", value=ollama.DEFAULT_URL)
    online, ollama_detailed = ollama_status(ollama_base_url)
    if st.button("Ricontrolla Ollama", key="refresh_ollama"):
        ollama_status.clear()
        online, ollama_detailed = ollama_status(ollama_base_url)
    ollama_models = [m["name"] for m in ollama_detailed]
    ram = ram_gb()
    fitting = ollama_models_that_fit(ollama_detailed, ram)
    if online and ollama_models:
        st.caption(f"Ollama attivo, {len(ollama_models)} modelli disponibili.")
    elif online:
        st.caption(f"Ollama è attivo ma non ha modelli. Dal terminale: `ollama pull {suggest_ollama_pull(ram)}`")
    else:
        st.caption("Ollama non trovato: il pacchetto studio userà le euristiche locali, più povere. "
                   "Installa Ollama da ollama.com per sintesi e appunti di qualità.")
    use_ollama = st.checkbox("Usa Ollama per sintesi, capitoli e glossario", value=bool(online and ollama_models),
                             disabled=not (online and ollama_models))
    pick_from = fitting or ollama_models
    default_ollama = ollama.pick_default_model(pick_from)
    ollama_model = st.selectbox(
        "Modello Ollama", ollama_models or ["nessuno"],
        index=(ollama_models.index(default_ollama) if default_ollama in ollama_models else 0),
        help=f"Per questo computer il modello consigliato da scaricare è {suggest_ollama_pull(ram)}.",
    )
    if ollama_models and ollama_model not in fitting and ram:
        size_gb = next((m["size"] for m in ollama_detailed if m["name"] == ollama_model), 0) / 2 ** 30
        st.warning(f"{ollama_model} occupa {size_gb:.0f} GB e il computer ha {ram:g} GB di memoria: "
                   f"potrebbe non partire o essere molto lento. Consigliato: {suggest_ollama_pull(ram)}.")
    want_notes = st.checkbox("Genera anche appunti completi", value=False, disabled=not use_ollama,
                             help="Per ogni blocco il modello scrive appunti discorsivi in Markdown: più lento ma molto più utile per studiare.")
    summary_mode, include_glossary, include_questions = "standard", True, True
    if advanced:
        summary_mode = st.selectbox("Livello sintesi", ["breve", "standard", "dettagliata"], index=1)
        include_glossary = st.checkbox("Includi glossario", value=True)
        include_questions = st.checkbox("Includi domande di ripasso", value=True)

    # --- cache ---
    if advanced:
        st.divider()
        st.subheader("Cache")
        st.caption(f"{human_size(cache_size_bytes())} in `{cache_root()}`")
        if st.checkbox("Mostra opzioni di pulizia", value=False):
            if st.button("Svuota tutta la cache", type="secondary"):
                clear_cache()
                for key in ("source", "transcript", "pack"):
                    st.session_state[key] = None
                st.session_state["queue"] = []
                st.rerun()
    st.caption(f"Versione {__version__}")

transcribe_opts = {
    "backend": backend, "model_size": model_size, "language": language, "device": device,
    "compute_type": compute_type, "vad_filter": vad_filter, "chunk_minutes": chunk_minutes,
    "allow_cpu_fallback": allow_cpu_fallback,
}
pack_opts_base = {
    "use_ollama": bool(use_ollama and online and ollama_models),
    "model": ollama_model if use_ollama else None,
    "base_url": ollama_base_url,
    "summary_mode": summary_mode,
    "want_notes": bool(want_notes and use_ollama),
    "include_glossary": include_glossary,
    "include_questions": include_questions,
}


# ---------------------------------------------------------------------------
# pagina
# ---------------------------------------------------------------------------
st.title("Sbobinatore")
st.caption("Trascrive lezioni audio e video sul tuo computer e prepara sintesi, capitoli, glossario e appunti. Nessun dato esce da qui.")

if not ffmpeg_works():
    _ffmpeg_hint = {
        "Darwin": "Su Mac: `brew install ffmpeg`.",
        "Windows": "Su Windows: `winget install Gyan.FFmpeg` in PowerShell, poi chiudi e riapri l'app.",
    }.get(platform.system(), "Su Linux: `sudo apt install ffmpeg`.")
    st.warning("ffmpeg non è installato: funziona tutto comunque, ma la lettura di audio e video è più lenta. " + _ffmpeg_hint)

notice = st.session_state.pop("notice", None)
if notice:
    kind, text, hint, detail = notice
    if kind == "error":
        st.error(f"**{text}.** {hint}")
        if detail.strip():
            with st.expander("Dettagli tecnici"):
                st.code(detail)
    else:
        getattr(st, kind)(text)

# --- 1. sorgente ---
st.subheader("1. Scegli la registrazione")
tab_path, tab_upload, tab_queue, tab_cache = st.tabs([
    "File sul computer (consigliato per file grandi)", "Carica file", "Più lezioni in coda", "Lezioni già elaborate",
])

with tab_path:
    if IS_WINDOWS:
        st.caption("Incolla il percorso del file. In Esplora file: Maiusc + clic destro sul file, poi \"Copia come percorso\".")
        _placeholder = r"C:\Users\tuonome\Downloads\lezione.mp4"
    else:
        st.caption("Incolla il percorso del file. Suggerimento: trascina il file nella finestra del Terminale per ottenerlo.")
        _placeholder = "/Users/tuonome/Downloads/lezione.mp4" if platform.system() == "Darwin" else "/home/tuonome/Scaricati/lezione.mp4"
    local_path = st.text_input("Percorso del file", placeholder=_placeholder)
    if st.button("Usa questo file", key="use_local"):
        candidate = clean_path(local_path)
        if not candidate or not os.path.isfile(candidate):
            st.error("Percorso non valido o file non trovato.")
        else:
            with st.spinner("Leggo il file…"):
                set_source(register_local_file(candidate))
            st.rerun()

with tab_upload:
    st.caption("File fino a 1 GB. Per registrazioni più grandi usa la scheda \"File sul computer\".")
    uploaded = st.file_uploader("Carica audio o video", type=SUPPORTED_TYPES, accept_multiple_files=False)
    if uploaded is not None:
        uid = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_uid") != uid:
            with st.spinner("Salvo il file…"):
                source = register_upload(uploaded)
            st.session_state["upload_uid"] = uid
            set_source(source)
            st.rerun()

with tab_queue:
    st.caption("Aggiungi più registrazioni e lasciale lavorare da sole, una dopo l'altra. "
               "Alla fine le trovi in \"Lezioni già elaborate\".")
    qcols = st.columns(2)
    with qcols[0]:
        paths_text = st.text_area("Percorsi dei file, uno per riga", height=120, key="queue_paths",
                                  placeholder="Puoi incollare anche il percorso di una cartella: prende tutti i file audio e video al suo interno.")
        if st.button("Aggiungi alla coda", key="queue_add_paths"):
            added, missing = 0, []
            for line in paths_text.splitlines():
                candidate = clean_path(line)
                if not candidate:
                    continue
                files = []
                if os.path.isdir(candidate):
                    files = sorted(
                        os.path.join(candidate, f) for f in os.listdir(candidate)
                        if f.lower().rsplit(".", 1)[-1] in SUPPORTED_TYPES and not f.startswith(".")
                    )
                elif os.path.isfile(candidate):
                    files = [candidate]
                else:
                    missing.append(line.strip())
                for path in files:
                    with st.spinner(f"Leggo {os.path.basename(path)}…"):
                        item = register_local_file(path)
                    if all(q["hash"] != item["hash"] for q in st.session_state["queue"]):
                        st.session_state["queue"].append(item)
                        added += 1
            if missing:
                st.error("Non trovati: " + ", ".join(missing))
            if added:
                st.rerun()
    with qcols[1]:
        multi = st.file_uploader("Oppure carica più file", type=SUPPORTED_TYPES, accept_multiple_files=True, key="queue_upload")
        if multi:
            uid = "|".join(getattr(u, "file_id", None) or f"{u.name}:{u.size}" for u in multi)
            if st.session_state.get("queue_upload_uid") != uid:
                for up in multi:
                    with st.spinner(f"Salvo {up.name}…"):
                        item = register_upload(up)
                    if all(q["hash"] != item["hash"] for q in st.session_state["queue"]):
                        st.session_state["queue"].append(item)
                st.session_state["queue_upload_uid"] = uid
                st.rerun()

    queue = st.session_state["queue"]
    queue_job = get_job(st.session_state.get("queue_job_id"))
    if queue_job is None:
        running_queue = find_running("queue", "all")
        if running_queue is not None:
            st.session_state["queue_job_id"] = running_queue.id
            queue_job = running_queue

    def _on_queue_done(results):
        errors = [r for r in results if r["status"] == "errore"]
        done = len(results) - len(errors)
        if errors:
            notify("error", f"Coda terminata: {done} lezioni pronte, {len(errors)} con errore",
                   "Apri la lezione dalla scheda \"Lezioni già elaborate\" e riprova la trascrizione da sola.",
                   "\n".join(f"{r['name']}: {r['error']}" for r in errors))
        else:
            notify("success", f"Coda terminata: {done} lezioni pronte. Le trovi nella scheda \"Lezioni già elaborate\".")
        st.session_state["queue"] = []

    consume_job("queue_job_id", _on_queue_done)
    queue_job = get_job(st.session_state.get("queue_job_id"))

    if queue:
        total_minutes = sum(q.get("duration") or 0 for q in queue) / 60
        est_total = format_duration_rough(estimate_seconds(total_minutes * 60, model_size, accel)) if total_minutes else "n/d"
        st.dataframe(
            [{"lezione": q["name"], "durata": format_ts(q.get("duration") or 0), "dimensione": human_size(q["size"]),
              "stato": "già trascritta" if LessonCache(q["hash"]).has_transcript() else "da trascrivere"} for q in queue],
            hide_index=True, **_WIDE,
        )
        st.caption(f"{len(queue)} lezioni, {format_ts(total_minutes * 60)} di audio in totale. "
                   f"Tempo stimato per la trascrizione con {model_size}: {est_total}"
                   + (", più il pacchetto studio con Ollama." if pack_opts_base["use_ollama"] else "."))
        bcols = st.columns([1, 2, 1])
        make_pack = bcols[1].checkbox("Genera anche il pacchetto studio di ogni lezione", value=True, key="queue_make_pack")
        if bcols[0].button("Avvia la coda", type="primary", disabled=queue_job is not None, **_WIDE):
            job = start_job("queue", "all", lambda j, items=[dict(q) for q in queue], o=dict(transcribe_opts),
                            po=dict(pack_opts_base), mp=make_pack: run_queue(j, items, o, po, mp))
            st.session_state["queue_job_id"] = job.id
            st.rerun()
        if bcols[2].button("Svuota la coda", disabled=queue_job is not None, **_WIDE):
            st.session_state["queue"] = []
            st.rerun()
    if queue_job is not None:
        render_job_monitor("queue_job_id", "Coda")
    elif not queue:
        st.caption("La coda è vuota.")

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
            set_source({"hash": item["hash"], "name": item["name"], "size": item.get("size") or 0,
                        "path": info.get("path"), "duration": file_duration(info.get("path"))})
            st.rerun()
        if cols[1].button("Elimina", key="delete_cached"):
            LessonCache(lessons[choice]["hash"]).clear()
            if st.session_state.get("source") and st.session_state["source"]["hash"] == lessons[choice]["hash"]:
                for key in ("source", "transcript", "pack"):
                    st.session_state[key] = None
            st.rerun()

source = st.session_state.get("source")
if not source:
    others = [j for j in running_jobs() if j.kind != "queue"]
    if others:
        st.info(f"{len(others)} elaborazioni in corso in background. Riapri la lezione dalla scheda \"Lezioni già elaborate\" per seguirle.")
    st.stop()

cache = LessonCache(source["hash"])
if not source.get("duration") and source.get("path"):
    source["duration"] = file_duration(source["path"])
_dur = f" · {format_ts(source['duration'])}" if source.get("duration") else ""
st.info(f"**{source['name']}**{_dur} · {human_size(source['size'])}")

# --- 2. trascrizione ---
st.subheader("2. Trascrizione")


def _on_transcript_done(result):
    st.session_state["transcript"] = result
    st.session_state["pack"] = None
    st.session_state["just_transcribed"] = True


consume_job("transcribe_job_id", _on_transcript_done)
transcribe_job = get_job(st.session_state.get("transcribe_job_id"))
transcript = st.session_state.get("transcript")

if transcribe_job is None and not transcript:
    est = format_duration_rough(estimate_seconds(source.get("duration") or 3600, model_size, accel))
    pieces = [f"Modello **{model_size}** su {accelerator_label(accel)}: tempo stimato {est}" if source.get("duration")
              else f"Modello **{model_size}** su {accelerator_label(accel)}."]
    if not model_is_downloaded(model_size, backend):
        pieces.append(f"La prima volta il modello viene scaricato (circa {MODEL_DOWNLOAD_GB.get(model_size, 1.6):g} GB): "
                      "possono servire alcuni minuti in più e la barra parte solo dopo il download.")
    if accel == "cpu" and model_size in ("large-v3", "large-v3-turbo", "medium") and (source.get("duration") or 0) > 1800:
        pieces.append("Su sola CPU è lento: se hai fretta scegli `small` nella barra laterale, poi ritrascrivi con calma.")
    st.info(" ".join(pieces))

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
    rtf_factor = meta.get("realtime_factor")
    if st.session_state.pop("just_transcribed", False):
        st.success(f"Trascrizione completata in {format_ts(meta.get('elapsed_seconds') or 0)}"
                   + (f", {rtf_factor:g} volte più veloce del tempo reale" if rtf_factor else "")
                   + ". Ora puoi generare il pacchetto studio qui sotto, oppure scaricare subito il testo nel passo 5.")
    if meta.get("fallback_reason"):
        st.warning(f"MLX ha fallito e si è usato faster-whisper su CPU: {meta['fallback_reason']}")
    with st.expander("Dettagli della trascrizione", expanded=False):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Durata", format_ts(meta.get("duration") or 0))
        m2.metric("Segmenti", len(transcript["segments"]))
        m3.metric("Lingua", meta.get("language") or "n/d")
        m4.metric("Motore", (meta.get("backend") or "n/d").replace("-whisper", "") + f" · {meta.get('model_size') or meta.get('model', '')}")
        m5.metric("Tempo", format_ts(meta.get("elapsed_seconds") or 0) + (f" ({rtf_factor:g}×)" if rtf_factor else ""))
else:
    cols[1].caption("Il file viene convertito, diviso in blocchi sulle pause e trascritto blocco per blocco. Puoi annullare e riprendere quando vuoi.")

if not transcript:
    st.stop()

segments = transcript["segments"]
meta = transcript["meta"]
paragraphs = segments_to_paragraphs(segments)

# --- 3. pacchetto studio ---
st.subheader("3. Pacchetto studio")
lesson_title = st.text_input("Titolo lezione", key="lesson_title")
pack_opts = dict(pack_opts_base)
pack_opts["title"] = lesson_title or "Lezione"
pack_opts["key"] = pack_key_for(pack_opts, len(segments), meta.get("transcribed_at"))


def _on_pack_done(result):
    st.session_state["pack"] = result


consume_job("pack_job_id", _on_pack_done)
pack_job = get_job(st.session_state.get("pack_job_id"))
cached_pack = cache.load_studypack(pack_opts["key"])

cols = st.columns([1, 1, 3])
gen_label = f"Genera con {ollama_model}" if pack_opts["use_ollama"] else "Genera (senza Ollama)"
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
elif not pack_opts["use_ollama"]:
    cols[2].caption("Senza Ollama il pacchetto è essenziale (frasi estratte dal testo). Con Ollama ottieni sintesi e appunti veri.")

pack = st.session_state.get("pack")

if pack and pack.get("chapters") and looks_generic_title(lesson_title):
    suggestion = (pack["chapters"][0].get("title") or "").strip()
    if suggestion and suggestion.lower() != lesson_title.lower():
        scols = st.columns([3, 1])
        scols[0].caption(f"Il titolo \"{lesson_title}\" non dice molto. Proposta dal contenuto: **{suggestion}**")
        if scols[1].button("Usa questo titolo", **_WIDE):
            st.session_state["lesson_title"] = suggestion
            pack["title"] = suggestion
            st.rerun()

# --- 4. risultati ---
st.subheader("4. Risultati")
tabs = st.tabs(["Pacchetto studio", "Appunti AI", "Testo completo", "Correggi la trascrizione", "Dettagli tecnici"])

with tabs[0]:
    if not pack:
        st.caption("Genera il pacchetto studio per vedere sintesi, concetti, capitoli, glossario e domande.")
    else:
        for warning in pack.get("warnings") or []:
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
        st.caption("Attiva \"Genera anche appunti completi\" nella barra laterale e rigenera il pacchetto con Ollama.")

with tabs[2]:
    fcols = st.columns([3, 1])
    query = fcols[0].text_input("Cerca nel testo", placeholder="una parola o una frase", key="search_text")
    show_ts = fcols[1].checkbox("Mostra i minuti", value=True, key="show_ts_full")
    terms = [c["term"] for c in (pack or {}).get("key_concepts", [])] + [g["term"] for g in (pack or {}).get("glossary", [])]
    shown = paragraphs
    if query.strip():
        shown = [p for p in paragraphs if query.strip().lower() in p["text"].lower()]
        st.caption(f"{len(shown)} paragrafi su {len(paragraphs)} contengono \"{query.strip()}\".")
        terms = [query.strip()]
    for para in shown:
        text = highlight_terms_in_markdown(para["text"], terms) if terms else para["text"]
        stamp = f"**[{format_ts(para['start'])}]** " if show_ts else ""
        st.markdown(stamp + text)

with tabs[3]:
    st.caption("Whisper sbaglia spesso nomi e termini tecnici. Correggili qui: la correzione vale per gli export e per il pacchetto studio, "
               "che andrà rigenerato.")

    def _save_edited(new_segments, message):
        new_meta = dict(meta)
        new_meta["transcribed_at"] = time.time()
        new_meta["edited_at"] = new_meta["transcribed_at"]
        cache.save_transcript(new_segments, new_meta)
        st.session_state["transcript"] = {"segments": new_segments, "meta": new_meta}
        st.session_state["pack"] = None
        notify("success", message)
        st.rerun()

    rcols = st.columns([2, 2, 1])
    find_text = rcols[0].text_input("Trova", key="find_text", placeholder="es. Cheynes")
    replace_text = rcols[1].text_input("Sostituisci con", key="replace_text", placeholder="es. Keynes")
    rcols[2].write("")
    if rcols[2].button("Sostituisci ovunque", **_WIDE) and find_text:
        count = sum(s["text"].count(find_text) for s in segments)
        if count:
            _save_edited([dict(s, text=s["text"].replace(find_text, replace_text)) for s in segments],
                         f"{count} occorrenze di \"{find_text}\" sostituite con \"{replace_text}\".")
        else:
            st.warning(f"\"{find_text}\" non compare nella trascrizione.")
    table = pd.DataFrame([{"inizio": format_ts(s["start"]), "fine": format_ts(s["end"]), "testo": s["text"]} for s in segments])
    edited = st.data_editor(table, hide_index=True, disabled=["inizio", "fine"], num_rows="fixed",
                            key=f"seg_editor_{meta.get('transcribed_at')}", **_WIDE)
    if st.button("Salva le correzioni", type="primary"):
        new_texts = edited["testo"].tolist()
        changed = sum(1 for s, t in zip(segments, new_texts) if s["text"] != t)
        if changed:
            _save_edited([dict(s, text=str(t)) for s, t in zip(segments, new_texts)], f"{changed} segmenti corretti e salvati.")
        else:
            st.info("Nessuna modifica da salvare.")

with tabs[4]:
    st.json({"transcript": meta, "study_pack": {k: v for k, v in (pack or {}).items() if k in ("generator", "model", "ollama_stats", "warnings", "created_at")}})

# --- 5. export ---
st.subheader("5. Scarica")
base = slugify(lesson_title or source["name"])
export_pack = pack or {}
with_ts = st.checkbox("Minuti nei documenti", value=True, key="export_ts")
if not pack:
    st.caption("Senza pacchetto studio i documenti contengono solo la trascrizione.")
md_doc = build_markdown(pack_opts["title"], paragraphs, export_pack, with_timestamps=with_ts)
cols = st.columns(3)
cols[0].download_button("Word (.docx)", data=build_docx(pack_opts["title"], paragraphs, export_pack, with_timestamps=with_ts),
                        file_name=f"{base}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", **_WIDE)
cols[1].download_button("Markdown (.md)", data=md_doc.encode("utf-8"), file_name=f"{base}.md", mime="text/markdown", **_WIDE)
if pack and pack.get("notes_md"):
    cols[2].download_button("Appunti AI (.md)", data=pack["notes_md"].encode("utf-8"), file_name=f"{base}_appunti.md", mime="text/markdown", **_WIDE)
else:
    cols[2].download_button("Testo semplice (.txt)", data=build_txt(paragraphs, with_ts).encode("utf-8"), file_name=f"{base}.txt", mime="text/plain", **_WIDE)
with st.expander("Altri formati"):
    cols = st.columns(4)
    cols[0].download_button("RTF (Pages, LibreOffice)", data=build_rtf(pack_opts["title"], paragraphs, export_pack, with_timestamps=with_ts),
                            file_name=f"{base}.rtf", mime="application/rtf", **_WIDE)
    cols[1].download_button("Testo (.txt)", data=build_txt(paragraphs, with_ts).encode("utf-8"), file_name=f"{base}_trascrizione.txt", mime="text/plain", **_WIDE)
    cols[2].download_button("Sottotitoli (.srt)", data=build_srt(segments).encode("utf-8"), file_name=f"{base}.srt", mime="text/plain", **_WIDE)
    cols[3].download_button("Dati completi (.json)", data=build_json(segments, meta, pack).encode("utf-8"), file_name=f"{base}.json", mime="application/json", **_WIDE)
