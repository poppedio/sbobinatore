# Sbobinatore

[![Licenza MIT](https://img.shields.io/badge/licenza-MIT-green.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Piattaforma](https://img.shields.io/badge/piattaforma-macOS%20Apple%20Silicon%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

App Streamlit che trascrive lezioni audio/video **sul tuo Mac** con Whisper e produce un pacchetto studio: sintesi, concetti chiave, capitoli con minutaggio, glossario, domande di ripasso e, se vuoi, appunti completi in Markdown generati con Ollama.

Nessun dato esce dal computer.

## Come funziona

1. **Sorgente**: indichi il percorso di un file sul Mac (consigliato per file grandi, zero upload) oppure lo carichi dal browser.
2. **Trascrizione**: il file viene convertito in WAV 16 kHz mono (ffmpeg, o PyAV se ffmpeg manca), diviso in blocchi tagliati sulle pause (VAD Silero) e trascritto blocco per blocco con `mlx-whisper` (GPU Apple Silicon) o `faster-whisper`. Vedi il progresso con stima del tempo, puoi annullare, e se riapri la stessa lezione riparte dai blocchi già fatti.
3. **Pacchetto studio**: con Ollama, ogni blocco di circa 10 minuti viene analizzato da solo (titolo, punti, concetti, termini, appunti) e poi una chiamata finale unisce tutto in sintesi, capitoli con minuti, glossario e domande. La finestra di contesto viene dimensionata sul prompt, il thinking dei modelli Qwen3 è disattivato, l'output è vincolato da uno schema JSON. Senza Ollama si usano euristiche locali.
4. **Export**: RTF (Word/Pages/LibreOffice), Markdown, appunti AI in Markdown, TXT con minuti, sottotitoli SRT, JSON completo.

Tutto viene salvato in cache per impronta del file (`~/.cache/transcriptor`, cambiabile con `TRANSCRIPTOR_CACHE_DIR`): rigenerare il pacchetto con un altro modello o livello di sintesi non ritrascrive nulla.

## Avvio rapido

Serve Python 3.9 o successivo.

```bash
git clone https://github.com/poppedio/sbobinatore.git
cd sbobinatore
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Alla prima trascrizione il modello Whisper viene scaricato da Hugging Face (circa 1,6 GB per `large-v3-turbo`).

### Ollama (opzionale ma consigliato)

```bash
brew install ollama        # oppure l'app da ollama.com
ollama pull qwen3:30b      # principale, MoE veloce su Apple Silicon
ollama pull gemma3:12b     # alternativa più leggera
```

L'app rileva Ollama da sola e preseleziona il modello migliore disponibile.

## Impostazioni utili

| Impostazione | Consiglio su Mac Apple Silicon |
| --- | --- |
| Backend | `auto` (usa MLX, segnala l'errore invece di passare in silenzio alla CPU) |
| Modello Whisper | `large-v3-turbo` |
| Lingua | `it` (evita il rilevamento automatico su ogni blocco) |
| Durata blocchi | 8 minuti |
| Appunti completi | attivala quando vuoi appunti discorsivi; raddoppia il tempo Ollama |

Tempi indicativi su M-series con `large-v3-turbo`: una lezione di 90 minuti si trascrive in 3-6 minuti. Il pacchetto studio con `qwen3:30b` richiede qualche minuto in più, proporzionale al numero di blocchi.

## Struttura del progetto

```
app.py                     interfaccia Streamlit (sorgente, job in background, risultati, export)
transcriptor/audio.py      conversione ffmpeg/PyAV, hash, VAD e piano dei blocchi
transcriptor/transcribe.py backend mlx-whisper / faster-whisper, progresso, ripresa, filtro allucinazioni
transcriptor/cache.py      cache su disco per impronta del file
transcriptor/textproc.py   pulizia testo, paragrafi con minuti, finestre per il modello, euristiche
transcriptor/ollama.py     client Ollama: schema JSON, num_ctx dinamico, think off
transcriptor/studypack.py  map/reduce per il pacchetto studio e gli appunti; fallback euristico
transcriptor/export.py     Markdown, RTF, TXT, SRT, JSON
transcriptor/jobs.py       job in thread con progresso e annullamento
tests/                     test unitari e smoke test headless dell'app
```

Test:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Dipendenze di sistema

- `ffmpeg` consigliato (`brew install ffmpeg`). Se manca o non parte, la conversione usa PyAV, già incluso con faster-whisper.
- Su Linux: `sudo apt install ffmpeg`. Su Windows: installa ffmpeg e aggiungilo al PATH.

## Note

- Se chiudi la pagina durante una trascrizione il lavoro continua in background: riapri la lezione dalla scheda "Lezioni già elaborate".
- I blocchi già trascritti restano in cache anche se annulli: la volta dopo si riparte da lì.
- I segmenti con frasi tipiche delle allucinazioni di Whisper ("Sottotitoli a cura di…", ripetizioni) vengono scartati.

## Contribuire

Segnalazioni e pull request sono benvenute. Prima di aprire una PR esegui `pytest -q` e descrivi il caso d'uso (tipo di lezione, modello Whisper e Ollama usati).

## Licenza

Distribuito con licenza [MIT](LICENSE).
