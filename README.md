# Sbobinatore

[![Licenza MIT](https://img.shields.io/badge/licenza-MIT-green.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Piattaforma](https://img.shields.io/badge/piattaforma-macOS%20Apple%20Silicon%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

App Streamlit che trascrive lezioni audio/video **sul tuo computer** (Mac, Windows o Linux) con Whisper e produce un pacchetto studio: sintesi, concetti chiave, capitoli con minutaggio, glossario, domande di ripasso e, se vuoi, appunti completi in Markdown generati con Ollama.

Nessun dato esce dal computer.

## Come funziona

1. **Sorgente**: indichi il percorso di un file sul computer (consigliato per file grandi, zero upload) oppure lo carichi dal browser.
2. **Trascrizione**: il file viene convertito in WAV 16 kHz mono (ffmpeg, o PyAV se ffmpeg manca), diviso in blocchi tagliati sulle pause (VAD Silero) e trascritto blocco per blocco con `mlx-whisper` (GPU Apple Silicon) o `faster-whisper` (Windows, Linux e Mac Intel: CPU, oppure GPU NVIDIA con CUDA). Vedi il progresso con stima del tempo, puoi annullare, e se riapri la stessa lezione riparte dai blocchi già fatti.
3. **Pacchetto studio**: con Ollama, ogni blocco di circa 10 minuti viene analizzato da solo (titolo, punti, concetti, termini, appunti) e poi una chiamata finale unisce tutto in sintesi, capitoli con minuti, glossario e domande. La finestra di contesto viene dimensionata sul prompt, il thinking dei modelli Qwen3 è disattivato, l'output è vincolato da uno schema JSON. Senza Ollama si usano euristiche locali.
4. **Export**: RTF (Word/Pages/LibreOffice), Markdown, appunti AI in Markdown, TXT con minuti, sottotitoli SRT, JSON completo.

Tutto viene salvato in cache per impronta del file (`~/.cache/transcriptor` su Mac e Linux, `C:\Users\<nome>\.cache\transcriptor` su Windows; cambiabile con la variabile d'ambiente `TRANSCRIPTOR_CACHE_DIR`): rigenerare il pacchetto con un altro modello o livello di sintesi non ritrascrive nulla.

## Avvio rapido

Serve Python 3.9 o successivo. Alla prima trascrizione il modello Whisper viene scaricato da Hugging Face (circa 1,6 GB per `large-v3-turbo`).

### macOS e Linux

```bash
git clone https://github.com/poppedio/sbobinatore.git
cd sbobinatore
brew install ffmpeg            # Linux: sudo apt install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Windows

1. Installa **Python** da [python.org](https://www.python.org/downloads/windows/) spuntando *Add python.exe to PATH*, oppure da PowerShell: `winget install Python.Python.3.12`.
2. Installa **ffmpeg**: `winget install Gyan.FFmpeg`. Chiudi e riapri PowerShell, poi controlla con `ffmpeg -version`.
3. Installa **Git** se manca: `winget install Git.Git` (oppure scarica lo ZIP del progetto dal pulsante *Code* su GitHub).
4. In PowerShell:

```powershell
git clone https://github.com/poppedio/sbobinatore.git
cd sbobinatore
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Se l'attivazione del venv viene bloccata ("l'esecuzione di script è disabilitata"), una volta sola:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Note per Windows:

- `mlx-whisper` è solo per Apple Silicon e viene saltato automaticamente: su Windows la trascrizione usa `faster-whisper`.
- **GPU NVIDIA**: nella barra laterale apri *Avanzate faster-whisper* e imposta Device = `cuda`, Compute type = `float16`. Servono driver NVIDIA aggiornati; se compare un errore su cuBLAS o cuDNN installa `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` oppure torna a `cpu`.
- **Solo CPU**: lascia `cpu` e `int8`. I tempi sono molto più lunghi che su GPU: per provare parti con `small` o `medium`, poi passa a `large-v3-turbo` se la qualità non basta.
- **Percorso del file**: nella scheda *File sul computer* incolla un percorso come `C:\Users\nome\Downloads\lezione.mp4`. In Esplora file, Maiusc + clic destro sul file e *Copia come percorso*; le virgolette vengono tolte dall'app.
- Se Windows Defender o il firewall chiedono il permesso al primo avvio di Streamlit, basta consentire l'accesso alle reti private: l'app è visibile solo da `localhost`.

### Ollama (opzionale ma consigliato)

Ollama genera il pacchetto studio (sintesi, capitoli, glossario, domande, appunti). Senza Ollama l'app usa euristiche locali più povere.

- **Mac**: `brew install ollama` oppure l'app da [ollama.com](https://ollama.com).
- **Windows**: installer da [ollama.com/download/windows](https://ollama.com/download/windows) oppure `winget install Ollama.Ollama`. Dopo l'installazione Ollama resta attivo nella barra delle applicazioni.
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`.

Poi scarica un modello in base alla memoria disponibile:

```bash
ollama pull qwen3:30b      # consigliato con 32 GB di RAM (Apple Silicon) o GPU da 24 GB
ollama pull gemma3:12b     # buon compromesso con 16 GB di RAM o GPU da 12 GB
ollama pull qwen3:8b       # leggero, per 8-16 GB
```

L'app rileva Ollama da sola e preseleziona il modello migliore disponibile.

## Impostazioni utili

| Impostazione | Mac Apple Silicon | Windows / Linux con GPU NVIDIA | Windows / Linux solo CPU |
| --- | --- | --- | --- |
| Backend | `auto` (usa MLX, segnala l'errore invece di passare in silenzio alla CPU) | `faster-whisper` | `faster-whisper` |
| Device / Compute type | — | `cuda` / `float16` | `cpu` / `int8` |
| Modello Whisper | `large-v3-turbo` | `large-v3-turbo` | `medium` (o `small` per provare) |
| Lingua | `it` (evita il rilevamento automatico su ogni blocco) | `it` | `it` |
| Durata blocchi | 8 minuti | 8 minuti | 8 minuti |
| Appunti completi | attivala quando vuoi appunti discorsivi; raddoppia il tempo Ollama | idem | idem |

Tempi indicativi su Mac M-series con `large-v3-turbo`: una lezione di 90 minuti si trascrive in 3-6 minuti. Su GPU NVIDIA recenti i tempi sono simili; su sola CPU aspettati tempi dell'ordine della durata della lezione o più, a seconda del modello. Il pacchetto studio con `qwen3:30b` richiede qualche minuto in più, proporzionale al numero di blocchi.

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

- `ffmpeg` consigliato. Se manca o non parte, la conversione usa PyAV, già incluso con faster-whisper (più lento).
  - Mac: `brew install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` (o `choco install ffmpeg`), poi riapri il terminale. Se preferisci lo ZIP da [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), aggiungi la cartella `bin` al PATH.
  - Linux: `sudo apt install ffmpeg`
- Verifica con `ffmpeg -version` dal terminale che userai per lanciare l'app.

## Note

- Se chiudi la pagina durante una trascrizione il lavoro continua in background: riapri la lezione dalla scheda "Lezioni già elaborate".
- I blocchi già trascritti restano in cache anche se annulli: la volta dopo si riparte da lì.
- I segmenti con frasi tipiche delle allucinazioni di Whisper ("Sottotitoli a cura di…", ripetizioni) vengono scartati.

## Contribuire

Segnalazioni e pull request sono benvenute. Prima di aprire una PR esegui `pytest -q` e descrivi il caso d'uso (tipo di lezione, modello Whisper e Ollama usati).

## Licenza

Distribuito con licenza [MIT](LICENSE).
