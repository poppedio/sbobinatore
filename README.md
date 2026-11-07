# Sbobinatore

[![Licenza MIT](https://img.shields.io/badge/licenza-MIT-green.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Piattaforma](https://img.shields.io/badge/piattaforma-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)

Hai la registrazione audio o video di una lezione? Sbobinatore la trasforma in testo scritto, con i minuti a fianco, e se vuoi ti prepara anche sintesi, capitoli, glossario, domande di ripasso e appunti completi.

Funziona **interamente sul tuo computer**: nessuna registrazione viene caricata online, non servono account né abbonamenti.

## Se non sei esperto: parti da qui

Servono circa 15 minuti la prima volta, quasi tutti di attesa per i download. Le volte successive basta un doppio clic.

**1. Scarica il programma.** In alto in questa pagina premi il pulsante verde **Code** e poi **Download ZIP**. Estrai il file scaricato: su Mac con un doppio clic, su Windows con tasto destro e *Estrai tutto*. Sposta la cartella `sbobinatore` dove preferisci, per esempio in Documenti.

**2. Installa Python** (il linguaggio con cui è scritto il programma), se non lo hai già.
- **Mac**: scarica l'installer da [python.org/downloads/macos](https://www.python.org/downloads/macos/) e installalo come una normale app.
- **Windows**: scarica l'installer da [python.org/downloads/windows](https://www.python.org/downloads/windows/). Nella prima schermata dell'installazione **spunta la casella "Add python.exe to PATH"**, poi premi *Install Now*.

**3. Avvia con un doppio clic.**
- **Mac**: doppio clic su `Avvia.command`. Se macOS dice che "non può essere aperto", fai clic con il tasto destro sul file, scegli *Apri* e conferma. Se invece si apre come testo, apri il Terminale, scrivi `chmod +x ` (con lo spazio), trascina il file `Avvia.command` nella finestra e premi Invio, poi riprova il doppio clic.
- **Windows**: doppio clic su `Avvia.bat`. Se compare l'avviso di Windows SmartScreen, premi *Ulteriori informazioni* e poi *Esegui comunque*. Se il firewall chiede un permesso, consenti l'accesso alle reti private.

Si apre una finestra nera con del testo: è normale. La prima volta scarica e installa i componenti (qualche minuto, qualche centinaio di MB). Lo script ti chiede anche se installare `ffmpeg`, un programma che rende più veloce la lettura di audio e video: rispondi S se puoi. Alla fine si apre il browser con Sbobinatore. **Lascia aperta la finestra nera finché usi il programma**: per chiudere tutto basta chiuderla.

**4. Trascrivi la prima lezione.** Nella pagina scegli la scheda *File sul computer* e incolla il percorso della registrazione, oppure usa *Carica file* e trascina il file nel riquadro. Premi **Trascrivi**. La prima volta viene scaricato il modello di riconoscimento vocale (circa 1,6 GB): l'app te lo avvisa e possono passare alcuni minuti prima che la barra di avanzamento inizi a muoversi. Poi la trascrizione va da sola e puoi seguire il progresso.

Quando è finita trovi il testo con i minuti, un pacchetto di studio di base e i pulsanti per scaricare tutto in Word (RTF), Markdown, testo, sottotitoli o JSON.

### Vuoi sintesi e appunti fatti bene? Installa Ollama

Senza Ollama il pacchetto studio è essenziale. Con Ollama, un programma gratuito che fa girare un'intelligenza artificiale sul tuo computer, ottieni sintesi, capitoli con minuti, glossario, domande e appunti discorsivi.

1. Scarica Ollama da [ollama.com](https://ollama.com) e installalo (Mac e Windows). Resta attivo in background, lo vedi nella barra dei menu o accanto all'orologio.
2. Scarica un modello. Apri l'app di Ollama, scegli un modello dal menu e manda un messaggio qualsiasi: il modello viene scaricato. Oppure dal Terminale (Mac) o PowerShell (Windows):

   | Il tuo computer ha… | Modello consigliato | Comando |
   | --- | --- | --- |
   | 8 GB di memoria | qwen3:8b (circa 5 GB) | `ollama pull qwen3:8b` |
   | 16 GB di memoria | gemma3:12b (circa 8 GB) | `ollama pull gemma3:12b` |
   | 32 GB o più (Mac) o scheda video da 24 GB | qwen3:30b (circa 19 GB) | `ollama pull qwen3:30b` |

3. Riapri Sbobinatore: riconosce Ollama da solo e sceglie il modello migliore che hai. Sotto la trascrizione premi il pulsante **Genera con ...** seguito dal nome del modello.

### Problemi comuni

- **Il doppio clic non fa niente o si apre un editor di testo** (Mac): vedi il punto 3, serve il comando `chmod +x`.
- **"Python non è installato" ma l'ho installato** (Windows): quasi sempre manca la spunta *Add python.exe to PATH*. Reinstalla Python spuntandola.
- **Sembra bloccato**: il primo avvio e la prima trascrizione scaricano file grandi senza mostrare una barra. Aspetta qualche minuto e guarda la finestra nera: lì compaiono i messaggi.
- **È lento**: su computer senza scheda video NVIDIA e senza chip Apple Silicon la trascrizione può durare quanto la lezione o più. Nella barra laterale scegli il modello Whisper `small` o `medium`.
- **Il browser non si apre**: apri tu il browser e vai all'indirizzo che la finestra nera mostra dopo "Local URL" (di solito `http://localhost:8501`).
- **Ho chiuso la pagina per sbaglio**: niente paura, la trascrizione continua. Riapri l'indirizzo e trova la lezione nella scheda *Lezioni già elaborate*.
- **Voglio ripartire da zero**: chiudi la finestra nera, cancella la cartella `.venv` dentro `sbobinatore` e riavvia.

---

## Come funziona

1. **Sorgente**: indichi il percorso di un file sul computer (consigliato per file grandi, zero upload) oppure lo carichi dal browser.
2. **Trascrizione**: il file viene convertito in WAV 16 kHz mono (ffmpeg, o PyAV se ffmpeg manca), diviso in blocchi tagliati sulle pause (VAD Silero) e trascritto blocco per blocco con `mlx-whisper` (GPU Apple Silicon) o `faster-whisper` (Windows, Linux e Mac Intel: CPU, oppure GPU NVIDIA con CUDA). Vedi il progresso con stima del tempo, puoi annullare, e se riapri la stessa lezione riparte dai blocchi già fatti.
3. **Pacchetto studio**: con Ollama, ogni blocco di circa 10 minuti viene analizzato da solo (titolo, punti, concetti, termini, appunti) e poi una chiamata finale unisce tutto in sintesi, capitoli con minuti, glossario e domande. La finestra di contesto viene dimensionata sul prompt, il thinking dei modelli Qwen3 è disattivato, l'output è vincolato da uno schema JSON. Senza Ollama si usano euristiche locali.
4. **Export**: RTF (Word/Pages/LibreOffice), Markdown, appunti AI in Markdown, TXT con minuti, sottotitoli SRT, JSON completo.

Tutto viene salvato in cache per impronta del file (`~/.cache/transcriptor` su Mac e Linux, `C:\Users\<nome>\.cache\transcriptor` su Windows; cambiabile con la variabile d'ambiente `TRANSCRIPTOR_CACHE_DIR`): rigenerare il pacchetto con un altro modello o livello di sintesi non ritrascrive nulla.

## Per chi vuole il controllo: installazione manuale

Gli script `Avvia.command` e `Avvia.bat` fanno esattamente questo: controllano Python 3.9+ e ffmpeg, creano `.venv`, installano `requirements.txt` quando cambia e lanciano Streamlit. Se preferisci farlo a mano:

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

Su Linux puoi anche usare `bash Avvia.command`.

### Windows (PowerShell)

```powershell
winget install Python.Python.3.12 Gyan.FFmpeg Git.Git   # poi chiudi e riapri PowerShell
git clone https://github.com/poppedio/sbobinatore.git
cd sbobinatore
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Se l'attivazione del venv viene bloccata ("l'esecuzione di script è disabilitata"), una volta sola: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Note per Windows e Linux:

- `mlx-whisper` è solo per Apple Silicon e viene saltato automaticamente: la trascrizione usa `faster-whisper`.
- **GPU NVIDIA**: nella barra laterale apri *Avanzate faster-whisper* e imposta Device = `cuda`, Compute type = `float16`. Servono driver NVIDIA aggiornati; se compare un errore su cuBLAS o cuDNN installa `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` oppure torna a `cpu`.
- **Solo CPU**: lascia `cpu` e `int8` e parti con `small` o `medium`; passa a `large-v3-turbo` se la qualità non basta.
- **Percorso del file su Windows**: in Esplora file, Maiusc + clic destro sul file e *Copia come percorso*; le virgolette vengono tolte dall'app.

### Ollama

- Mac: `brew install ollama` oppure l'app da [ollama.com](https://ollama.com).
- Windows: installer da [ollama.com/download/windows](https://ollama.com/download/windows) oppure `winget install Ollama.Ollama`.
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`.

Poi `ollama pull <modello>` (vedi la tabella sopra). L'app rileva Ollama da sola e preseleziona il modello migliore disponibile.

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
Avvia.command / Avvia.bat  avvio con doppio clic (Mac e Linux / Windows): venv, dipendenze, Streamlit
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
