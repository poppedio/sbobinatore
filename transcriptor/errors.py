"""Traduce gli errori tecnici dei job in messaggi comprensibili con un rimedio."""
from __future__ import annotations

import re
from typing import Optional, Tuple

_RULES = [
    (r"(getaddrinfo|Name or service not known|nodename nor servname|Max retries|ConnectionError|HTTPSConnectionPool|huggingface\.co|Temporary failure in name resolution|NewConnectionError)",
     "Connessione a internet assente durante il download del modello",
     "Il modello di riconoscimento vocale viene scaricato la prima volta. Controlla la connessione e premi di nuovo Trascrivi: riparte da dove si era fermato."),
    (r"(out of memory|OutOfMemory|MemoryError|Cannot allocate memory|std::bad_alloc|OOM|Insufficient Memory)",
     "Memoria insufficiente",
     "Chiudi altre applicazioni oppure scegli un modello Whisper più piccolo (medium o small) nella barra laterale. Se usi Ollama, prova un modello più leggero."),
    (r"(cublas|cudnn|libcudart|CUDA driver|cuda.*not available|no CUDA-capable device|CUDA error)",
     "Librerie o driver NVIDIA non disponibili",
     "In 'Avanzate' imposta Device = cpu e Compute type = int8, oppure aggiorna i driver NVIDIA e installa le librerie: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12."),
    (r"(11434|Ollama|ollama)",
     "Ollama non risponde",
     "Controlla che Ollama sia aperto (icona nella barra dei menu o accanto all'orologio) e premi 'Ricontrolla Ollama'. Poi rigenera il pacchetto: la trascrizione è salva."),
    (r"(Impossibile convertire|Invalid data found|moov atom|could not find codec|Invalid argument|decode)",
     "File audio o video non leggibile",
     "Il file potrebbe essere danneggiato o in un formato insolito. Prova a riesportarlo in mp3 o mp4, oppure installa ffmpeg."),
    (r"File sorgente non trovato",
     "File non trovato",
     "Il file è stato spostato o cancellato. Scegli di nuovo la registrazione nel passo 1."),
    (r"(Permission denied|PermissionError|Errno 13)",
     "Permesso negato",
     "L'app non può leggere o scrivere il file. Copialo in Documenti o Download e riprova."),
    (r"(No space left|Errno 28|disk full)",
     "Spazio su disco esaurito",
     "Libera spazio oppure svuota la cache dalla barra laterale (Avanzate)."),
    (r"(trascrizione è vuota|niente da riassumere)",
     "Nessun parlato riconosciuto",
     "La registrazione sembra vuota o troppo bassa. Controlla il file e, se il volume è basso, disattiva il VAD in 'Avanzate'."),
]


def friendly_error(error: Optional[str]) -> Tuple[str, str, str]:
    """(titolo, suggerimento, dettaglio tecnico) a partire dal testo dell'eccezione."""
    detail = (error or "").strip()
    for pattern, title, hint in _RULES:
        if re.search(pattern, detail, flags=re.IGNORECASE):
            return title, hint, detail
    return "Qualcosa è andato storto", "Riprova. Se il problema continua, apri 'Dettagli' e segnala l'errore su GitHub.", detail
