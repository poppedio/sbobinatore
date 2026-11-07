## Sbobinatore per Windows, versione portatile

Scarica `Sbobinatore-<versione>-windows-x64.zip`, estrailo (tasto destro, *Estrai tutto*) e fai doppio clic su `Sbobinatore.bat`. Dentro c'è già Python con tutte le librerie: non serve installare nulla.

- Al primo avvio Windows SmartScreen può mostrare "PC protetto da Windows": premi *Ulteriori informazioni* e poi *Esegui comunque*. Il pacchetto non è firmato digitalmente.
- La prima trascrizione scarica il modello di riconoscimento vocale (circa 1,6 GB).
- Il pacchetto usa la CPU. Con una scheda NVIDIA puoi impostare Device = `cuda` nelle impostazioni avanzate.
- Per sintesi, capitoli e appunti installa [Ollama](https://ollama.com) e scarica un modello: vedi il README.

Per Mac e Linux usa `Avvia.command` dal codice sorgente, come spiegato nel README.
