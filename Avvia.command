#!/bin/bash
# Avvio di Sbobinatore con doppio clic (Mac) o "bash Avvia.command" (Linux).
# La prima volta crea l'ambiente Python e installa i componenti; poi apre l'app nel browser.
cd "$(dirname "$0")" || exit 1

echo
echo "  === Sbobinatore ==="
echo

attendi_e_esci() {
  echo
  read -r -p "Premi Invio per chiudere questa finestra. " _
  exit "${1:-1}"
}

# --- Python 3.9+ ---
PY=""
for candidato in python3 python; do
  if command -v "$candidato" >/dev/null 2>&1 && "$candidato" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    PY="$candidato"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9 o successivo non e' installato (o non e' raggiungibile)."
  echo "Scaricalo da https://www.python.org/downloads/ , installalo e poi riapri Avvia.command."
  if [ "$(uname)" = "Darwin" ]; then open "https://www.python.org/downloads/macos/"; fi
  attendi_e_esci 1
fi
echo "Python trovato: $("$PY" --version 2>&1)"

# --- ffmpeg (consigliato, non obbligatorio) ---
if ffmpeg -version >/dev/null 2>&1; then
  echo "ffmpeg trovato."
else
  echo "ffmpeg non trovato: l'app funziona comunque, ma la conversione dell'audio sara' piu' lenta."
  if command -v brew >/dev/null 2>&1; then
    read -r -p "Vuoi installarlo adesso con Homebrew (brew install ffmpeg)? [s/N] " risposta
    case "$risposta" in
      s|S|si|SI|sì) brew install ffmpeg || echo "Installazione ffmpeg non riuscita: continuo senza." ;;
    esac
  elif [ "$(uname)" = "Darwin" ]; then
    echo "Per installarlo in futuro: installa Homebrew da https://brew.sh e poi esegui: brew install ffmpeg"
  else
    echo "Per installarlo in futuro: sudo apt install ffmpeg (o il pacchetto della tua distribuzione)."
  fi
fi

# --- ambiente Python del progetto ---
if [ ! -x ".venv/bin/python" ]; then
  echo "Creo l'ambiente Python del progetto (solo la prima volta)..."
  "$PY" -m venv .venv || { echo "Creazione dell'ambiente fallita."; attendi_e_esci 1; }
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installo i componenti necessari: possono servire alcuni minuti e qualche centinaio di MB..."
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt \
    || { echo "Installazione dei componenti fallita. Controlla la connessione e riprova."; attendi_e_esci 1; }
  cp requirements.txt .venv/requirements.installed
  echo "Componenti installati."
fi

# --- evita la domanda sull'email al primo avvio di Streamlit ---
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
  printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo
echo "Avvio l'app: tra qualche secondo si apre nel browser."
echo "Lascia aperta questa finestra mentre usi Sbobinatore. Per chiudere l'app chiudi questa finestra."
echo
exec .venv/bin/python -m streamlit run app.py
