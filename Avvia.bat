@echo off
setlocal
cd /d "%~dp0"
title Sbobinatore
echo.
echo   === Sbobinatore ===
echo.

rem --- Python 3.9+ ---
set "PY="
where py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
  if not errorlevel 1 set "PY=py -3"
)
if defined PY goto python_ok
where python >nul 2>&1
if not errorlevel 1 (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if defined PY goto python_ok

echo Python 3.9 o successivo non e' installato.
echo Scaricalo da https://www.python.org/downloads/windows/ e durante l'installazione
echo spunta la casella "Add python.exe to PATH". Poi riapri Avvia.bat.
start https://www.python.org/downloads/windows/
goto fine

:python_ok
for /f "tokens=*" %%v in ('%PY% --version 2^>^&1') do echo Python trovato: %%v

rem --- ffmpeg (consigliato, non obbligatorio) ---
where ffmpeg >nul 2>&1
if not errorlevel 1 (
  echo ffmpeg trovato.
  goto ffmpeg_ok
)
echo ffmpeg non trovato: l'app funziona comunque, ma la conversione dell'audio sara' piu' lenta.
where winget >nul 2>&1
if errorlevel 1 (
  echo Per installarlo in futuro: winget install Gyan.FFmpeg
  goto ffmpeg_ok
)
set "RISP=N"
set /p "RISP=Vuoi installarlo adesso con winget? [S/N] "
if /i not "%RISP%"=="S" goto ffmpeg_ok
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
echo.
echo ffmpeg installato. Chiudi questa finestra e riapri Avvia.bat per renderlo attivo.
goto fine

:ffmpeg_ok
rem --- ambiente Python del progetto ---
if exist ".venv\Scripts\python.exe" goto venv_ok
echo Creo l'ambiente Python del progetto (solo la prima volta)...
%PY% -m venv .venv
if errorlevel 1 (
  echo Creazione dell'ambiente fallita.
  goto fine
)

:venv_ok
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if not errorlevel 1 goto deps_ok
echo Installo i componenti necessari: possono servire alcuni minuti e qualche centinaio di MB...
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo Installazione dei componenti fallita. Controlla la connessione e riprova.
  goto fine
)
copy /y requirements.txt .venv\requirements.installed >nul
echo Componenti installati.

:deps_ok
rem --- evita la domanda sull'email al primo avvio di Streamlit ---
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

echo.
echo Avvio l'app: tra qualche secondo si apre nel browser.
echo Lascia aperta questa finestra mentre usi Sbobinatore. Per chiudere l'app chiudi questa finestra.
echo.
.venv\Scripts\python.exe -m streamlit run app.py

:fine
echo.
pause
