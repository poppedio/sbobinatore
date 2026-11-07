@echo off
setlocal
cd /d "%~dp0"
title Sbobinatore
rem UTF-8 per Python: evita errori di codifica con accenti ed emoji nella console
set PYTHONUTF8=1

rem evita la domanda sull'email al primo avvio di Streamlit
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

echo.
echo   === Sbobinatore ===
echo.
echo Avvio l'app: tra qualche secondo si apre nel browser.
echo La prima trascrizione scarica il modello di riconoscimento vocale (circa 1,6 GB): abbi pazienza.
echo Lascia aperta questa finestra mentre usi Sbobinatore. Per chiudere l'app chiudi questa finestra.
echo.
"%~dp0python\python.exe" -m streamlit run "%~dp0app.py"
echo.
echo Sbobinatore si e' chiuso.
pause
