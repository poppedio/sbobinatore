<#
Costruisce il pacchetto portatile di Sbobinatore per Windows a 64 bit:
Python "embeddable" di python.org + tutte le librerie già installate + l'app + un lanciatore .bat.
Va eseguito su Windows (lo fa il workflow GitHub Actions). Risultato: dist\Sbobinatore-<versione>-windows-x64.zip
#>
param(
    [string]$PythonVersion = "3.12.10",
    [string]$OutDir = "dist"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$build = Join-Path $root "build"
$stage = Join-Path $build "Sbobinatore"

if (Test-Path $build) { Remove-Item -Recurse -Force $build }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# --- Python embeddable ---
$zipName = "python-$PythonVersion-embed-amd64.zip"
Write-Host "Scarico $zipName"
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PythonVersion/$zipName" -OutFile (Join-Path $build $zipName)
Expand-Archive -Path (Join-Path $build $zipName) -DestinationPath (Join-Path $stage "python")
$python = Join-Path $stage "python\python.exe"

# abilita site-packages e "import site" nel file ._pth (di default disattivati nell'embeddable)
$pth = Get-ChildItem (Join-Path $stage "python") -Filter "python*._pth" | Select-Object -First 1
$lines = Get-Content $pth.FullName | ForEach-Object { $_ -replace '^\s*#\s*import site\s*$', 'import site' }
$lines += "Lib\site-packages"
Set-Content -Path $pth.FullName -Value $lines -Encoding ascii

# --- pip e dipendenze ---
Write-Host "Installo pip"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile (Join-Path $build "get-pip.py")
& $python (Join-Path $build "get-pip.py") --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "installazione di pip fallita" }
Write-Host "Installo le dipendenze"
& $python -m pip install --no-warn-script-location --no-cache-dir -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "installazione delle dipendenze fallita" }

# --- app ---
foreach ($f in @("app.py", "LICENSE", "README.md", "requirements.txt")) { Copy-Item (Join-Path $root $f) $stage }
Copy-Item -Recurse (Join-Path $root "transcriptor") (Join-Path $stage "transcriptor")
Copy-Item -Recurse (Join-Path $root ".streamlit") (Join-Path $stage ".streamlit")
Copy-Item (Join-Path $PSScriptRoot "Sbobinatore.bat") $stage
Copy-Item (Join-Path $PSScriptRoot "LEGGIMI.txt") $stage
Get-ChildItem -Path $stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# --- versione e zip ---
$version = (& $python -c "import sys; sys.path.insert(0, r'$stage'); import transcriptor; print(transcriptor.__version__)").Trim()
if (-not $version) { throw "versione non letta" }
$outDirFull = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $outDirFull | Out-Null
$zipOut = Join-Path $outDirFull "Sbobinatore-$version-windows-x64.zip"
if (Test-Path $zipOut) { Remove-Item $zipOut }
Write-Host "Creo $zipOut"
Compress-Archive -Path $stage -DestinationPath $zipOut -CompressionLevel Optimal
$mb = [math]::Round((Get-Item $zipOut).Length / 1MB)
Write-Host "Pronto: $zipOut ($mb MB)"
"zip=$zipOut" | Out-File -Append -FilePath $env:GITHUB_OUTPUT -Encoding ascii -ErrorAction SilentlyContinue
"version=$version" | Out-File -Append -FilePath $env:GITHUB_OUTPUT -Encoding ascii -ErrorAction SilentlyContinue
