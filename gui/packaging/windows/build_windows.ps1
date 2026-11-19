# Build the Windows installer of LHDetect3.
#
#   cd lhdetect3\gui
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\packaging\windows\build_windows.ps1
#
# Needs: the ONNX model in lhdetect3_gui\models (git lfs pull), internet access.
# Python 3.12 and Inno Setup 6 are installed with winget for the current user if missing.
# Result: gui\dist\windows\LHDetect3-<version>-win64-setup.exe
param(
    [string]$Python = "",
    [switch]$PortableZip
)
$ErrorActionPreference = "Stop"
$Gui = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Build = Join-Path $Gui "build\windows"
$Out = Join-Path $Gui "dist\windows"
$Model = Join-Path $Gui "lhdetect3_gui\models\lhdetect3_rfdetr_seg_small.onnx"

function Step($t) { Write-Host "`n==> $t" -ForegroundColor Green }
function Invoke-Native {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $exe = $args[0]; $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { throw "$exe failed (exit code $code)" }
}

# ---- 1. model
if (-not (Test-Path $Model) -or (Get-Item $Model).Length -lt 50MB) {
    throw "Model not found or not downloaded: $Model`nRun 'git lfs pull' in the repository, or export it with tools\export_onnx.py."
}
Step "Model: $Model ($([math]::Round((Get-Item $Model).Length / 1MB)) MB)"

# ---- 2. Python 3.12 (3.11 and 3.13 also work)
function Find-Python {
    foreach ($v in @("3.12", "3.11", "3.13")) {
        try {
            $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
        } catch { }
    }
    foreach ($d in @("Python312", "Python311", "Python313")) {
        $p = Join-Path $env:LOCALAPPDATA "Programs\Python\$d\python.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}
if (-not $Python) { $Python = Find-Python }
if (-not $Python) {
    Step "Installing Python 3.12 with winget (current user)"
    Invoke-Native winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    $Python = Find-Python
    if (-not $Python) { throw "Python was not found after installation. Open a new PowerShell window and run the script again." }
}
Step "Python: $Python"

# ---- 3. build environment
$Venv = Join-Path $Build "venv"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Step "Creating the build environment"
    Invoke-Native $Python -m venv $Venv
}
$Py = Join-Path $Venv "Scripts\python.exe"
Step "Installing packages"
Invoke-Native $Py -m pip install --upgrade pip wheel
Invoke-Native $Py -m pip install -r (Join-Path $PSScriptRoot "requirements-windows.txt")
Invoke-Native $Py -c "import onnxruntime as o; print('ONNX Runtime', o.__version__, o.get_available_providers())"
$Version = (& $Py -c "import sys; sys.path.insert(0, r'$Gui'); import lhdetect3_gui; print(lhdetect3_gui.__version__)").Trim()

# ---- 4. PyInstaller
Step "Building LHDetect3 $Version with PyInstaller"
$Dist = Join-Path $Build "dist"
Push-Location $Gui
try {
    Invoke-Native $Py -m PyInstaller --noconfirm --clean --distpath $Dist --workpath (Join-Path $Build "work") "packaging\lhdetect3.spec"
} finally { Pop-Location }
$App = Join-Path $Dist "LHDetect3"

# ---- 5. self-test of the built program
Step "Self-test (model loading, device selection, reference image)"
$Report = Join-Path $Build "selftest.txt"
Remove-Item $Report -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $App "LHDetect3.exe") -ArgumentList @("--selftest", "`"$Report`"") -Wait -PassThru
if (Test-Path $Report) { Get-Content $Report | ForEach-Object { Write-Host "    $_" } }
if ($p.ExitCode -ne 0) { throw "Self-test failed, see $Report" }

# ---- 6. installer (Inno Setup 6)
New-Item -ItemType Directory -Force -Path $Out | Out-Null
function Find-Iscc {
    $c = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    foreach ($x in $c) { if ($x -and (Test-Path $x)) { return $x } }
    return $null
}
$Iscc = Find-Iscc
if (-not $Iscc) {
    Step "Installing Inno Setup 6 with winget (current user)"
    try {
        Invoke-Native winget install -e --id JRSoftware.InnoSetup --scope user --silent --accept-package-agreements --accept-source-agreements
    } catch { Write-Warning "winget could not install Inno Setup: $_" }
    $Iscc = Find-Iscc
}
if ($Iscc) {
    Step "Building the installer"
    $Icon = Join-Path $Gui "lhdetect3_gui\resources\icon.ico"
    Invoke-Native $Iscc "/Qp" "/DAppVersion=$Version" "/DSourceDir=$App" "/DOutputDir=$Out" "/DIconFile=$Icon" (Join-Path $PSScriptRoot "lhdetect3.iss")
    $Setup = Join-Path $Out "LHDetect3-$Version-win64-setup.exe"
    Step "Installer: $Setup ($([math]::Round((Get-Item $Setup).Length / 1MB)) MB)"
} else {
    Write-Warning "Inno Setup not found; making a portable ZIP instead."
    $PortableZip = $true
}
if ($PortableZip) {
    $Zip = Join-Path $Out "LHDetect3-$Version-win64-portable.zip"
    Remove-Item $Zip -ErrorAction SilentlyContinue
    Compress-Archive -Path $App -DestinationPath $Zip -CompressionLevel Optimal
    Step "Portable ZIP: $Zip ($([math]::Round((Get-Item $Zip).Length / 1MB)) MB)"
}
Write-Host "`nDone. Test the program: $App\LHDetect3.exe" -ForegroundColor Green
