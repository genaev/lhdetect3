# PyInstaller specification for LHDetect3 (one-folder build).
#   pyinstaller --noconfirm packaging/lhdetect3.spec      (run from the gui folder)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

GUI = Path(SPECPATH).resolve().parent
PKG = GUI / "lhdetect3_gui"
ICON = PKG / "resources" / ("icon.ico" if sys.platform.startswith("win") else "icon.png")

datas = [
    (str(PKG / "models" / "lhdetect3_rfdetr_seg_small.onnx"), "lhdetect3_gui/models"),
    (str(PKG / "models" / "reference.jpg"), "lhdetect3_gui/models"),
    (str(PKG / "resources"), "lhdetect3_gui/resources"),
]
binaries = collect_dynamic_libs("onnxruntime")   # includes DirectML.dll on Windows

excludes = [
    "tkinter", "matplotlib", "scipy", "pandas", "torch", "IPython", "pytest",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtNetwork",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtXml", "PySide6.QtDBus", "PySide6.QtPdf", "PySide6.QtSvg", "PySide6.QtSvgWidgets",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools", "PySide6.QtConcurrent",
]

a = Analysis(
    [str(GUI / "packaging" / "launcher.py")],
    pathex=[str(GUI)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["lhdetect3_gui.app", "lhdetect3_gui.cli"],
    excludes=excludes,
    noarchive=False,
)
b = Analysis(
    [str(GUI / "packaging" / "launcher_cli.py")],
    pathex=[str(GUI)],
    hiddenimports=["lhdetect3_gui.cli"],
    excludes=excludes + ["PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="LHDetect3",
    icon=str(ICON),
    console=False,
    upx=False,
)
pyz_cli = PYZ(b.pure)
exe_cli = EXE(
    pyz_cli, b.scripts, [],
    exclude_binaries=True,
    name="lhdetect3-cli",
    icon=str(ICON),
    console=True,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, exe_cli, b.binaries, b.datas,
               strip=False, upx=False, name="LHDetect3")
