# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for DC Twin Desktop Application.

Build with:
    pyinstaller build/dc_twin.spec --clean --noconfirm

Output:
    dist/DC Twin.app   (macOS)
    dist/dc-twin/      (Windows/Linux — onedir)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = str(Path(SPECPATH).parent)
IS_MAC = sys.platform == "darwin"

block_cipher = None

# ── Data files ─────────────────────────────────────────────────────────────────

datas = [
    # Application source modules
    (str(Path(PROJECT_ROOT) / "dashboard" / "app.py"),   "dashboard"),
    (str(Path(PROJECT_ROOT) / "dashboard" / "__init__.py"), "dashboard"),
    (str(Path(PROJECT_ROOT) / "backend" / "main.py"),         "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "models.py"),       "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "database.py"),     "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "simulator.py"),    "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "metrics.py"),      "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "alerts.py"),       "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "recommendations.py"), "backend"),
    (str(Path(PROJECT_ROOT) / "backend" / "__init__.py"),     "backend"),
]

# Streamlit ships static web assets (React build, images, etc.)
datas += collect_data_files("streamlit")
# Plotly ships its own JS bundle
datas += collect_data_files("plotly")
# Altair schema files
datas += collect_data_files("altair")
# pyarrow timezone data
datas += collect_data_files("pyarrow")
# Pydeck
datas += collect_data_files("pydeck")

# ── Hidden imports ──────────────────────────────────────────────────────────────

hidden_imports = [
    # Streamlit internals
    "streamlit",
    "streamlit.web.bootstrap",
    "streamlit.web.server",
    "streamlit.web.server.server",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.components.v1",
    "streamlit.components.v1.components",
    # FastAPI / Starlette
    "fastapi",
    "fastapi.middleware.cors",
    "starlette",
    "starlette.middleware.cors",
    # Uvicorn
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # Database
    "sqlalchemy",
    "sqlalchemy.dialects.sqlite",
    "aiosqlite",
    # Data
    "pandas",
    "numpy",
    "scipy",
    "scipy.stats",
    "plotly",
    "plotly.graph_objects",
    "plotly.subplots",
    # Pydantic
    "pydantic",
    "pydantic.v1",
    # Application modules
    "backend",
    "backend.main",
    "backend.models",
    "backend.database",
    "backend.simulator",
    "backend.metrics",
    "backend.alerts",
    "backend.recommendations",
    # PyWebView platform backend
    "webview",
]

if IS_MAC:
    hidden_imports += [
        "webview.platforms.cocoa",
        "Foundation",
        "AppKit",
        "WebKit",
    ]
else:
    hidden_imports += ["webview.platforms.edgechromium"]

# Pull in all streamlit and uvicorn submodules automatically
hidden_imports += collect_submodules("streamlit")
hidden_imports += collect_submodules("uvicorn")
hidden_imports += collect_submodules("httpx")

# ── Analysis ────────────────────────────────────────────────────────────────────

a = Analysis(
    [str(Path(PROJECT_ROOT) / "launcher.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "notebook",
        "jupyter",
        "docutils",
        "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dc-twin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can break some native libs; disable for safety
    console=False,      # No terminal window visible to the user
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(Path(PROJECT_ROOT) / "assets" / "icon.icns") if IS_MAC else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="dc-twin",
)

# ── macOS .app bundle ───────────────────────────────────────────────────────────

if IS_MAC:
    app = BUNDLE(
        coll,
        name="DC Twin.app",
        icon=str(Path(PROJECT_ROOT) / "assets" / "icon.icns"),
        bundle_identifier="com.dctwin.desktop",
        info_plist={
            "CFBundleName":                "DC Twin",
            "CFBundleDisplayName":         "DC Twin",
            "CFBundleVersion":             "1.0.0",
            "CFBundleShortVersionString":  "1.0.0",
            "NSPrincipalClass":            "NSApplication",
            "NSHighResolutionCapable":     True,
            "NSAppleScriptEnabled":        False,
            "LSMinimumSystemVersion":      "11.0",
            "LSUIElement":                 False,   # Show in Dock
        },
    )
