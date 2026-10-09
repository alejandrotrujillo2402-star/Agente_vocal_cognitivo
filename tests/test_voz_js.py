"""Corre las pruebas de static/voz.js (corrección por vocabulario y diarización) con Node."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not shutil.which("node"), reason="Node no está instalado")
def test_voz_js():
    r = subprocess.run(["node", "--test", "tests/voz.test.js"], cwd=Path(__file__).parent.parent,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
