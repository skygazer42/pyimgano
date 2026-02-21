import subprocess
import sys


def test_importing_openclip_backend_does_not_import_open_clip():
    # Even if OpenCLIP is installed, importing the backend module should not
    # eagerly import `open_clip` (optional dependency).
    code = (
        "import sys\n"
        "import pyimgano.models.openclip_backend\n"
        "assert 'open_clip' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
