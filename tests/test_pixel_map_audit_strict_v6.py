import subprocess
import sys


def test_pixel_map_audit_strict_passes():
    subprocess.run(
        [sys.executable, "tools/audit_pixel_map_models.py", "--strict"],
        check=True,
    )

