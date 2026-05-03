from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_audit_release_surface_module():
    script_path = Path("tools/audit_release_surface.py")
    spec = importlib.util.spec_from_file_location("audit_release_surface", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_readme_link_audit_flags_relative_publish_paths() -> None:
    module = _load_audit_release_surface_module()
    issues = module._find_package_readme_link_issues(
        """
<img src="assets/banner.svg" alt="banner"/>
<a href="docs/START_HERE.md">Docs</a>
[Quickstart](docs/QUICKSTART.md)
[License](LICENSE)
<a href="#top">Top</a>
        """.strip()
    )

    assert any("assets/banner.svg" in issue for issue in issues)
    assert any("docs/START_HERE.md" in issue for issue in issues)
    assert any("docs/QUICKSTART.md" in issue for issue in issues)
    assert any("LICENSE" in issue for issue in issues)
    assert not any("#top" in issue for issue in issues)


def test_package_readme_link_audit_accepts_absolute_urls_and_anchors() -> None:
    module = _load_audit_release_surface_module()
    issues = module._find_package_readme_link_issues(
        """
<img src="https://raw.githubusercontent.com/skygazer42/pyimgano/main/assets/banner.svg" alt="banner"/>
<a href="https://github.com/skygazer42/pyimgano/blob/main/docs/START_HERE.md">Docs</a>
[Quickstart](https://github.com/skygazer42/pyimgano/blob/main/docs/QUICKSTART.md)
[Top](#top)
        """.strip()
    )

    assert issues == []


def test_audit_release_surface_script_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_release_surface.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
