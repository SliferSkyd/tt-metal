import json
from pathlib import Path

import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(args):
    script = REPO_ROOT / ".github" / "scripts" / "ci_impact" / "ci_impact.py"
    out = subprocess.check_output(["python3", str(script), *args], cwd=str(REPO_ROOT))
    return json.loads(out.decode("utf-8"))


def test_recommends_ttnn_on_ttnn_change(tmp_path):
    # Create a synthetic changed file inside ttnn
    changed = REPO_ROOT / "ttnn" / "dummy.py"
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("import ttnn\n", encoding="utf-8")

    try:
        # Call with explicit changed-file to avoid depending on git state
        data = run_script(["--changed-file", str(changed.relative_to(REPO_ROOT))])
    finally:
        try:
            changed.unlink()
        except FileNotFoundError:
            pass

    comps = set(data["impacted_components"]) if isinstance(data, dict) else set()
    recs = [r.lower() for r in data.get("recommended_workflows", [])]

    assert "ttnn" in comps
    assert any("ttnn" in r or "tt-nn" in r for r in recs)


