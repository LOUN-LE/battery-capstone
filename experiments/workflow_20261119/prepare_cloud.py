from pathlib import Path
import hashlib
import json
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path(r"<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo")
DEST = HERE / "aws_pilot32_release"
DEST.mkdir(exist_ok=True)
files = ("simulation/risk_scenarios.py", "simulation/rebuilt_lifetime.py", "simulation/test_risk_scenarios.py",
         "configs/model.json", "configs/m50_c_joint_v3.json")
for relative in files:
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / relative, target)
(DEST / "simulation/__init__.py").write_text("", encoding="utf-8")
for name in ("run_aws_pilot.sh", "cloud_receipt.py"):
    (DEST / name).write_text((HERE / name).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
(DEST / "requirements-pilot.txt").write_text("pybamm==26.8.0.0\npybammsolvers==0.9.1\nnumpy==2.4.6\nscipy==1.17.1\ncasadi==3.7.2\n", encoding="utf-8")
manifest = {"release": "capstone-risk-pilot32-20261115", "files": {
    p.relative_to(DEST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(DEST.rglob("*")) if p.is_file() and p.name != "release_manifest.json"}}
(DEST / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
archive = ROOT / "outputs/aws_pilot32_code.tar.gz"
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive, "w:gz") as handle:
    handle.add(DEST, arcname="capstone-risk-pilot32-20261115")
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "manifest": manifest}, indent=2))
