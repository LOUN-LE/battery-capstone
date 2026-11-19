from pathlib import Path
import hashlib
import json
import zipfile

cwd = Path(r'<USER_HOME>\Documents\Codex\2026-11-15\dl')
repo = Path(r'<USER_HOME>\Documents\Codex\2026-10-30\xzoq\outputs\capstone\repo')
files = {}
for folder in ('scenario_validation', 'thermal_validation', 'thermal_abuse_validation'):
    for path in (cwd / 'outputs' / folder).rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.csv', '.txt', '.md'):
            files['evidence/' + path.relative_to(cwd / 'outputs').as_posix()] = path
for relative in ('simulation/risk_scenarios.py', 'simulation/test_risk_scenarios.py',
                 'simulation/thermal_validation.py', 'simulation/verify_thermal_validation.py',
                 'simulation/thermal_abuse_validation.py', 'simulation/test_thermal_abuse_validation.py',
                 'simulation/rebuilt_lifetime.py', 'configs/model.json', 'configs/m50_c_joint_v3.json',
                 'docs/thermal_abuse_validation.md'):
    files['code/' + relative] = repo / relative
manifest = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
            for name, path in sorted(files.items())}
target = cwd / 'outputs' / 'capstone_validation_evidence_20261115.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
    for name, path in sorted(files.items()):
        z.write(path, name)
    z.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    z.writestr('README.txt', '2026-11-15 capstone validation evidence snapshot.\n'
               'Includes local numerical test reports/traces, measured thermal replay reports, public ARC processing results and relevant code/config.\n'
               'Earlier iterations are preserved; numerical_verification.json contains the final bounded pilot verdict and explicit source versions.\n'
               'No AWS completion CSV or physical thermal-runaway/fire validation is claimed. See Notion current progress for cloud runtime status.\n'
               'MANIFEST.json records SHA-256 for every included evidence/code file.\n')
print(json.dumps({'file': str(target), 'bytes': target.stat().st_size, 'files': len(files),
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}, ensure_ascii=False))
