"""Current ParameterValues and runtime identity for new diagnostic checkpoints."""
from __future__ import annotations
import hashlib,importlib.metadata,inspect,json,platform
from pathlib import Path
from run_exposure_scenario_smoke import sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_identity(parameter_values_json,versions,source_paths):
    sources={str(Path(p).resolve()):sha(p) for p in sorted(set(source_paths),key=str)}
    content=dict(actual_parameter_values=parameter_values_json,runtime_versions=versions,source_sha256=sources)
    return dict(schema_version=1,sha256=digest(content),**content)


def capture(parameters,project_sources):
    """PyBaMM's native encoder retains callable parameter expressions and data."""
    import pybamm
    versions=dict(python=platform.python_version(),pybamm=pybamm.__version__)
    for package in ('numpy','scipy','casadi','pybammsolvers'):
        versions[package]=importlib.metadata.version(package)
    sources=[Path(pybamm.__file__).resolve(),*map(Path,project_sources)]
    for key,value in parameters.items():
        if callable(value):
            filename=inspect.getsourcefile(value)
            if filename is None:raise ValueError('Callable parameter source cannot be bound: '+key)
            sources.append(Path(filename).resolve())
    return make_identity(parameters.to_json(),versions,sources)


def validate_identical(old,new):
    for item in (old,new):
        content={k:item[k] for k in ('actual_parameter_values','runtime_versions','source_sha256')}
        if item.get('schema_version')!=1 or item.get('sha256')!=digest(content):
            raise ValueError('Runtime identity content or digest changed')
    if old['sha256']!=new['sha256']:
        raise ValueError('Physical parameters, callable source, dependency source or runtime changed')
