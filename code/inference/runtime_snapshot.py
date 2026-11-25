"""Content-addressed source snapshots; operator artifacts never enter Agent input."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import tempfile


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture_snapshot(repo, destination, *, extra_files=None):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    files = {str(p.relative_to(repo)): p.read_bytes() for p in (repo / 'code/inference').glob('*.py')
             if not p.name.startswith(('._', 'test_'))}
    for directory in ('prompts/inference', 'prompts/schemas'):
        for p in (repo / directory).rglob('*'):
            if p.is_file() and not p.name.startswith('._') and p.suffix in {'.md', '.json'}:
                files[str(p.relative_to(repo))] = p.read_bytes()
    # Provider config contains routing and credential environment-variable names,
    # never the values of those variables or HTTP Authorization headers.
    for name in ('config/model_providers.json', 'release_manifest.json', 'SHA256SUMS'):
        if (repo / name).is_file():
            files[name] = (repo / name).read_bytes()
    files.update(extra_files or {})
    for name in files:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Snapshot entry must be relative')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    dependencies = {'python': platform.python_version()}
    for package in ('jsonschema',):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = 'unavailable'
    payload = {'schema_version': 'proseekbench.runtime-source-snapshot.v1', 'files': hashes,
               'dependencies': dependencies}
    digest = canonical_hash(payload)
    target = destination / digest
    destination.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=destination) as tmp:
            staging = Path(tmp) / 'bundle'
            staging.mkdir()
            for name, data in files.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (staging / 'snapshot.json').write_text(json.dumps(payload, indent=2) + '\n')
            try:
                staging.rename(target)
            except OSError:
                if not target.exists():
                    raise
    verify_snapshot(target, expected=digest)
    return {'sha256': digest, 'path': str(target), 'manifest': str(target / 'snapshot.json'),
            'dependencies': dependencies}


def verify_snapshot(target, expected=None):
    target = Path(target)
    data = json.loads((target / 'snapshot.json').read_text())
    digest = canonical_hash(data)
    if expected is not None and digest != expected:
        raise ValueError('Runtime snapshot manifest mismatch')
    for name, expected_hash in data['files'].items():
        path = (target / name).resolve()
        if not path.is_relative_to(target.resolve()):
            raise ValueError('Snapshot path escapes bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Runtime snapshot file mismatch: {name}')
    actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()
              and not p.name.startswith('._') and '__pycache__' not in p.parts and p.name != 'snapshot.json'}
    if actual != set(data['files']):
        raise ValueError('Runtime snapshot contains unlisted or missing files')
    return digest
