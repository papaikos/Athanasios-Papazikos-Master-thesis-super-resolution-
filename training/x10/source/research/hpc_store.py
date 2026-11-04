"""Verified durable filesystem publication for a single Slurm writer.

Object and receipt replacements are atomic. A crash between the two fails closed
on recovery; never infer a fresh run from missing or inconsistent durable bytes.
Retention is disabled by default; explicit best/latest retention is confined to
canonical checkpoint names in this new run.
"""
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from .checkpointing import sha256


def _fsync_directory(path):
    if os.name == 'posix':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _contained(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe object path')
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(p in ('.', '..') or p.startswith('.hpc-') for p in parts) or parts[0] == '.receipts':
        raise ValueError('Unsafe object path')
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Object path escapes root')
    return path


def _verify(path, receipt):
    if path.stat().st_size != receipt['bytes'] or sha256(path) != receipt['sha256']:
        raise ValueError('Durable filesystem integrity mismatch')


class FilesystemStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.recovery_diagnostics = []

    def _receipt_path(self, relative):
        path = _contained(self.root, relative)
        receipt = self.root / '.receipts' / path.relative_to(self.root)
        receipt = receipt.with_name(receipt.name + '.json')
        if not receipt.resolve().is_relative_to(self.root):
            raise ValueError('Receipt path escapes root')
        return receipt

    def _publish(self, source, relative, immutable):
        target = _contained(self.root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix='.hpc-publish-', suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'wb') as output, open(source, 'rb') as input_file:
                remaining = os.fstat(input_file.fileno()).st_size
                while remaining:
                    chunk = input_file.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('Source truncated during snapshot')
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            receipt = {'uri': target.as_uri(), 'sha256': sha256(temporary), 'bytes': Path(temporary).stat().st_size}
            if immutable:
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    try:
                        _verify(target, receipt)
                    except ValueError as error:
                        raise ValueError('Durable immutable object differs') from error
            else:
                os.replace(temporary, target)
            _fsync_directory(target.parent)
            _verify(target, receipt)
            receipt_path = self._receipt_path(relative)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, receipt_temp = tempfile.mkstemp(dir=receipt_path.parent, prefix='.hpc-receipt-', suffix='.tmp')
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
                    json.dump(receipt, output, allow_nan=False)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(receipt_temp, receipt_path)
                _fsync_directory(receipt_path.parent)
            finally:
                if os.path.exists(receipt_temp):
                    os.unlink(receipt_temp)
            _verify(target, json.loads(receipt_path.read_text(encoding='utf-8')))
            return receipt
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def upload_verified(self, path, relative):
        return self._publish(path, relative, immutable=True)

    def sync_file(self, path, relative):
        return self._publish(path, relative, immutable=False)

    def pointer(self, relative, value):
        _contained(self.root, relative)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'pointer.json'
            source.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
            return self.sync_file(source, relative)

    def download_verified(self, relative, destination, receipt=None):
        source = _contained(self.root, relative)
        if receipt is None:
            receipt_path = self._receipt_path(relative)
            if not receipt_path.is_file():
                raise ValueError('Missing durable receipt')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        _verify(source, receipt)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix='.hpc-download-', suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'wb') as output, source.open('rb') as input_file:
                shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            _verify(Path(temporary), receipt)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return receipt

    def restore_run(self, destination):
        destination = Path(destination).resolve()
        if destination == self.root or destination.is_relative_to(self.root) or self.root.is_relative_to(destination):
            raise ValueError('Scratch and durable paths must be separate')
        files = [path for path in self.root.rglob('*') if path.is_file()]
        self.recovery_diagnostics = []
        committed = []
        for path in files:
            if re.fullmatch(r'\.hpc-(?:publish|receipt)-[a-z0-9_]{8}\.tmp', path.name):
                self.recovery_diagnostics.append({'event': 'IGNORED_UNCOMMITTED_TEMPORARY', 'relative': path.relative_to(self.root).as_posix()})
            else:
                committed.append(path)
        objects = [path for path in committed if '.receipts' not in path.relative_to(self.root).parts]
        if not objects and committed:
            raise ValueError('Partial durable receipt state; fresh fallback prohibited')
        for path in objects:
            relative = path.relative_to(self.root).as_posix()
            self.download_verified(relative, _contained(destination, relative))
        if objects and not (self.root / 'checkpoints/latest.json').is_file():
            raise ValueError('Partial durable run has no latest checkpoint; fresh fallback prohibited')
        if objects:
            latest = json.loads((destination / 'checkpoints/latest.json').read_text(encoding='utf-8'))
            self.verify_checkpoint_reference(destination, latest)
            for relative in ('checkpoints/best.json', 'checkpoints/events.json'):
                path = destination / relative
                if path.is_file():
                    value = json.loads(path.read_text(encoding='utf-8'))
                    best = value if relative.endswith('best.json') else value.get('best')
                    if best:
                        self.verify_checkpoint_reference(destination, best['checkpoint'])
        return bool(objects)

    @staticmethod
    def verify_checkpoint_reference(root, receipt):
        filename = receipt['filename']
        if not isinstance(filename, str) or Path(filename).name != filename or '/' in filename or '\\' in filename or ':' in filename:
            raise ValueError('Unsafe checkpoint filename')
        path = _contained(Path(root).resolve(), 'checkpoints/' + filename)
        _verify(path, receipt)

    def _read_json_verified(self, relative):
        path = _contained(self.root, relative)
        receipt = json.loads(self._receipt_path(relative).read_text(encoding='utf-8'))
        _verify(path, receipt)
        return json.loads(path.read_text(encoding='utf-8'))

    def prune_checkpoints(self, scratch):
        """Explicit best/latest policy; commit and verify pointers before deletion."""
        scratch = Path(scratch).resolve()
        if scratch == self.root or scratch.is_relative_to(self.root) or self.root.is_relative_to(scratch):
            raise ValueError('Scratch and durable paths must be separate')
        latest = self._read_json_verified('checkpoints/latest.json')
        events = self._read_json_verified('checkpoints/events.json')
        if events['completed_updates'] != latest['step']:
            raise ValueError('Retention requires committed events at latest checkpoint boundary')
        self.verify_checkpoint_reference(self.root, latest)
        protected = {latest['filename']}
        best = events.get('best')
        if best:
            self.verify_checkpoint_reference(self.root, best['checkpoint'])
            published_best = self._read_json_verified('checkpoints/best.json')
            if published_best != best:
                raise ValueError('Best pointer differs from committed events boundary')
            protected.add(best['checkpoint']['filename'])
        elif (self.root / 'checkpoints/best.json').exists():
            raise ValueError('Best pointer lacks committed events reference')
        candidates = []
        for root in (self.root, scratch):
            directory = _contained(root, 'checkpoints')
            for path in directory.glob('*.pt'):
                if path.name in protected or not re.fullmatch(r'step-[0-9]{9}-[0-9a-f]{64}\.pt', path.name):
                    continue
                candidates.append(_contained(root, 'checkpoints/' + path.name))
        if (self.root / 'preflight/results.json').is_file():
            result = self._read_json_verified('preflight/results.json')
            if result.get('status') == 'PASS':
                for root in (self.root, scratch):
                    for path in _contained(root, 'preflight').glob('*.pt'):
                        if path.name == 'remote-resume.pt' or re.fullmatch(r'step-[0-9]{9}-[0-9a-f]{64}\.pt', path.name):
                            candidates.append(_contained(root, 'preflight/' + path.name))
        # All references and containment checks above must succeed before deletion.
        for path in candidates:
            path.unlink()
            _fsync_directory(path.parent)
            if path.is_relative_to(self.root):
                receipt_path = self._receipt_path(path.relative_to(self.root).as_posix())
                receipt_path.unlink(missing_ok=True)
                _fsync_directory(receipt_path.parent)

    def invalidate_finalized(self):
        """Invalidate only this run's completion gate before a new export attempt."""
        relative = 'final/FINALIZED.json'
        path = _contained(self.root, relative)
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)
        receipt_path = self._receipt_path(relative)
        if receipt_path.exists():
            receipt_path.unlink()
            _fsync_directory(receipt_path.parent)
