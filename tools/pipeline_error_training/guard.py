"""Stdlib-only access guards; install before importing numerical dependencies."""
import os
import hashlib
import threading
from pathlib import Path
import sys


def install(*, source=None, fresh_root=None, allowed_models=(), images=()):
    if source not in {None, '44b6', '6bba'}:
        raise ValueError('Explicit permitted source embryo required')
    target = {'44b6': '6bba', '6bba': '44b6'}.get(source)
    fresh = Path(fresh_root).resolve() if fresh_root else None
    allowed = [Path(p).resolve() for p in [*allowed_models, *images]]
    # These are fixed structuring-element lookup tables shipped by scikit-image,
    # not study predictions or weights. Record the exact bytes actually read.
    runtime_arrays = {str((Path(p)/'skimage/morphology'/name).resolve())
        for p in sys.path if Path(p).name == 'site-packages'
        for name in ['disk_decompositions.npy', 'ball_decompositions.npy']}
    receipt = dict(blocked_reads=0, blocked_network=0,
                   runtime_array_sha256={},
                   blocked_paths=[],
                   installed_before_numerical=not any(s in sys.modules for s in ['numpy', 'torch', 'zarr']))
    relative_open = threading.local()
    original_open = os.open

    def open_at(path, flags, mode=0o777, *, dir_fd=None):
        # Python's audit event omits dir_fd. Preserve the actual directory for
        # relative opens (notably TemporaryDirectory cleanup), without changing
        # the requested filesystem operation or trusting the current directory.
        previous = getattr(relative_open, 'location', None)
        if dir_fd is not None and not Path(os.fsdecode(path)).is_absolute():
            base = Path(os.readlink(f'/proc/self/fd/{dir_fd}'))
            relative_open.location = (os.fsdecode(path), base/os.fsdecode(path))
        try:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            relative_open.location = previous

    if original_open in os.supports_dir_fd:
        os.supports_dir_fd.add(open_at)
    os.open = open_at

    def reject(reason, path):
        import traceback
        receipt['blocked_reads'] += 1
        receipt['blocked_paths'].append(dict(path=str(path),reason=reason,
            stack=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_stack(limit=12)[:-1]]))
        raise PermissionError(reason+': '+str(path))

    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'}:
            receipt['blocked_network'] += 1
            raise PermissionError('Study worker has no network access')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        raw = os.fsdecode(args[0])
        relative = getattr(relative_open, 'location', None)
        path = (relative[1] if relative and relative[0] == raw else Path(raw)).resolve()
        text = str(path)
        if text in runtime_arrays:
            if text not in receipt['runtime_array_sha256']:
                # Reserve before reading to avoid recursively auditing the hash.
                receipt['runtime_array_sha256'][text] = 'hashing'
                receipt['runtime_array_sha256'][text] = hashlib.sha256(path.read_bytes()).hexdigest()
            return
        parts = set(path.parts)
        labels = '.geff' in text or bool(parts & {'evaluation', 'labels', 'oracles', 'oracle', 'matches'})
        if target and any(p == target or p.startswith(target+'_') for p in path.parts):
            reject('Source worker rejected target-embryo input',path)
        if fresh:
            is_own = path == fresh or fresh in path.parents
            permitted = any(path == p or p in path.parents for p in allowed)
            if labels and not is_own:
                reject('Fresh inference rejected annotation input',path)
            if (not is_own and not permitted and
                    (path.suffix in {'.npz', '.npy', '.pt', '.pth', '.keras', '.joblib', '.parquet', '.csv', '.tsv', '.pkl', '.pickle'}
                     or parts & {'predictions', 'selected_predictions', 'candidate_graphs', 'fresh_evidence', 'candidate_evidence'})):
                reject('Fresh inference rejected old prediction/feature cache',path)
    sys.addaudithook(audit)
    return receipt
