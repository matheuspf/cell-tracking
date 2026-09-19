"""Fail-closed regression guard, not a complete operating-system sandbox.

Workers install before numpy/torch/zarr. Source image/label roots are enumerated
from a manifest. Subprocesses and network are forbidden after installation.
"""
import os
from pathlib import Path
import sys
import threading


def install(*, inputs, outputs, code_roots):
    if any(x in sys.modules for x in ('torch', 'numpy', 'zarr')):
        raise RuntimeError('Read guard installed after numerical imports')
    allowed = tuple(Path(p).resolve() for p in inputs)
    own = tuple(Path(p).resolve() for p in outputs)
    runtime = tuple(Path(p).resolve() for p in code_roots)
    receipt = dict(installed_before_numerical=True, denied=0, network_denied=0, subprocess_denied=0)
    state = threading.local()
    original_open = os.open

    def beneath(p, roots):
        return any(p == root or p.is_relative_to(root) for root in roots)

    def open_at(path, flags, mode=0o777, *, dir_fd=None):
        previous = getattr(state, 'relative', None)
        if dir_fd is not None and not Path(os.fsdecode(path)).is_absolute():
            state.relative = (os.fsdecode(path), Path(os.readlink(f'/proc/self/fd/{dir_fd}'))/os.fsdecode(path))
        try:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            state.relative = previous

    if original_open in os.supports_dir_fd:
        os.supports_dir_fd.add(open_at)
    os.open = open_at

    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'):
            receipt['network_denied'] += 1
            raise PermissionError('Worker network forbidden')
        if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty'):
            receipt['subprocess_denied'] += 1
            raise PermissionError('Delegated worker reads forbidden')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        raw = os.fsdecode(args[0])
        pair = getattr(state, 'relative', None)
        path = (pair[1] if pair and pair[0] == raw else Path(raw)).resolve()
        if beneath(path, allowed + own):
            return
        if beneath(path, runtime) and not any(p.endswith(('.geff', '.zarr')) for p in path.parts):
            return
        # System configuration and devices are not model/data artifacts.
        if beneath(path, tuple(Path(x) for x in ('/proc', '/sys', '/dev', '/etc', '/usr/lib', '/usr/share'))):
            return
        receipt['denied'] += 1
        raise PermissionError('Unapproved worker input: ' + str(path))

    sys.addaudithook(audit)
    return receipt
