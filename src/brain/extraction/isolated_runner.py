# MIT License. Copyright (c) 2023-present DanswerAI, Inc.; Copyright (c) 2026 Angel Murillo.
# Derived from onyx/utils/isolated_runner.py.
"""Child entry point for `run_in_isolated_process`.

Reads a pickled (callable, args, kwargs) from stdin, runs it, and writes the
pickled result back over a private copy of stdout.

Run as ``python -m brain.extraction.isolated_runner`` so it works even when the
caller is a daemon process, which cannot spawn multiprocessing children.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Keep a private copy of real stdout for the result, then point stdout and
    # stderr at devnull so imports or native-library chatter cannot corrupt the
    # pickled payload the parent is reading.
    result_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    import pickle

    from brain.extraction.isolation import STATUS_EXC, STATUS_OK, STATUS_UNRELAYABLE

    fn, args, kwargs = pickle.loads(sys.stdin.buffer.read())
    try:
        payload = (STATUS_OK, fn(*args, **kwargs))
    except Exception as e:
        payload = (STATUS_EXC, e)

    try:
        data = pickle.dumps(payload)
    except Exception:
        # Result or exception didn't pickle; relay its repr so the parent still raises.
        data = pickle.dumps((STATUS_UNRELAYABLE, repr(payload[1])))

    os.write(result_fd, data)
    os.close(result_fd)


if __name__ == "__main__":
    main()
