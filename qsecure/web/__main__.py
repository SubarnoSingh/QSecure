"""The one command: ``python -m qsecure.web``."""

from __future__ import annotations

import argparse
import socket
from typing import Sequence

import uvicorn

from qsecure import __version__
from qsecure.audit import DEFAULT_CAPACITY, AuditLog
from qsecure.web.api import STATIC_DIR, create_app
from qsecure.web.limits import (
    LIVE_KEY_LENGTH_MAX,
    MAX_CONCURRENT_RUNS,
    MAX_REQUEST_BYTES,
)


__all__ = ["bind_hosts", "build_parser", "main", "open_listeners"]


#: Hosts meaning "the local machine", bound in both address families.
_LOOPBACK_PAIRS: dict[str, tuple[str, ...]] = {
    "127.0.0.1": ("127.0.0.1", "::1"),
    "localhost": ("127.0.0.1", "::1"),
    "::1": ("::1", "127.0.0.1"),
    "0.0.0.0": ("0.0.0.0", "::"),
    "::": ("::", "0.0.0.0"),
}


def bind_hosts(host: str) -> tuple[str, ...]:
    """Return every address ``host`` should be bound on, in order.

    >>> from qsecure.web.__main__ import bind_hosts
    >>> bind_hosts("127.0.0.1")
    ('127.0.0.1', '::1')
    >>> bind_hosts("0.0.0.0")
    ('0.0.0.0', '::')

    A named interface is one socket, because the operator named it:
    >>> bind_hosts("192.168.1.40")
    ('192.168.1.40',)
    """
    return _LOOPBACK_PAIRS.get(host, (host,))


def open_listeners(
    host: str, port: int
) -> tuple[list[socket.socket], list[str], list[str]]:
    """Bind the listening sockets, before anything is printed.

    A loopback bind opens both families and reports both:
    >>> import socket as _socket
    >>> from qsecure.web.__main__ import open_listeners
    >>> probe = _socket.socket()
    >>> probe.bind(("127.0.0.1", 0))
    >>> free = probe.getsockname()[1]
    >>> probe.close()
    >>> sockets, bound, skipped = open_listeners("127.0.0.1", free)
    >>> "http://127.0.0.1:%d" % free in bound
    True
    >>> for listener in sockets:
    ...     listener.close()

    >>> sockets, bound, skipped = open_listeners("127.0.0.1", 0)
    >>> ports = {listener.getsockname()[1] for listener in sockets}
    >>> len(ports) == 1 and 0 not in ports
    True
    >>> [url.rsplit(":", 1)[1] for url in bound] == [
    ...     str(listener.getsockname()[1]) for listener in sockets
    ... ]
    True
    >>> for listener in sockets:
    ...     listener.close()

    A busy port raises here rather than being announced and then failing:
    >>> held = _socket.socket()
    >>> held.setsockopt(_socket.SOL_SOCKET, _socket.SO_EXCLUSIVEADDRUSE
    ...                 if hasattr(_socket, "SO_EXCLUSIVEADDRUSE")
    ...                 else _socket.SO_REUSEADDR, 1)
    >>> held.bind(("127.0.0.1", 0))
    >>> held.listen(8)
    >>> busy = held.getsockname()[1]
    >>> try:
    ...     open_listeners("127.0.0.1", busy)
    ... except OSError:
    ...     print("refused, and nothing was printed")
    refused, and nothing was printed
    >>> held.close()
    """
    sockets: list[socket.socket] = []
    bound: list[str] = []
    skipped: list[str] = []
    # `--port 0` yields a port per socket, so the first bind fixes it for the rest.
    wanted = port
    for index, address in enumerate(bind_hosts(host)):
        family = (
            socket.AF_INET6 if ":" in address else socket.AF_INET
        )
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                # Otherwise Windows lets a stale instance keep serving the same address.
                listener.setsockopt(
                    socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                )
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family is socket.AF_INET6:
                # Dual-stack v6 would clash with the v4 wildcard bind on Linux.
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            listener.bind((address, wanted))
            listener.listen(128)
        except OSError as failure:
            listener.close()
            if index == 0:
                for opened in sockets:
                    opened.close()
                raise
            skipped.append(f"{address} ({failure.strerror or failure})")
            continue
        sockets.append(listener)
        # The port actually bound; differs from the request under `--port 0`.
        wanted = listener.getsockname()[1]
        shown = f"[{address}]" if ":" in address else address
        bound.append(f"http://{shown}:{wanted}")
    return sockets, bound, skipped


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    >>> from qsecure.web.__main__ import build_parser
    >>> args = build_parser().parse_args([])
    >>> args.host, args.port
    ('127.0.0.1', 8141)
    >>> build_parser().parse_args(["--port", "9000"]).port
    9000
    """
    parser = argparse.ArgumentParser(
        prog="python -m qsecure.web",
        description=(
            "Serve the QSecure detection dashboard: the JSON API and the "
            "static frontend, in one process, fetching nothing from a network."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface to bind. Defaults to loopback; pass 0.0.0.0 "
            "deliberately to expose the demo on a venue network. Loopback and "
            "the wildcards are bound in BOTH address families, so a browser "
            "that resolves localhost to ::1 reaches the same server."
        ),
    )
    parser.add_argument("--port", type=int, default=8141, help="TCP port.")
    parser.add_argument(
        "--audit-log",
        default=None,
        metavar="PATH",
        help=(
            "Append the security event log to this file, as JSON Lines, one "
            "line per verification outcome. Without it the log is kept in "
            "memory only and dies with the process; either way it retains at "
            f"most {DEFAULT_CAPACITY} events and GET /api/events serves them."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="uvicorn log level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Start the server."""
    args = build_parser().parse_args(argv)
    try:
        sockets, bound, skipped = open_listeners(args.host, args.port)
    except OSError as failure:
        print(
            f"QSecure {__version__} -- COULD NOT START.\n"
            f"  port {args.port} on {args.host} is not available: "
            f"{failure.strerror or failure}\n"
            f"  Nothing is serving and no address is printed above, "
            f"deliberately: an instance you started earlier may still be "
            f"holding this port and answering on it. Stop that one, or pass "
            f"--port with a free port.",
            flush=True,
        )
        return 1
    # After the bind, so a busy port leaves no empty log file behind.
    try:
        audit = AuditLog(path=args.audit_log)
    except OSError as failure:
        for listener in sockets:
            listener.close()
        print(
            f"QSecure {__version__} -- COULD NOT START.\n"
            f"  the security event log cannot be written to "
            f"{args.audit_log}: {failure.strerror or failure}\n"
            f"  Nothing is serving. Pass a path in a directory that exists, "
            f"or drop --audit-log and the log is kept in memory.",
            flush=True,
        )
        return 1

    frontend = "present" if (STATIC_DIR / "index.html").is_file() else "MISSING"
    addresses = "\n".join(f"                  {url}" for url in bound)
    print(
        f"QSecure {__version__} -- listening, bound before this line was "
        f"printed:\n{addresses}\n"
        + (
            f"  not bound       {', '.join(skipped)}\n"
            if skipped
            else ""
        )
        + f"  frontend        {frontend} ({STATIC_DIR})\n"
        f"  live key length up to L = {LIVE_KEY_LENGTH_MAX}; longer runs are "
        f"refused, never clamped\n"
        f"  concurrent runs at most {MAX_CONCURRENT_RUNS}\n"
        f"  request body    at most {MAX_REQUEST_BYTES} bytes; larger is 413 "
        f"before the app sees it\n"
        f"  security log    "
        + (
            f"appended to {audit.path} and "
            if audit.path is not None
            else "in memory only, "
        )
        + f"at most {audit.capacity} events; GET /api/events\n"
        f"  network         nothing is fetched; /docs is off because Swagger "
        f"UI loads from a CDN",
        flush=True,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(audit_log=audit), log_level=args.log_level
        )
    )
    server.run(sockets=sockets)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
