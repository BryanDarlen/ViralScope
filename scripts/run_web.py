from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) != 0


def _choose_port(host: str, preferred_port: int, attempts: int) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        if _port_available(host, port):
            return port
    raise RuntimeError(
        f"Could not find an open port between {preferred_port} and {preferred_port + attempts - 1}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ViralScope FastAPI web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--port-search", type=int, default=10, help="How many ports to try from the starting port.")
    args = parser.parse_args()

    selected_port = _choose_port(args.host, args.port, max(1, args.port_search))
    if selected_port != args.port:
        print(f"Port {args.port} is already in use. Starting ViralScope on port {selected_port} instead.")
    print(f"Open ViralScope at http://{args.host}:{selected_port}")
    uvicorn.run("webapp.main:app", host=args.host, port=selected_port, reload=False)


if __name__ == "__main__":
    main()
