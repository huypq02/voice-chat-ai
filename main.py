#!/usr/bin/env python
from __future__ import annotations

import os
import socket
import sys

# Add src to path so imports resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn
from voicechatai.interfaces.api.app import app


def _is_port_available(host: str, port: int) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			sock.bind((host, port))
		except OSError:
			return False
	return True


def _find_available_port(host: str, start_port: int, max_attempts: int = 20) -> int:
	for port in range(start_port, start_port + max_attempts):
		if _is_port_available(host, port):
			return port
	raise RuntimeError(
		f"No available port found in range {start_port}-{start_port + max_attempts - 1}."
	)


if __name__ == "__main__":
	host = os.getenv("APP_HOST", "127.0.0.1")
	requested_port = int(os.getenv("APP_PORT", "8000"))
	devmode = os.getenv("APP_DEV", "false").lower() in {"1", "true", "yes", "on"}
	port = _find_available_port(host, requested_port)

	if port != requested_port:
		print(f"Port {requested_port} is busy. Falling back to available port {port}.")

	uvicorn.run(
		app,
		host=host,
		port=port,
		reload=devmode,
		log_level="info",
	)

